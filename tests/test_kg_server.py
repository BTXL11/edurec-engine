"""KG HTTP 服务的边界测试：鉴权、限流/限额、就绪探针与请求 ID。

这些约束是「engine 从纯离线变成带在线服务」后的工程边界；测试不依赖真实
Wikifier/模型——`generate` 被替换成确定性假实现。
"""

from __future__ import annotations

import http.client
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from kg import server  # noqa: E402


@pytest.fixture
def start_server(monkeypatch):
    servers = []

    def fake_generate(**kwargs):
        if kwargs.get("title") == "bad":
            raise ValueError("boom")
        return {
            "contract_version": 1,
            "point_code": kwargs.get("point_code", ""),
            "title": kwargs.get("title", ""),
            "concepts": [{"name": "积分", "page_rank": 0.045}],
            "elapsed_seconds": 0.0,
        }

    monkeypatch.setattr(server, "generate", fake_generate)

    def _start(**overrides):
        config = server.ServiceConfig(**overrides)
        httpd = server.KGServer(("127.0.0.1", 0), server.Handler, config)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        servers.append(httpd)
        return httpd

    yield _start

    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


def _request(httpd, method: str, path: str, payload=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
    body = None
    all_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        all_headers.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=body, headers=all_headers)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    parsed = json.loads(raw) if raw else {}
    return resp, parsed


def test_health_and_readiness(start_server):
    httpd = start_server()
    resp, body = _request(httpd, "GET", "/health")
    assert resp.status == 200
    assert body["contract_version"] == 1
    assert "knowledge/generate" in body["capabilities"]
    assert resp.getheader("X-Contract-Version") == "1"

    resp, body = _request(httpd, "GET", "/ready")
    assert resp.status == 200 and body["status"] == "ready"

    httpd2 = start_server(ready_file="/definitely/not/here")
    resp, body = _request(httpd2, "GET", "/ready")
    assert resp.status == 503
    assert body["status"] == "not_ready"
    assert "ready_file" in body["reason"]


def test_auth_and_request_id(start_server):
    httpd = start_server(token="secret")

    resp, _ = _request(httpd, "POST", "/knowledge/generate", {"title": "数学分析"})
    assert resp.status == 401

    resp, _ = _request(
        httpd, "POST", "/knowledge/generate", {"title": "数学分析"},
        {"Authorization": "Bearer wrong"},
    )
    assert resp.status == 401

    resp, body = _request(
        httpd, "POST", "/knowledge/generate", {"title": "数学分析"},
        {"Authorization": "Bearer secret", "X-Request-ID": "req-123"},
    )
    assert resp.status == 200
    assert body["concepts"][0]["name"] == "积分"
    assert resp.getheader("X-Request-ID") == "req-123"


def test_body_size_limit(start_server):
    httpd = start_server(max_body_bytes=64)
    resp, _ = _request(
        httpd, "POST", "/knowledge/generate", {"title": "x" * 200})
    assert resp.status == 413


def test_batch_item_limit(start_server):
    httpd = start_server(max_batch_items=2)
    resp, _ = _request(
        httpd, "POST", "/knowledge/generate_batch",
        {"items": [{"title": "a"}, {"title": "b"}, {"title": "c"}]},
    )
    assert resp.status == 400


def test_batch_partial_failure_does_not_abort(start_server):
    httpd = start_server()
    resp, body = _request(
        httpd, "POST", "/knowledge/generate_batch",
        {"items": [{"title": "good"}, {"title": "bad"}]},
    )
    assert resp.status == 200
    assert body["contract_version"] == 1
    assert body["results"][0]["concepts"][0]["name"] == "积分"
    assert "boom" in body["results"][1]["error"]
