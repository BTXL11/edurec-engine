"""知识生成服务实现：把 src/kg/generate.py 的「标题/描述 → 概念图」能力暴露成 HTTP 接口。

平台（edurec-platform）在搜索或知识点详情里实时调用，拿到概念后落库展示；
生成算法、Wikifier 调用与模型依赖都留在 engine 侧，平台不直接执行 Python。

启动（仓库根目录，推荐通过入口脚本）：

    python scripts/kg_server.py --host 127.0.0.1 --port 8090

接口：

    GET  /health
    POST /knowledge/generate        {"title","description","point_code"}
    POST /knowledge/generate_batch  {"items":[...]}

数据与权重在 model/（不入库），算法代码在 src/kg/。
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KG_DIR = os.path.dirname(os.path.abspath(__file__))          # <repo>/src/kg
REPO_ROOT = os.path.dirname(os.path.dirname(KG_DIR))        # <repo>
MODEL_DIR = os.path.join(REPO_ROOT, "model")                # 数据/权重

MIN_PAGE_RANK = 0.005
MAX_CONCEPTS = 30
MIN_NAME_LEN = 2

# Wikifier 在短文本上会抽到这些泛词/虚词，直接丢弃
STOPWORDS = {
    "的", "了", "和", "与", "或", "是", "在", "有", "为", "对", "从", "到",
    "学习", "分钟", "小时", "课程", "视频", "教程", "讲解", "句号", "顿号",
    "逗号", "连", "微", "严姓", "年的", "一个", "我们", "他们",
}

SERVICE_VERSION = "1.1"
CONTRACT_VERSION = 1

# 生产环境的默认边界。都可通过 CLI / 环境变量覆盖。
DEFAULT_MAX_BODY_BYTES = 1 << 20      # 1 MiB：概念生成请求只含标题/描述
DEFAULT_MAX_BATCH_ITEMS = 50          # 批量入口的硬上限，防止单请求打满 CPU
DEFAULT_MAX_CONCURRENCY = 4           # 同时执行的生成任务数

# 健康与就绪是两件事：
#   /health  进程活着（liveness）；
#   /ready   可以接业务请求（readiness）——生产应指向一个模型/权重就绪标记文件。
DEFAULT_READY_FILE = ""


@dataclass(frozen=True)
class ServiceConfig:
    """服务的运行参数。集中放这里，避免散落的模块级全局变量。"""

    host: str = "127.0.0.1"
    port: int = 8090
    token: str = ""                    # 非空时要求 Authorization: Bearer <token>
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    max_batch_items: int = DEFAULT_MAX_BATCH_ITEMS
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    min_page_rank: float = MIN_PAGE_RANK
    max_concepts: int = MAX_CONCEPTS
    ready_file: str = DEFAULT_READY_FILE


class ClientError(Exception):
    """请求侧错误；status 决定返回码。"""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class KGServer(ThreadingHTTPServer):
    """带并发上限的 HTTP 服务。

    `ThreadingHTTPServer` 默认每次请求起一个线程；生成任务是 CPU/IO 重活，
    必须限制同时执行数，否则一次批量请求就能把机器打满。
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, config: ServiceConfig):
        super().__init__(address, handler)
        self.config = config
        self.semaphore = threading.BoundedSemaphore(max(1, config.max_concurrency))


def _clean(name: str) -> str:
    return " ".join(str(name or "").split()).strip()


def filter_concepts(raw_concepts, min_rank: float, max_concepts: int):
    seen = set()
    out = []
    for item in raw_concepts or []:
        name = _clean(item.get("concept", ""))
        try:
            rank = float(item.get("pageRank", 0))
        except (TypeError, ValueError):
            continue
        if len(name) < MIN_NAME_LEN or name in STOPWORDS:
            continue
        if rank < min_rank:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "page_rank": round(rank, 6)})
    out.sort(key=lambda item: -item["page_rank"])
    return out[:max_concepts]


def generate(title: str, description: str = "", point_code: str = "",
             min_rank: float = MIN_PAGE_RANK, max_concepts: int = MAX_CONCEPTS) -> dict:
    """跑一次流水线，只取概念（嵌入结果丢弃，避免 BERT 加载与落盘开销）。"""
    from kg.recommender.pipeline import Resource, run

    title = _clean(title)
    if not title:
        raise ValueError("title 不能为空")

    out_dir = tempfile.mkdtemp(prefix="kg-server-")
    started = time.time()
    try:
        report = run(
            [Resource(title=title, description=_clean(description))],
            methods=["tfidf"],
            out_dir=out_dir,
            verbose=False,
        )
        item = (report.get("items") or [{}])[0]
        raw = ((item.get("graph") or {}).get("concepts")) or []
        concepts = filter_concepts(raw, min_rank, max_concepts)
        return {
            "contract_version": CONTRACT_VERSION,
            "point_code": point_code,
            "title": title,
            "concepts": concepts,
            "elapsed_seconds": round(time.time() - started, 1),
        }
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    server_version = f"edurec-engine-KG/{SERVICE_VERSION}"

    def _send(self, status: int, payload: dict, request_id: str = "") -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Contract-Version", str(CONTRACT_VERSION))
        if request_id:
            self.send_header("X-Request-ID", request_id)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, request_id: str = "") -> None:
        self._send(status, {"contract_version": CONTRACT_VERSION,
                            "error": message}, request_id)

    def _request_id(self) -> str:
        raw = (self.headers.get("X-Request-ID") or "").strip()
        return raw[:64] if raw else uuid.uuid4().hex

    def _authorized(self) -> bool:
        token = self.server.config.token
        if not token:
            return True
        header = self.headers.get("Authorization", "")
        return hmac.compare_digest(header, f"Bearer {token}")

    def _read_json(self) -> dict:
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type and content_type != "application/json":
            raise ClientError(415, "Content-Type 必须是 application/json")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ClientError(400, "Content-Length 不是合法整数") from exc
        if length <= 0:
            raise ClientError(400, "请求体为空")
        if length > self.server.config.max_body_bytes:
            raise ClientError(
                413, f"请求体超过上限 {self.server.config.max_body_bytes} 字节")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ClientError(400, f"请求体不是合法 JSON: {exc}") from exc

    def log_message(self, fmt: str, *args) -> None:
        print(json.dumps({"event": "http", "client": self.address_string(),
                          "message": fmt % args}, ensure_ascii=False), flush=True)

    def _readiness(self) -> tuple[bool, str]:
        ready_file = self.server.config.ready_file
        if not ready_file:
            return True, ""
        if os.path.isfile(ready_file):
            return True, ""
        return False, f"ready_file 不存在: {ready_file}"

    def do_GET(self) -> None:  # noqa: N802
        request_id = self._request_id()
        path = self.path.rstrip("/")
        if path == "/health":
            self._send(200, {"status": "ok", "engine": "edurec-engine",
                             "version": SERVICE_VERSION,
                             "contract_version": CONTRACT_VERSION,
                             "capabilities": ["knowledge/generate",
                                              "knowledge/generate_batch"]},
                       request_id)
            return
        if path == "/ready":
            ready, reason = self._readiness()
            payload = {"status": "ready" if ready else "not_ready",
                       "contract_version": CONTRACT_VERSION}
            if reason:
                payload["reason"] = reason
            self._send(200 if ready else 503, payload, request_id)
            return
        self._error(404, "not found", request_id)

    def _handle_post(self, request_id: str) -> None:
        body = self._read_json()
        path = self.path.rstrip("/")
        cfg = self.server.config

        if path == "/knowledge/generate":
            result = generate(
                title=body.get("title", ""),
                description=body.get("description", ""),
                point_code=body.get("point_code", ""),
                min_rank=cfg.min_page_rank,
                max_concepts=cfg.max_concepts,
            )
            self._send(200, result, request_id)
            return

        if path == "/knowledge/generate_batch":
            items = body.get("items")
            if not isinstance(items, list) or not items:
                raise ClientError(400, "items 必须是非空数组")
            if len(items) > cfg.max_batch_items:
                raise ClientError(
                    400, f"items 超过上限 {cfg.max_batch_items}（收到 {len(items)}）")
            results = []
            for item in items:
                if not isinstance(item, dict):
                    results.append({"point_code": "", "title": "", "concepts": [],
                                    "error": "item 必须是 JSON 对象"})
                    continue
                try:
                    results.append(generate(
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                        point_code=item.get("point_code", ""),
                        min_rank=cfg.min_page_rank,
                        max_concepts=cfg.max_concepts,
                    ))
                except Exception as exc:  # noqa: BLE001
                    results.append({
                        "point_code": item.get("point_code", ""),
                        "title": item.get("title", ""),
                        "concepts": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            self._send(200, {"contract_version": CONTRACT_VERSION,
                             "results": results}, request_id)
            return

        raise ClientError(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        request_id = self._request_id()
        started = time.time()
        if not self._authorized():
            self._error(401, "unauthorized", request_id)
            return
        if not self.server.semaphore.acquire(blocking=False):
            self._error(429, "too many concurrent requests", request_id)
            return
        try:
            self._handle_post(request_id)
        except ClientError as exc:
            self._error(exc.status, str(exc), request_id)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._error(500, f"{type(exc).__name__}: {exc}", request_id)
        finally:
            self.server.semaphore.release()
            print(json.dumps({
                "event": "request",
                "request_id": request_id,
                "path": self.path,
                "duration_ms": round((time.time() - started) * 1000, 1),
            }, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="edurec-engine 知识生成 HTTP 服务")
    parser.add_argument("--host", default=os.environ.get("KG_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("KG_PORT", "8090")))
    parser.add_argument("--token", default=os.environ.get("KG_SERVICE_TOKEN", ""),
                        help="非空时要求 Authorization: Bearer <token>")
    parser.add_argument("--threshold", type=float, default=MIN_PAGE_RANK)
    parser.add_argument("--max-concepts", type=int, default=MAX_CONCEPTS)
    parser.add_argument("--max-body-bytes", type=int,
                        default=int(os.environ.get("KG_MAX_BODY_BYTES",
                                                   DEFAULT_MAX_BODY_BYTES)))
    parser.add_argument("--max-batch-items", type=int,
                        default=int(os.environ.get("KG_MAX_BATCH_ITEMS",
                                                   DEFAULT_MAX_BATCH_ITEMS)))
    parser.add_argument("--max-concurrency", type=int,
                        default=int(os.environ.get("KG_MAX_CONCURRENCY",
                                                   DEFAULT_MAX_CONCURRENCY)))
    parser.add_argument("--ready-file", default=os.environ.get("KG_READY_FILE", ""),
                        help="readiness 检查的文件；生产建议指向模型就绪标记")
    args = parser.parse_args()

    config = ServiceConfig(
        host=args.host,
        port=args.port,
        token=args.token,
        max_body_bytes=max(1024, args.max_body_bytes),
        max_batch_items=max(1, args.max_batch_items),
        max_concurrency=max(1, args.max_concurrency),
        min_page_rank=args.threshold,
        max_concepts=max(1, args.max_concepts),
        ready_file=args.ready_file,
    )

    if config.host not in ("127.0.0.1", "localhost") and not config.token:
        print("[kg-server] 警告: 监听非回环地址但未设置 token，"
              "生产环境必须启用鉴权或仅在内网暴露", flush=True)

    server = KGServer((config.host, config.port), Handler, config)
    print(f"[kg-server] listening on http://{config.host}:{config.port} "
          f"(threshold={config.min_page_rank}, max_concepts={config.max_concepts}, "
          f"max_body={config.max_body_bytes}, max_batch={config.max_batch_items}, "
          f"max_concurrency={config.max_concurrency}, auth={'on' if config.token else 'off'})",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[kg-server] stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
