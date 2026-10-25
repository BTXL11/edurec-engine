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
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KG_DIR = os.path.dirname(os.path.abspath(__file__))          # <repo>/src/kg
REPO_ROOT = os.path.dirname(os.path.dirname(KG_DIR))        # <repo>
MODEL_DIR = os.path.join(REPO_ROOT, "model")                # 数据/权重
sys.path.insert(0, KG_DIR)

MIN_PAGE_RANK = 0.005
MAX_CONCEPTS = 30
MIN_NAME_LEN = 2

# Wikifier 在短文本上会抽到这些泛词/虚词，直接丢弃
STOPWORDS = {
    "的", "了", "和", "与", "或", "是", "在", "有", "为", "对", "从", "到",
    "学习", "分钟", "小时", "课程", "视频", "教程", "讲解", "句号", "顿号",
    "逗号", "连", "微", "严姓", "年的", "一个", "我们", "他们",
}

SERVICE_VERSION = "1.0"


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
    from recommender.pipeline import Resource, run

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
            "point_code": point_code,
            "title": title,
            "concepts": concepts,
            "elapsed_seconds": round(time.time() - started, 1),
        }
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    server_version = f"edurec-engine-KG/{SERVICE_VERSION}"

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("请求体为空")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"请求体不是合法 JSON: {exc}") from exc

    def log_message(self, fmt: str, *args) -> None:
        print(f"[kg-server] {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send(200, {"status": "ok", "engine": "edurec-engine",
                             "version": SERVICE_VERSION})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        started = time.time()
        try:
            body = self._read_json()
            path = self.path.rstrip("/")

            if path == "/knowledge/generate":
                result = generate(
                    title=body.get("title", ""),
                    description=body.get("description", ""),
                    point_code=body.get("point_code", ""),
                )
                self._send(200, result)
                return

            if path == "/knowledge/generate_batch":
                items = body.get("items") or []
                if not isinstance(items, list) or not items:
                    raise ValueError("items 必须是非空数组")
                results = []
                for item in items:
                    try:
                        results.append(generate(
                            title=item.get("title", ""),
                            description=item.get("description", ""),
                            point_code=item.get("point_code", ""),
                        ))
                    except Exception as exc:  # noqa: BLE001
                        results.append({
                            "point_code": item.get("point_code", ""),
                            "title": item.get("title", ""),
                            "concepts": [],
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                self._send(200, {"results": results})
                return

            self._send(404, {"error": "not found"})
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            if self.path.rstrip("/") != "/health":
                print(f"[kg-server] {self.path} {time.time() - started:.1f}s", flush=True)


def main() -> int:
    global MIN_PAGE_RANK, MAX_CONCEPTS

    parser = argparse.ArgumentParser(description="edurec-engine 知识生成 HTTP 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--threshold", type=float, default=MIN_PAGE_RANK)
    parser.add_argument("--max-concepts", type=int, default=MAX_CONCEPTS)
    args = parser.parse_args()

    MIN_PAGE_RANK = args.threshold
    MAX_CONCEPTS = args.max_concepts

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[kg-server] listening on http://{args.host}:{args.port} "
          f"(threshold={MIN_PAGE_RANK}, max_concepts={MAX_CONCEPTS})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[kg-server] stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
