"""交接契约 v1 校验器（仅标准库）。

权威规范见 `docs/contracts/platform-contract.md`；本文件把其中的硬约束变成可执行检查，
供两侧测试与手工排障共用。

用法：

    python contracts/v1/validate.py snapshot <snapshot-dir>
    python contracts/v1/validate.py recommendations <file> [--envelope <file>]
    python contracts/v1/validate.py envelope <file>
    python contracts/v1/validate.py all --snapshot <dir> --recommendations <file>

退出码：0 合法；1 违规；2 用法错误。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS_PATH = os.path.join(HERE, "snapshot.columns.json")

BOM = b"\xef\xbb\xbf"
RUN_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}$")
DECIMAL2_RE = re.compile(r"^[0-9]+\.[0-9]{2}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(Exception):
    """保留给调用方做显式失败；校验函数本身返回错误列表。"""


# --------------------------------------------------------------------------
# 基础读取
# --------------------------------------------------------------------------

def _read_bytes(path: str, errors: list[str]) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as exc:
        errors.append(f"无法读取 {path}: {exc}")
        return None


def _decode_text(path: str, errors: list[str]) -> str | None:
    data = _read_bytes(path, errors)
    if data is None:
        return None
    if data.startswith(BOM):
        errors.append(f"{os.path.basename(path)}: 带 UTF-8 BOM（必须无 BOM）")
        data = data[len(BOM):]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"{os.path.basename(path)}: 不是合法 UTF-8: {exc}")
        return None


def _load_json(path: str, errors: list[str]):
    text = _decode_text(path, errors)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        errors.append(f"{os.path.basename(path)}: JSON 解析失败: {exc}")
        return None


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_uint(value: str) -> bool:
    return bool(value) and value.isdigit()


def _check_uint(value: str, where: str, errors: list[str]) -> None:
    if not _is_uint(value):
        errors.append(f"{where}: 期望非负整数，实际 {value!r}")


def _check_unix_seconds(value: str, where: str, errors: list[str]) -> None:
    _check_uint(value, where, errors)


def _check_decimal2(value: str, where: str, errors: list[str]) -> None:
    if not DECIMAL2_RE.match(value):
        errors.append(f"{where}: 期望两位小数（如 4.50），实际 {value!r}")


def _check_json_text(value: str, kind: str, where: str, errors: list[str]) -> None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        errors.append(f"{where}: 不是合法 JSON: {exc}")
        return
    if kind == "json_array" and not isinstance(parsed, list):
        errors.append(f"{where}: 期望 JSON 数组，实际 {type(parsed).__name__}")
    if kind == "json_object" and not isinstance(parsed, dict):
        errors.append(f"{where}: 期望 JSON 对象，实际 {type(parsed).__name__}")


# --------------------------------------------------------------------------
# 快照
# --------------------------------------------------------------------------

def _load_columns() -> dict:
    with open(COLUMNS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _validate_meta(meta, errors: list[str]) -> None:
    if not isinstance(meta, dict):
        errors.append("meta.json: 顶层必须是 JSON 对象")
        return

    if meta.get("contract_version") != 1:
        errors.append(
            f"meta.json: contract_version 必须为整数 1，实际 {meta.get('contract_version')!r}")

    run_id = meta.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        errors.append(f"meta.json: run_id 必须匹配 YYYYMMDD_HHMMSS，实际 {run_id!r}")

    for key in ("exported_at", "users_count", "resources_count",
                "behaviors_count", "ratings_count", "categories_count"):
        value = meta.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"meta.json: {key} 必须是非负整数，实际 {value!r}")

    window = meta.get("behavior_window")
    if not isinstance(window, dict) or set(("start", "end")) - set(window):
        errors.append("meta.json: behavior_window 必须含 start/end")
    else:
        for key in ("start", "end"):
            value = window[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"meta.json: behavior_window.{key} 必须是非负整数")

    files = meta.get("files_sha256")
    if not isinstance(files, dict):
        errors.append("meta.json: files_sha256 必须是对象")
    else:
        for name, digest in files.items():
            if not isinstance(digest, str) or not HEX64_RE.match(digest):
                errors.append(f"meta.json: files_sha256[{name!r}] 不是 64 位十六进制")


def _validate_csv(path: str, spec: dict, meta_count: int | None,
                  errors: list[str]) -> int:
    """校验一个 CSV，返回数据行数。"""
    name = os.path.basename(path)
    text = _decode_text(path, errors)
    if text is None:
        return 0

    rows = list(csv.reader(text.splitlines()))
    if not rows:
        errors.append(f"{name}: 文件为空（至少要有表头）")
        return 0
    header = rows[0]
    if len(set(header)) != len(header):
        errors.append(f"{name}: 表头有重复列 {header}")
    missing = [c["name"] for c in spec["columns"]
               if c.get("required") and c["name"] not in header]
    if missing:
        errors.append(f"{name}: 缺少必需列 {missing}")
        return max(0, len(rows) - 1)

    index = {column: i for i, column in enumerate(header)}
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            errors.append(f"{name}:{line_no}: 列数 {len(row)} != 表头 {len(header)}")
            continue
        for column in spec["columns"]:
            if column["name"] not in index:
                continue
            raw = row[index[column["name"]]]
            where = f"{name}:{line_no}.{column['name']}"
            kind = column.get("type")
            if kind in ("uint",):
                _check_uint(raw, where, errors)
            elif kind == "unix_seconds":
                _check_unix_seconds(raw, where, errors)
            elif kind == "decimal2":
                _check_decimal2(raw, where, errors)
            elif kind in ("json_array", "json_object"):
                _check_json_text(raw, kind, where, errors)
            if "minimum" in column and _is_uint(raw):
                if int(raw) < column["minimum"]:
                    errors.append(f"{where}: 小于最小值 {column['minimum']}")
            if "maximum" in column and _is_uint(raw):
                if int(raw) > column["maximum"]:
                    errors.append(f"{where}: 大于最大值 {column['maximum']}")
            if column["name"] == "action" and raw not in ("view", "click", "favorite"):
                errors.append(f"{where}: action 必须是 view/click/favorite，实际 {raw!r}")

    count = len(rows) - 1
    if meta_count is not None and count != meta_count:
        errors.append(f"{name}: 实际 {count} 行，meta.json 声明 {meta_count} 行")
    return count


def validate_snapshot(snapshot_dir: str) -> list[str]:
    """校验一份 platform → engine 快照目录，返回错误列表（空 = 合法）。"""
    errors: list[str] = []
    if not os.path.isdir(snapshot_dir):
        return [f"快照目录不存在: {snapshot_dir}"]

    meta_path = os.path.join(snapshot_dir, "meta.json")
    meta = _load_json(meta_path, errors) if os.path.exists(meta_path) else None
    if meta is None and not os.path.exists(meta_path):
        errors.append("meta.json: 必需文件缺失（无法校验版本与 sha256）")
    if meta is not None:
        _validate_meta(meta, errors)

    spec = _load_columns()
    counts = {
        "users_count": "users.csv",
        "resources_count": "resources.csv",
        "behaviors_count": "behaviors.csv",
        "ratings_count": "ratings.csv",
        "categories_count": "categories.csv",
    }

    for meta_key, filename in counts.items():
        file_spec = spec["files"].get(filename)
        if file_spec is None:
            continue
        path = os.path.join(snapshot_dir, filename)
        optional = not file_spec.get("required", False)
        if not os.path.exists(path):
            if not optional:
                errors.append(f"{filename}: 必需文件缺失")
            continue
        expected = None
        if isinstance(meta, dict) and isinstance(meta.get(meta_key), int):
            expected = meta[meta_key]
        _validate_csv(path, file_spec, expected, errors)

    # sha256：meta 里列出的每个文件都必须存在且一致；必需文件必须出现在清单里
    if isinstance(meta, dict) and isinstance(meta.get("files_sha256"), dict):
        listed = meta["files_sha256"]
        for filename in counts.values():
            file_spec = spec["files"].get(filename, {})
            if file_spec.get("required") and filename not in listed:
                errors.append(f"meta.json: files_sha256 缺少必需文件 {filename}")
        for filename, digest in listed.items():
            path = os.path.join(snapshot_dir, filename)
            if not os.path.exists(path):
                errors.append(f"meta.json: files_sha256 列出的 {filename} 不存在")
                continue
            actual = _sha256(path)
            if isinstance(digest, str) and actual != digest:
                errors.append(f"{filename}: sha256 不匹配（meta={digest[:12]}…, 实际={actual[:12]}…）")

    return errors


# --------------------------------------------------------------------------
# 推荐主文件 / 信封
# --------------------------------------------------------------------------

def _validate_recommendation_payload(payload, errors: list[str]) -> None:
    if not isinstance(payload, dict):
        errors.append(f"推荐主文件: 顶层必须是 JSON 对象，实际 {type(payload).__name__}")
        return
    for key, value in payload.items():
        if not isinstance(key, str) or not key.isdigit():
            errors.append(f"推荐主文件: 键必须是十进制无符号整数，实际 {key!r}")
            continue
        if len(key) > 1 and key.startswith("0"):
            errors.append(f"推荐主文件: 键 {key!r} 有前导零（会与 {int(key)} 撞号）")
        if not isinstance(value, list):
            errors.append(f"推荐主文件[{key}]: 值必须是数组，实际 {type(value).__name__}")
            continue
        for i, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, int):
                errors.append(
                    f"推荐主文件[{key}][{i}]: 必须是 JSON 整数，实际 {item!r}")
            elif item < 0:
                errors.append(f"推荐主文件[{key}][{i}]: 资源 ID 不能为负（{item}）")
        if len(set(map(repr, value))) != len(value):
            errors.append(f"推荐主文件[{key}]: 资源 ID 有重复，必须自行去重")


def validate_recommendations(path: str) -> list[str]:
    """校验 engine → platform 推荐主文件，返回错误列表（空 = 合法）。"""
    errors: list[str] = []
    payload = _load_json(path, errors)
    if payload is None and not os.path.exists(path):
        errors.append(f"推荐主文件不存在: {path}")
    elif payload is not None:
        _validate_recommendation_payload(payload, errors)
    return errors


def validate_envelope(path: str, main_payload=None) -> list[str]:
    """校验旁挂信封；`main_payload` 传入时额外校验 scores/reasons 与主文件对齐。"""
    errors: list[str] = []
    env = _load_json(path, errors)
    if not isinstance(env, dict):
        if env is not None:
            errors.append(f"信封: 顶层必须是 JSON 对象，实际 {type(env).__name__}")
        return errors

    if env.get("contract_version") != 1:
        errors.append(
            f"信封: contract_version 必须为整数 1，实际 {env.get('contract_version')!r}")

    for key in ("generated_at", "top_n", "users_count"):
        if key in env:
            value = env[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"信封: {key} 必须是非负整数，实际 {value!r}")

    scores = env.get("scores")
    if scores is not None and not isinstance(scores, dict):
        errors.append("信封: scores 必须是对象")
    elif isinstance(scores, dict):
        for uid, values in scores.items():
            if not isinstance(uid, str) or not uid.isdigit():
                errors.append(f"信封: scores 键必须是用户 ID，实际 {uid!r}")
            if not isinstance(values, list) or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in values):
                errors.append(f"信封: scores[{uid!r}] 必须是数字数组")
                continue
            if isinstance(main_payload, dict) and uid in main_payload:
                if len(values) != len(main_payload[uid]):
                    errors.append(
                        f"信封: scores[{uid!r}] 长度 {len(values)} "
                        f"与主文件 {len(main_payload[uid])} 不一致")

    reasons = env.get("reasons")
    if reasons is not None and not isinstance(reasons, dict):
        errors.append("信封: reasons 必须是对象")
    elif isinstance(reasons, dict):
        for uid, values in reasons.items():
            if not isinstance(uid, str) or not uid.isdigit():
                errors.append(f"信封: reasons 键必须是用户 ID，实际 {uid!r}")
            if not isinstance(values, list) or not all(
                    isinstance(v, str) for v in values):
                errors.append(f"信封: reasons[{uid!r}] 必须是字符串数组")
                continue
            if isinstance(main_payload, dict) and uid in main_payload:
                if len(values) != len(main_payload[uid]):
                    errors.append(
                        f"信封: reasons[{uid!r}] 长度 {len(values)} "
                        f"与主文件 {len(main_payload[uid])} 不一致")

    return errors


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _print_result(title: str, errors: list[str]) -> int:
    if not errors:
        print(f"OK: {title}")
        return 0
    print(f"FAIL: {title}（{len(errors)} 条）", file=sys.stderr)
    for err in errors[:40]:
        print(f"  - {err}", file=sys.stderr)
    if len(errors) > 40:
        print(f"  ... 其余 {len(errors) - 40} 条省略", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="edurec 交接契约 v1 校验器")
    sub = parser.add_subparsers(dest="command", required=True)

    p_snap = sub.add_parser("snapshot", help="校验快照目录")
    p_snap.add_argument("path")

    p_rec = sub.add_parser("recommendations", help="校验推荐主文件")
    p_rec.add_argument("path")
    p_rec.add_argument("--envelope", default=None)

    p_env = sub.add_parser("envelope", help="校验旁挂信封")
    p_env.add_argument("path")
    p_env.add_argument("--recommendations", default=None,
                       help="同时校验与主文件的 scores/reasons 对齐")

    p_all = sub.add_parser("all", help="校验快照 + 推荐产物")
    p_all.add_argument("--snapshot", required=True)
    p_all.add_argument("--recommendations", required=True)
    p_all.add_argument("--envelope", default=None)

    args = parser.parse_args(argv)

    if args.command == "snapshot":
        return _print_result(f"snapshot {args.path}", validate_snapshot(args.path))

    if args.command == "recommendations":
        errors = validate_recommendations(args.path)
        if args.envelope:
            main = _load_json(args.path, [])
            errors += validate_envelope(args.envelope, main)
        return _print_result(f"recommendations {args.path}", errors)

    if args.command == "envelope":
        main = _load_json(args.recommendations, []) if args.recommendations else None
        return _print_result(f"envelope {args.path}",
                             validate_envelope(args.path, main))

    if args.command == "all":
        errors = validate_snapshot(args.snapshot)
        rec_errors = validate_recommendations(args.recommendations)
        errors += rec_errors
        envelope = args.envelope
        if envelope is None:
            auto = os.path.splitext(args.recommendations)[0] + ".meta.json"
            if os.path.exists(auto):
                envelope = auto
        if envelope:
            main = _load_json(args.recommendations, [])
            errors += validate_envelope(envelope, main)
        return _print_result("snapshot + recommendations", errors)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
