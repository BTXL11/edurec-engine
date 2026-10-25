"""知识生成服务入口：代码在 src/kg/server.py，数据/权重在 model/。

用法（仓库根目录）：

    python scripts/kg_server.py --host 127.0.0.1 --port 8090
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG_DIR = os.path.join(REPO_ROOT, "src", "kg")
sys.path.insert(0, KG_DIR)

from server import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
