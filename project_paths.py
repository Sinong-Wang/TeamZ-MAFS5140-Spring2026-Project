"""
仓库根目录（与 main.py、strategy*.py 同级）。换机器、换盘符时路径仍正确。

其它模块请使用: `from project_paths import project_root`
Notebook 请在仓库根目录启动内核（或 cwd 指向仓库根），以便 `import project_paths`。
"""
from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent
