"""RAG-01 冒烟测试：只验证隔离环境可用，不涉及任何检索算法。

三条不变量：
1. 实验包能被 import（证明 src/ 布局与 pytest 配置可用）；
2. 解释器版本满足要求；
3. **实验代码不依赖主工程**（不出现 `import app.*`）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SNAKE = Path(__file__).resolve().parents[1]

_MAIN_APP_IMPORT = re.compile(r"^\s*(from|import)\s+app(\.|\s|$)", re.MULTILINE)


def test_package_is_importable() -> None:
    import rag_lab

    assert rag_lab.__doc__
    assert rag_lab.__version__


def test_python_version_is_supported() -> None:
    assert sys.version_info >= (3, 12)


def test_lab_never_imports_the_main_app() -> None:
    """隔离约束：实验代码不得依赖主工程（``app.*``）。

    这比"记得别 import"可靠 —— 一旦有人图省事写
    ``from app.config import ...``，这条测试立刻变红。
    """
    offenders = [
        path.relative_to(SNAKE).as_posix()
        for path in (SNAKE / "src").rglob("*.py")
        if _MAIN_APP_IMPORT.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
