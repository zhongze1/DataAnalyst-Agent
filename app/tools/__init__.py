"""工具层。

导入本包会自动注册 ``app.tools.pandas_tools`` 中的全部工具。

``knowledge_search``（DA-018）走**显式注册**而不是 ``@tool`` 装饰器：
``AGENT_KNOWLEDGE_ENABLED=false`` 时它必须"连 schema 都不出现"，
而装饰器只能在定义处无条件执行，做不到这一点。
"""

from app.tools.registry import ToolRegistry, registry, tool
from app.tools import find_tools as _find_tools  # noqa: F401  仅为触发注册
from app.tools import pandas_tools as _pandas_tools  # noqa: F401  仅为触发注册
from app.tools.knowledge_tools import register_knowledge_tool as _register_knowledge

# 关闭知识库开关时这里返回 None（不注册）；注册失败不得冒泡 —— 它跑在 import 期
_register_knowledge(registry)

__all__ = ["ToolRegistry", "registry", "tool"]
