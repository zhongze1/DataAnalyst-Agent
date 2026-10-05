"""DA-006 安全不变量测试。

要锁定的不是「prompt 文案写得对」，而是两条硬不变��：

1. **模型无法读取服务器上的任何文件** —— 无论代码怎么写；
2. **系统提示词不含任何可用于定位文件的信息** —— 没有路径、没有用户名。

第 2 条不是锦上添花：提示词里一旦出现绝对路径，模型就会想办法去读它，
而 pandas 自带的 read_* 接口曾经是敞开的。
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path

import pandas as pd
import pytest

from app.agent.prompts import build_system_prompt
from app.tools.base import ToolContext

# ============================================================ 不变量 1：文件不可达
ESCAPES = [
    # pandas 读入口
    "pd.read_excel(r'D:\\secret.xlsx')",
    "pd.read_csv('C:/data/secret.csv')",
    "pd.read_json('/etc/passwd')",
    "pd.read_parquet('x.parquet')",
    "pd.read_pickle('x.pkl')",
    "pd.read_feather('x.feather')",
    "pd.read_html('http://evil')",
    "pd.read_sql('select 1', conn)",
    "pd.read_stata('x.dta')",
    "pd.read_orc('x.orc')",
    # 绕过命名空间
    "pd.io.parsers.read_csv('x.csv')",
    "pd.io.common.file_exists('x')",
    # 写盘（同样是数据外泄）
    "df.to_csv('C:/tmp/leak.csv')",
    "df.to_excel('x.xlsx')",
    "df.to_json('x.json')",
    "df.to_pickle('x.pkl')",
    "df.to_parquet('x.parquet')",
    "df.to_sql('t', conn)",
    "df.to_hdf('x.h5', key='k')",
    "df.to_stata('x.dta')",
    "df.to_html('x.html')",
    # 经由其他对象与方法链
    "pd.Series([1]).to_csv('x.csv')",
    "df['a'].to_frame().to_csv('x.csv')",
    "df.copy().to_csv('x.csv')",
    # numpy 文件 I/O
    "np.load('x.npy')",
    "np.save('x.npy', arr)",
    "np.loadtxt('x.txt')",
    "np.genfromtxt('x.txt')",
    "np.memmap('x.bin')",
    "np.fromfile('x.bin', dtype=float)",
]


@pytest.fixture
def tiny_ctx() -> ToolContext:
    # 需要 1 个类别列 + 2 个数值列，groupby().agg(['sum','mean']) 才有对象可聚合
    frame = pd.DataFrame(
        {"g": ["x", "y", "x"], "a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
    )
    return ToolContext(session_id="t", sheets=["S"], frames={"S": frame})


@pytest.mark.parametrize("code", ESCAPES, ids=range(len(ESCAPES)))
def test_sandbox_blocks_every_file_io_path(registry, tiny_ctx: ToolContext, code: str) -> None:
    ok, out = registry.run("run_pandas", {"code": code}, tiny_ctx)
    assert ok is False, f"沙箱漏洞：{code} 竟然执行成功"
    assert "文件系统" in out or "import" in out


@pytest.mark.parametrize(
    "code",
    [
        "df.shape",
        "df['a'].sum()",
        "df.groupby('g')[['a', 'b']].agg(['sum', 'mean'])",
        "df['a'].rolling(2).mean()",
        "df.describe()",
        "df.to_string()",          # 渲染到 stdout，不是写文件
        "df.to_dict()",            # 纯转换
        "df.to_numpy()",
        "pd.to_numeric(df['a'])",  # 纯转换
        "pd.to_datetime(['2025-01-01'])",
        "out = df[['a', 'b']].corr()",
        "print(df.head())",
    ],
)
def test_sandbox_does_not_block_legitimate_analysis(
    registry, tiny_ctx: ToolContext, code: str
) -> None:
    """封堵 I/O 不能变成「什么都跑不了」，纯计算必须放行。"""
    ok, out = registry.run("run_pandas", {"code": code}, tiny_ctx)
    assert ok, f"误杀：{code} -> {out.splitlines()[-1] if out else ''}"


def test_sandbox_rejects_even_when_path_is_known(registry, session) -> None:
    """组合不变量：即便路径以某种方式泄漏给了模型，它依然读不到文件。"""
    leaked_path = str(session.path)
    assert Path(leaked_path).exists(), "前提：该文件确实存在"

    ctx = ToolContext(
        session_id=session.session_id,
        file_path=leaked_path,
        sheets=list(session.sheets),
    )
    for code in (
        f"pd.read_excel({leaked_path!r})",
        f"open({leaked_path!r})",
        f"pd.read_csv({leaked_path!r})",
    ):
        ok, _ = registry.run("run_pandas", {"code": code}, ctx)
        assert ok is False, f"已知路径仍可读取文件：{code}"


# ============================================================ 不变量 2：提示词不泄露
def test_system_prompt_contains_no_filesystem_path(session) -> None:
    prompt = build_system_prompt(session, ["load_sheet"])
    assert str(session.path) not in prompt
    assert str(session.path.parent) not in prompt
    assert "uploads" not in prompt
    # Windows 盘符 / 绝对路径形态
    assert ":\\" not in prompt
    assert ":/" not in prompt
    assert str(Path.home()) not in prompt


def test_system_prompt_contains_no_local_username(session) -> None:
    prompt = build_system_prompt(session, ["load_sheet"])
    try:
        username = getpass.getuser()
    except Exception:  # pragma: no cover - 容器环境可能没有用户概念
        pytest.skip("当前环境无法获取用户名")
    if username and len(username) > 2:
        assert username not in prompt


def test_system_prompt_does_not_teach_blocked_apis(session) -> None:
    """提示词不能教模型使用已被封禁的接口，否则它会反复尝试。"""
    prompt = build_system_prompt(session, ["load_sheet", "run_pandas"])
    for forbidden in ("read_excel", "read_csv", "read_parquet", "pd.read_"):
        assert forbidden not in prompt


def test_system_prompt_still_tells_model_how_to_access_data(session) -> None:
    """去掉路径不等于去掉引导：必须明确告诉模型唯一的取数方式。"""
    prompt = build_system_prompt(session, ["load_sheet", "run_pandas"])
    assert session.filename in prompt
    for sheet in session.sheets:
        assert sheet in prompt
    assert "load_sheet" in prompt
    assert "df" in prompt


def test_system_prompt_without_session_has_no_leak() -> None:
    prompt = build_system_prompt(None, ["list_sheets"])
    assert "上传" in prompt
    assert ":\\" not in prompt
    assert os.sep + "data" not in prompt


# ============================================================ 提示词本身无密钥
def test_prompt_builder_has_no_access_to_secrets() -> None:
    """结构性保证：build_system_prompt 只接收 session 与工具名，拿不到任何配置。"""
    import inspect

    from app.agent import prompts

    params = set(inspect.signature(prompts.build_system_prompt).parameters)
    assert params == {"session", "available_tools"}
    source = inspect.getsource(prompts)
    for secret_name in ("llm_api_key", "get_settings", "os.environ", "API_KEY"):
        assert secret_name not in source
