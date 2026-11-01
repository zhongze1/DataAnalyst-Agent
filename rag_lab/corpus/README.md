# DataAnalyst Agent · 项目说明

> 本文是 RAG 实验语料之一，介绍**主工程 DataAnalyst Agent**（不是 rag_lab 自己）。
> 内容依据仓库中的真实代码与文档编写，凡无法从代码确认的地方均标注「待确认」。

## 1. 项目用途

**DataAnalyst Agent** 是一个"上传 Excel → 用自然语言提问 → 由 Agent 自主调用工具分析并给出结论"的
数据分析服务。它解决的核心问题不是"能不能算出数"，而是：

- **可信**：结论里的每个数字都来自真实执行过的工具调用，而不是模型凭印象编的；
- **可追溯**：用户能看到 Agent 调了哪些工具、传了什么参数、拿到了什么结果；
- **可交付**：分析结论能导出成 Markdown / Excel 报告，连同数据明细一起发给同事。

面向的典型场景：业务人员手里有 Excel，想快速得到"这个月什么情况、哪里有问题、为什么"的结论，
但不想写代码、也不想为了一个临时问题去搭一套 BI。

## 2. 主要能力

| 能力 | 说明 | 关键实现位置 |
| --- | --- | --- |
| 单 Agent Loop | 一次提问 = 一整轮「模型思考 → 调工具 → 观察结果 → 继续或收尾」，循环上限可配置 | `app/agent/loop.py` |
| 11 个分析工具 | 见下方工具清单，全部为真实 pandas 计算，非模型生成 | `app/tools/pandas_tools.py`、`app/tools/find_tools.py`、`app/tools/knowledge_tools.py` |
| 代码沙箱 | `run_pandas` 用 AST **黑名单**执行模型写的 pandas 代码：拦截 `import`、dunder、常见文件 I/O 属性。注意它**不是安全边界**（`pd.eval` 字符串表达式等可绕过），只适用于可信提问者场景，详见根目录 `SECURITY.md` | `app/tools/pandas_tools.py` |
| 流式输出 | SSE 逐事件推送（工具开始/完成、答案增量、收尾快照），支持"停止生成" | `app/routers/chat_stream.py`、`app/agent/events.py` |
| 会话持久化 | 对话与元数据落 SQLite，进程重启可恢复；上传的 Excel 留在磁盘供 pandas 读取 | `app/storage/session_store.py`、`app/storage/sqlite_repo.py` |
| 报告导出 | 一键导出 Markdown / Excel（结论 + 执行轨迹 + 数据明细 sheet） | `app/services/export_service.py`、`app/routers/export.py` |
| 结构化产物 | `run_pandas` 返回的 DataFrame 另存为 CSV + manifest，供导出使用 | `app/storage/artifacts.py` |
| 用户体系 | 口令 PBKDF2 哈希、服务端登录会话、会话归属隔离（非本人会话按"不存在"返回 404） | `app/services/auth_service.py`、`app/routers/auth.py` |
| 统一错误响应 | 所有错误收敛为 `{"error": {code, message, request_id}}`，内部异常细节只进日志 | `app/errors.py`、`app/main.py` |
| 结构化日志 | 支持 text / json 两种格式，请求级 `request_id` 自动贯穿，响应头回写 `X-Request-ID` | `app/logging_setup.py`、`app/middleware.py` |
| 工具调用审计 | 每次工具调用落 JSONL（只记"形状"不记内容），供事后排查 | `app/audit.py` |
| 敏感字段打码 | 界面 / 导出 / 产物三处对手机号、身份证、银行卡、邮箱打码；**喂给模型的仍是原文** | `app/security/redact.py` |
| 配额与并发保护 | 请求次数硬配额（用户 / 会话）+ 用户级与全局并发上限，超限返回 429 友好提示 | `app/services/quota_service.py` |

### 工具清单（名称即模型可调用的函数名）

| 工具 | 作用 |
| --- | --- |
| `list_sheets` | 列出工作表 |
| `load_sheet` | 把工作表读成 DataFrame（切换 `df` 指向） |
| `data_health_check` | 整表体检（类型、缺失率、唯一值、IQR 离群点、偏度、重复行、告警清单） |
| `preview_rows` | 查看前 N 行 |
| `column_profile` | 逐列画像 |
| `value_counts` | 单列取值分布 |
| `describe_dataframe` | 数值列描述统计 |
| `correlation` | 相关性矩阵 |
| `run_pandas` | 沙箱内执行任意 pandas 代码（兜底万能工具） |
| `find_value` | 存在性检查：某个值/关键词出现在哪些列、共多少行 |

## 3. 架构概览

```
浏览器（无构建的原生 JS 前端）
   │  fetch / SSE
   ▼
FastAPI 应用（app/main.py: create_app）
   ├── RequestContextMiddleware  生成/透传 request_id、写访问日志
   ├── 全局异常处理器            统一错误体 + Retry-After
   ├── 静态资源 / 登录页
   └── 路由层
         ├── /api/auth/*       登录、用户管理
         ├── /api/files/*      上传、查询、删除会话
         ├── /api/chat         一次性问答
         ├── /api/chat/stream  SSE 流式问答
         ├── /api/sessions/*   历史会话、报告导出
         └── /api/health       健康检查（含 provider / model / max_steps）
   ▼
Agent 编排（app/agent/loop.py: DataAnalystAgent）
   ├── 系统提示词（app/agent/prompts.py）
   ├── LLM 客户端（app/llm/：OpenAI 兼容实现 / 离线 Mock）
   └── 工具注册表（app/tools/registry.py: ToolRegistry）
   ▼
工具执行层（app/tools/）—— pandas 计算、沙箱校验、产物落盘
   ▼
存储层
   ├── SQLite data/agent.db（sessions / messages / quota_daily_usage）
   ├── 上传文件 data/uploads/<session_id>/
   ├── 结构化产物 …/artifacts/*.csv + manifest.json
   └── 审计日志 data/audit/<日期>.jsonl
```

配置统一走 `app/config.py` 的 `Settings`（环境变量前缀 `AGENT_`，支持 `.env`），
启动时做 fail-fast 校验；`AGENT_LLM_PROVIDER=mock` 可在**不联网、无 API Key**的情况下跑通整条链路。

## 4. 当前状态

- **可运行**：`uv run uvicorn main:app --reload`（`main.py` 只是 `app.main:app` 的入口）。
  mock 模式离线可跑；接真实模型需配置 `AGENT_LLM_API_KEY` / `AGENT_LLM_BASE_URL` / `AGENT_LLM_MODEL`。
- **测试**：仓库 `tests/` 下共 23 个测试文件。
  最近一次全量实测（DA-015 提交时）：`488 passed / 0 failed`，覆盖率 92.78%（门槛 70%）。
  这是**带日期的快照**，不是持续保证；复跑命令：`uv run python -m pytest -q`。
- **部署假设**：单进程 + 单文件 SQLite。限流的分钟窗口与并发占用是**进程内状态**，
  多 worker 部署下会按 worker 数量放大，项目**不声称**支持多 worker 的全局一致限流。
- **数据规模**：`data/samples/` 提供 5 套业务测试数据集（SaaS 订阅、连锁门店、广告投放、
  客服工单、电商订单），固定随机种子生成、可复现，用于人工核对 Agent 回答是否正确。

### 已知限制（详见对应文档）

- 流式输出期间无法对增量文本做打码，落定后会被脱敏版整体替换（见 `docs/DA-015-敏感字段打码.md`）。
- 每日配额计数写入失败时会拒绝请求（fail-closed）；用量结算失败只记日志。
- RFM 等指标的基准日与分层阈值在测试数据基线中**未定义**（见 `指标定义.md`），因此分层结果不可跨分析直接复现。
- 「待确认」项：前端是否存在显式的"查看分析过程"折叠开关、是否存在全局额度仪表盘
  —— 这两点未在本次核查中确认。
