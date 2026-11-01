# rag_lab · RAG 学习实验

> 独立子工程，**与主工程 `app/` 完全隔离**。
> 当前进度：**RAG-01（目录+环境+语料盘点）、RAG-02（语料与分块）、RAG-03（向量生成+缓存+错误处理）已完成**。

## 目标与原则

- 先验证**检索器能否找对内容**，再验证 **Agent 能否正确使用检索结果** ——
  否则检索错误与推理错误混在一起，极难排查。
- 因此本实验在**检索稳定前不接入主工程**。接入闸门（RAG-05 前必须同时满足）：
  1. `recall@5 ≥ 0.9`；2. 每一条召回都能一句话解释"为什么它被召回"。

## 与主工程的边界（硬约束）

1. 不修改 `app/`、根 `pyproject.toml`、`data/`；
2. 不共享虚拟环境、不共享依赖（本子工程有自己的 `pyproject.toml` + `.venv` + `uv.lock`）；
3. 不执行 `git init`，不创建嵌套 Git 仓库；
4. 真实 API Key 只放在 `.env`（已被忽略），永不写进代码或提交；
5. 实验代码禁止 `import app.*` —— 由 `tests/test_lab_smoke.py` 机械守住。

## 目录

| 路径 | 用途 | 状态 |
| --- | --- | --- |
| `corpus/` | 语料原文（6 份 Markdown） | ✅ RAG-02 |
| `src/rag_lab/chunker.py` | Markdown 分块器（纯文本，无向量） | ✅ RAG-02 |
| `chunks.jsonl` | 分块结果（73 片），一行一个 JSON 对象 | ✅ RAG-02 |
| `src/rag_lab/embedding.py` | 向量生成 + 缓存 + 错误处理（**不含检索**） | ✅ RAG-03 |
| `index/embeddings.jsonl` | 向量缓存（运行期产物，**不入库**） | ✅ RAG-03 |
| `src/rag_lab/retrieve.py` | 三种检索：向量 / 字符 2-gram BM25 / RRF 混合（**只检索不生成**） | ✅ RAG-04 |
| `src/rag_lab/evaluate.py` | recall@k / MRR 评估，写 `eval/report.md` | ✅ RAG-04 |
| `eval/questions.jsonl` | 20 道评估题（18 计分 + 2 陷阱）+ 标准答案标注 | ✅ RAG-04 |
| `eval/report.md` | 评估报告（**脚本生成，勿手改**） | ✅ RAG-04 |
| `tests/` | 123 条用例（全离线：FakeEmbedder + MockTransport） | ✅ RAG-04 |

## 环境与命令（已实测）

| 项 | 值 |
| --- | --- |
| uv / Python | `0.12.17` / `3.14.7`（独立 `.venv`，与主工程同版本但互不影响） |
| 依赖 | `httpx`（RAG-03 起）+ `pytest`；本包以 editable 方式安装 |
| 测试 | `uv run pytest -q` → **123 passed** |

```powershell
cd rag_lab
uv venv --python 3.14     # 建独立虚拟环境（本机已有 3.14，零下载）
uv sync                   # 安装依赖 + editable 安装本包
uv run pytest -q          # => 123 passed
```

> 打包方式：`[build-system] uv_build`（uv 自带，不额外下载 hatchling）。
> RAG-01 曾用 `package = false`，代价是 `python -m rag_lab.*` 找不到模块，RAG-02 已改。
>
> Python 版本：计划路线是"API 做 embedding + numpy 做检索"，3.14 够用。
> 若将来要跑 `torch`/`sentence-transformers` 这类本地模型，删掉 `.venv` 后
> `uv venv --python 3.12` 重建即可（`requires-python = ">=3.12"`，无需改代码）。

---

## RAG-02：语料与分块

### 语料（6 份，`corpus/`）

| 文件 | 来源 | 说明 |
| --- | --- | --- |
| `README.md` | **新写** | 主工程用途、10 个工具、架构概览、当前状态 |
| `架构说明.md` | **新写** | 请求链路、Agent Loop、工具注册与执行、会话存储、SSE、横切关注点 |
| `指标定义.md` | **新写** | 销售额 / 利润率 / 客单价 / RFM 的口径、公式、分母与边界（含「待确认」清单） |
| `测试指南.md` | 复制自 `data/samples/` | 5 套业务数据集的表结构 + 关键真值 + 埋入异常（**逐字节一致，有测试守着**） |
| `DA-017-配额与并发保护.md` | 复制自 `docs/` | 单张工单的完整设计记录（**逐字节一致**） |
| `DA-015-敏感字段打码.md` | 复制自 `docs/` | 结构相似 → 可检验"能否区分同类文档"（**逐字节一致**） |

排除 `.codebuddy/plans/*.md`（IDE 计划缓存，非项目文档）。

### 分块命令

```powershell
uv run python -m rag_lab.chunker                                  # 默认 1200/150 → chunks.jsonl
uv run python -m rag_lab.chunker --max-chars 400 --overlap 60 --out chunks_small.jsonl
```

### 分块策略（四步）

1. **标题优先，但标题不孤立**：标题是硬边界；**紧跟另一个标题的空标题不单独成片**，
   并入下一节（否则会产出十几个字符的碎片）。
2. **长度上限是硬约束**：新内容按 `max_chars − overlap_chars − 1` 装箱，
   重叠从预留额度里出 —— 片段长度**永不突破 `max_chars`**。
3. **重叠按整行回退，且不跨块**：只回退最后一个原子块内部的整行；
   若回退起点落在表格中段，**补上该表格的表头**。
4. **表格续片补表头**：表格按行切开时，续片开头重复原始表头，并记在 `repeated_header` 字段里。

### 分块结果（默认参数）

| 指标 | 默认（1200/150） | 对照（400/60） |
| --- | --- | --- |
| 片段总数 / 文档数 | **73 / 6** | 161 / 6 |
| 长度 min / p50 / max | 42 / 486 / **1051** | 42 / 273 / **368** |
| 表格续片（已补表头） | 2 | 15 |
| 过短片段（<40 字符） | **0** | **0** |
| 超上限片段 | **0** | **0** |
| 可疑切分 | **无** | **无** |

字段：`chunk_id / doc_id / source_path / section / text / line_start / line_end / content_hash / repeated_header`。
可追溯契约：`text == repeated_header + "\n" + 源文件行[line_start-1:line_end]`（字符硬切除外，见测试）。

---

## RAG-03：向量生成、缓存与错误处理

### 生成方式：只认"OpenAI 兼容端点"

代码走 **`POST {base_url}/embeddings`**（请求体 `{"model": ..., "input": [...]}`，
读取 `data[*].embedding` 并按 `index` 归位）。这意味着**换服务商只改配置、不改代码**。

另外内置 `provider=fake`：`sha256(文本)` 展开成固定维度的确定性向量。
它**不懂语义**，只能用来跑通链路 / 写测试 / 在没 Key 的机器上验证缓存与错误处理。
RAG-04 的检索评估**必须**换真实模型。

### 配置方式（4 个必填 + 3 个可选）

复制 `.env.example` 为 `.env`，填：

| 变量 | 说明 |
| --- | --- |
| `RAG_LAB_EMBEDDING_PROVIDER` | `fake`（默认）\| `api` |
| `RAG_LAB_EMBEDDING_BASE_URL` | api 必填，形如 `https://<服务商域名>/v1`（**不要带 `/embeddings`**） |
| `RAG_LAB_EMBEDDING_MODEL` | api 必填，向量模型名（以服务商文档为准） |
| `RAG_LAB_EMBEDDING_API_KEY` | api 必填，只放 `.env`，**不提交** |
| `RAG_LAB_EMBEDDING_DIM` | 可选，`0`=按首个响应推断；填了就校验，不符直接报错 |
| `RAG_LAB_EMBEDDING_BATCH_SIZE` / `_TIMEOUT` / `_MAX_RETRIES` | 可选，默认 **10** / 30s / 3 |

**关于"用哪家、哪个模型"：这必须由你定**（涉及账号与费用），代码不做假设。两点提示：

1. 主工程现在用的是 **DeepSeek 的 chat 接口**；但据我所知 **DeepSeek 官方平台不提供
   embeddings 端点**（请以官方文档为准）。所以 RAG-03 的现实做法是**另配一个向量服务**，
   这属于"填 `.env`"，不是改代码。
2. 先用 `--dry-run` 看会算多少条，再用 `--limit 8` 小规模试跑（只花 8 条的钱），
   确认维度与响应正常后再全量。

### 命令

```powershell
# 只看会算多少条，不调用任何服务
uv run python -m rag_lab.embedding --dry-run

# 离线跑通（假向量，用来验证链路/缓存/断点续跑）
uv run python -m rag_lab.embedding --provider fake

# 真实向量：先小规模试跑，再全量
uv run python -m rag_lab.embedding --limit 8
uv run python -m rag_lab.embedding
```

### 缓存策略

- 键 = **缓存空间 + 内容指纹**；空间 = `provider/model/dim=...`，
  所以**换模型或换维度后旧向量自动失效**，不会串用；
- **追加写、每条算完立即落盘** → 中途失败或 Ctrl-C 都不会白花钱，重跑自动续上；
- 命中缓存时**一次服务都不调**（有测试守着）。

### 错误处理策略

| 情况 | 处理 |
| --- | --- |
| 缺 Key / base_url / model，或取值非法 | `EMBEDDING_CONFIG`，**立刻中止**，不浪费重试 |
| 401 / 403 | `EMBEDDING_AUTH`，**不重试**（重试没意义，先查 Key） |
| 其余 4xx（如模型名错） | `EMBEDDING_REQUEST`，**不重试** |
| 429 / 5xx / 超时 / 连接失败 | `EMBEDDING_UNAVAILABLE`，指数退避重试 `max_retries` 次 |
| 条数不符 / 维度不符 / 非 JSON / 缺字段 | `EMBEDDING_RESPONSE`，**直接报错**，绝不"凑合着用" |
| 任一批失败 | 异常抛出，但**已完成批次已落盘**；CLI 会提示"重跑会自动续上" |

隐私：API Key 用 `field(repr=False)`，且错误消息里只有状态码与响应片段 —— 有测试断言
`key not in repr(config)` 与 `key not in str(异常)`。

### 实测（离线，`provider=fake`，73 个片段）

```
$ uv run python -m rag_lab.embedding --dry-run
空间: fake/fake-hash-v1/dim=auto
片段总数: 73     缓存已有: 0     待计算: 73（dry-run，未调用向量服务）

$ uv run python -m rag_lab.embedding --provider fake
空间: fake/fake-hash-v1/dim=auto
片段总数: 73     缓存命中: 0     本次计算: 73（3 批）     向量维度: 256
缓存文件: rag_lab/index/embeddings.jsonl（当前共 167 条，约 2.5 MB）

$ uv run python -m rag_lab.embedding --provider fake   # 再跑一次
缓存命中: 73     本次计算: 0（0 批）
```

---

## RAG-04：三种检索 + 评估

### 为什么是三种而不是一个"融合黑盒"

现在的目标是**比较**，不是调优：把三路揉成一个数字，出了问题也不知道该修哪一路。

| 方式 | 做法 | 已知弱点（评估里都现形了） |
| --- | --- | --- |
| `vector` | 查询向量与片段向量的余弦 Top-K（复用 RAG-03 缓存） | 对"用不同措辞表达的设计理由"不敏感 |
| `keyword` | **字符 2-gram + BM25**（中文不引 jieba，零新依赖） | 答案落在"续片"（表格续行/引用段）时容易掉出前 k |
| `hybrid` | 两路 **RRF** 融合（`Σ 1/(60+rank)`，无参数可调） | 只是排序融合，不产生新的召回 |

每条结果都带 `why`：向量给相似度、关键词给命中的 2-gram、混合给两路名次 ——
**"能解释为什么召回"比"分数是多少"更重要**。

### 命令

```powershell
# 三种结果分别打印（--mode 可只跑一路；keyword 完全离线，不需要向量服务）
uv run python -m rag_lab.retrieve "客单价的分母是谁？" --k 5
uv run python -m rag_lab.retrieve "SANDBOX_VIOLATION" --mode keyword

# 跑评估并写 eval/report.md（20 个 query 首跑会各调一次向量服务，之后全命中缓存）
uv run python -m rag_lab.evaluate
uv run python -m rag_lab.evaluate --limit 5     # 先小规模试跑
```

### 评估结果（k=5，18 道计分题，73 个片段）

| 方式 | recall@5 | MRR | 命中 |
| --- | --- | --- | --- |
| vector | 0.944 | 0.770 | 17/18 |
| keyword | 0.944 | 0.835 | 17/18 |
| **hybrid** | **1.000** | **0.858** | **18/18** |

两处漏召回恰好互为反例，说明混合确实在补齐短板：

- **Q17**（"为什么打码不能喂给模型"，答案在 `DA-015`）：向量漏（top1 落在 README 的"主要能力"），
  关键词命中第 5，**混合命中**；
- **Q15**（"电商数据集埋了哪些异常"）：关键词漏（答案所在的续片没进前 5），
  向量命中第 2，**混合命中第 3**。

**进 RAG-05 的闸门（recall@5 ≥ 0.9）已满足**；第二条"每条召回都能解释为什么"由报告第三部分
逐题给出（`why` 字段）。

### 这轮修掉的两个真问题（都是评估先暴露的）

1. **融合池太浅**：原本只在两路各自的 top-k 上融合，单路深处的正确结果会被直接丢掉
   （Q17 就是这样漏的）。改成在每路前 `4×k` 上融合、再截到 k 后，hybrid 从 17/18 变 18/18。
2. **`cache or ...` 吞掉空缓存**：`EmbeddingCache` 定义了 `__len__`，空缓存是 **falsy**，
   于是显式传进去的空缓存被静默换成配置里的路径（测试里表现为"空缓存却读到了向量"）。
   已改为显式判 `None`。

### 已知限制

1. **陷阱题分数不可当阈值用**：T01/T02 的向量相似度（0.61 / 0.52）与 BM25 分数没有可比标尺，
   现在只能人工看"召回了什么"。真正的价值在 RAG-05：看模型会不会拿不相干片段编答案。
2. **BM25 对"续片"弱**：表格续行、引用段这类片段与提问的 2-gram 重合度天然偏低。
   根因在分块策略（表格续片补了表头，但语义重心仍不在前几行），不是打分公式的问题。
3. **评估集是我起草的**（18+2 题），存在"出题人偏见"：题面用词与语料用词重合度偏高。
   若要更可信，应由你改写题面（`eval/questions.jsonl` 直接改，重跑即可）。

---

## 下一步

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| RAG-01 | 目录 + 隔离环境 + 语料盘点 | ✅ |
| RAG-02 | 语料准备 + 分块 + 来源信息 | ✅ |
| RAG-03 | 向量生成 + 缓存 + 错误处理 | ✅ |
| RAG-04 | 三种检索（向量 / 2-gram BM25 / RRF）+ 20 题评估 | ✅ recall@5 = 1.000（hybrid） |
| RAG-05 | 过闸门后接入主工程 `app/tools/`（注册成知识库检索工具，复用审计 / 打码 / 配额） | 待审 |
