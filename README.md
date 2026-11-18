# MedEvidence Agentic RAG

**这是本人离职后重新开发的基础预览版项目**，用于展示个人在 Agent、RAG、前后端协作、
评估流程与工程边界方面的设计思路和代码实现。

仓库以代码预览和实现交流为目的。可重点查看模块划分、调用流程、权限处理、状态收尾、
评估接口与自动化测试；运行所需的模型、资料、数据库及外部服务由使用者自行配置。

## 实现概览

```text
Vue 页面 → FastAPI 认证与资源归属 → 会话 / Agent / 工具
                                   ├─ 查询规划 → 混合召回 → RRF → 重排 → 原文证据
                                   ├─ NDJSON 回答与执行事件 → 持久化 / 取消收尾
                                   └─ 邮件草稿 → 用户页面确认 → 单次执行

外部评估输入 → 校验 → 运行 / 回放适配器 → 指标与可选判分
                                   → 聚合 → 基线比较 → 质量门禁
```

- **Agent**：查询规划、Todo、工具调用限制、用户上下文和 PostgreSQL checkpoint。
- **检索**：Dense、BM25、可选多语言辅助召回、RRF 和 Cross-Encoder；用户资料按身份过滤。
- **应用流程**：访问码/邮箱登录、独立会话、资料上传与删除、管理员资料和提示词管理。
- **可靠性**：验证码原子消费及失败限次，活动会话/资料操作协调，流取消与部分回答收尾，安全错误返回。
- **外发确认**：Agent 只创建邮件草稿，页面展示绑定的收件人、主题和正文；用户确认后才调用发送服务。
- **评估代码**：外部数据校验、生产检索/HTTP/回放适配器、分组指标、可选参考答案或远程判分、运行对比和门禁。
- **自动检查**：后端与前端测试、生产源码严格类型检查、Lint、前端构建、依赖审计和公开内容检查。

## 代码导航

| 路径 | 可查看的实现 |
|---|---|
| `src/med_evidence_agentic_rag/api/` | 登录、会话、聊天、资料、管理和邮件确认接口 |
| `agent/`、`middlewares/` | Agent 装配、上下文、事件转换与调用守卫 |
| `rag_creation/`、`rag_retrieval/` | 解析入库、检索融合、重排、索引生命周期与权限过滤 |
| `persistence/`、`models_and_database/` | 业务存储、checkpoint、模型和数据库连接 |
| `services/` | 流式收尾、资源协调、资料清理与邮件确认执行 |
| `evaluation/` | 数据契约、适配器、指标、判分、聚合、比较和命令行入口 |
| `tools/`、`prompts/` | 可调用工具、默认提示词和保存版本装配 |
| `frontend/src/` | 登录、聊天、管理页面、证据展示和确认交互 |
| `tests/`、`frontend/test/` | 以合成输入、临时目录和模拟依赖验证边界与流程 |
| `.github/workflows/ci.yml` | 自动化检查与独立 Redis 原子操作测试 |

表中未写完整前缀的后端目录均位于 `src/med_evidence_agentic_rag/`。

## 本地运行

需要 Python 3.12、Node.js 20.19+ 或 22.12+、PostgreSQL、Redis 6.2+，以及本地模型所需的 NVIDIA CUDA 环境。
当前 Embedding/重排实现使用 CUDA；评估回放与隔离测试不加载模型。

在项目根目录执行 PowerShell：

```powershell
py -3.12 -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
& $python -m pip install -e '.[dev]'
Push-Location frontend
npm ci
npm run build
Pop-Location
Copy-Item .env.example .env
```

填写 `.env` 中的 LLM、模型目录、专属数据库、Redis、JWT 和访问码配置。
进程环境变量优先，`.env` 仅补充未设置的变量。邮箱登录需要 SMTP；复杂文档解析需要 MinerU。
请为普通和管理员访问码设置不同的私有值，JWT 密钥至少使用 32 字节随机值。

启动 PostgreSQL 和 Redis 后：

```powershell
.\scripts\start.ps1 -Python $python -CheckOnly
.\scripts\start.ps1 -Python $python -Port 8001
```

访问 `http://127.0.0.1:8001/`，健康接口为 `/api/health`。应用会在专属数据库中创建缺失表；
已有表结构变更需另行迁移。服务按**单 worker**运行，当前资源协调和邮件确认状态属于该进程。
邮件确认事项有有效期，重启后失效；同一事项的重复确认不会再次发送。

资料放入本机 `data/raw/` 后，可手动运行
`python -m med_evidence_agentic_rag.rag_creation.base_corpus`。
该操作替换共享基础语料索引，请只对指定环境执行，不属于常规启动步骤。

## 评估框架使用

数据集与结果文件通过外部路径提供，仓库不附带实际测评集、历史成绩或运行报告。
输入契约见 `evaluation/schemas.py`：用例包含 `id`、`query`，可选参考答案、相关证据键、
可回答标记、期望工具及标签；回放输入采用同文件中的 `Observation` 契约。

```powershell
python -m med_evidence_agentic_rag.evaluation validate --dataset $Dataset --results $Observations
python -m med_evidence_agentic_rag.evaluation replay --dataset $Dataset --results $Observations --output $Report --k 1 3 --judge reference
python -m med_evidence_agentic_rag.evaluation compare --baseline $Baseline --current $Report --output $Comparison
python -m med_evidence_agentic_rag.evaluation gate --report $Report --minimum "hit@3=$MinimumHitAt3" --maximum error_rate=0
```

上述路径和门禁阈值由使用者设置，输出文件必须尚不存在。
生产检索使用 `run --adapter retrieval --allow-live --user-id ...`，默认接入混合检索和重排；
`--retrieval-mode dense` 用于显式比较 Dense 路径。
HTTP 流程使用 `run --adapter http --allow-live --base-url ...`，令牌从 `EVAL_AGENT_TOKEN` 读取，
为每个用例创建独立会话并默认清理。完整选项可用 `python -m med_evidence_agentic_rag.evaluation --help` 查看。

框架支持按实际观测深度计算 Hit、Recall、MRR 和 nDCG；不可观测的截止位保持空值。
引用编号有效性、参考答案词面指标和模型判分分别计算，不能互相替代。
远程判分须同时指定 `--judge remote --allow-remote-judge --judge-url ... --judge-model ...`，
密钥从 `EVAL_JUDGE_API_KEY` 读取；该操作会向指定服务发送评估文本。
正确性判分需要参考答案，忠实性判分需要实际证据上下文；未知值不补成分数。

可配置重复次数、并发和超时，聚合延迟、首字耗时、错误率及服务实际提供的 token/费用。
默认报告不保存完整问题、回答和证据正文；`--include-private-text` 需显式开启。
同步检索无法强制终止已开始的模型线程，超时会等待该线程退出以保持资源安全。

## 验证代码

```powershell
python -m pytest -q tests
python -m ruff check src tests scripts
python -m pyright --pythonpath $python
python -m pip check
python scripts/check_preview.py
Push-Location frontend
npm test
npm run build
npm audit --audit-level=moderate
Pop-Location
```

自动化测试使用合成数据与隔离依赖。Redis 集成测试仅在显式指定独立测试容器时运行；CI 已配置该服务。
外部模型、邮件和解析服务不会由普通测试自动调用。涉及真实业务时，应在自建测试环境中运行对应流程。
