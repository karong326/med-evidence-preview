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
- **可靠性**：验证码原子消费及失败限次，PostgreSQL 与本地索引操作锁，持久化的操作日志与有界恢复，流超时和取消时的部分结果保留。
- **外发确认**：草稿、过期和执行状态保存在 PostgreSQL。页面显示绑定的收件人、主题和正文，经用户确认后执行；发送结果未知时，由用户核实并记录结果。
- **运维**：独立存活与就绪探针、关联请求的 JSON 日志、受保护的 Prometheus 指标，以及显式迁移、备份和恢复命令。
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
| `services/health.py`、`services/telemetry.py` | 依赖探针、运行状态、请求关联与指标 |
| `evaluation/` | 数据契约、适配器、指标、判分、聚合、比较和命令行入口 |
| `tools/`、`prompts/` | 可调用工具、默认提示词和保存版本装配 |
| `frontend/src/` | 登录、聊天、管理页面、证据展示和确认交互 |
| `tests/`、`frontend/test/` | 以合成输入、临时目录和模拟依赖验证边界与流程 |
| `migrations/`、`scripts/maintenance.py` | 业务表迁移、数据库备份、恢复和回滚验证 |
| `Dockerfile`、`compose.yaml`、`deploy/` | 非 root 应用镜像、服务编排与 Prometheus 模板 |
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

填写 `.env` 或通过进程环境注入配置。模型目录应指向本地模型，普通和管理员访问码使用不同的私有值，
JWT 密钥至少使用 32 字节随机值。邮箱登录需要 SMTP，复杂文档解析需要 MinerU。
进程环境优先，`.env` 仅补充缺项。`DATA_ROOT` 可指定绝对路径，默认是源码目录下的 `data/`。
原文、上传和管理员提示词覆盖分别存于其中的 `raw/`、`uploads/` 和 `prompts/`。
Chroma 默认使用 `DATA_ROOT/chroma`，也可用 `CHROMA_PERSIST_DIRECTORY` 指定独立路径。
如果部署保留了旧源码目录中的提示词覆盖，把该版本文件与历史存档复制到新的 `DATA_ROOT/prompts/`。

启动专属 PostgreSQL 和 Redis，并在当前进程中显式设置 `POSTGRESQL_URL` 后，执行这些操作：

```powershell
.\scripts\start.ps1 -Python $python -CheckOnly
& $python -m med_evidence_agentic_rag.persistence.migrations upgrade
& $python -m med_evidence_agentic_rag.persistence.migrations check
.\scripts\start.ps1 -Python $python -Port 8001
```

首次使用旧版业务数据库时，先停止并排空旧实例，再保留同一时点的数据库和文件快照。
运行 `python -m med_evidence_agentic_rag.persistence.migrations adopt-legacy` 核对既有表结构，
然后执行 `upgrade` 和 `check`。结构不匹配时命令会拒绝纳管。
应用启动验证业务表迁移，并由 LangGraph 独立管理 checkpoint 表。升级期间避免混用旧、新 worker。
绑定地址与端口可通过 `APP_HOST`、`APP_PORT` 或 Python 入口的 `--host`、`--port` 设置。
预览部署按**一个应用 worker 和一个本地索引所有者**运行。

离线重建基础索引时，将资料放入 `DATA_ROOT/raw/`，停止应用后运行
`python -m med_evidence_agentic_rag.rag_creation.base_corpus`。该命令替换共享基础索引。
只测试单个文档解析时，可运行 `python -m med_evidence_agentic_rag.rag_creation.loader $ExternalPath`。
解析复杂文档可能调用配置的 MinerU 服务。

### Compose 部署

使用支持 `gpus: all` 的 Docker Compose、Linux x86-64 容器和可访问 NVIDIA GPU 的容器运行时。
准备仓库外的 Compose 环境文件（下文为 `$ComposeEnv`），其中三个路径为必填项：

- `MEDEVIDENCE_RUNTIME_ENV` 指向应用环境文件，配置方式参考 `.env.example`。
  Compose 内的 PostgreSQL、Redis 主机名分别为 `postgres` 和 `redis`。
  模型路径使用容器内的 `/models/...`，状态路径由 Compose 设为 `/state`。
- `MEDEVIDENCE_DATABASE_ENV` 指向 PostgreSQL 环境文件，设置 `POSTGRES_DB`、
  `POSTGRES_USER`、`POSTGRES_PASSWORD`，并确保与应用的连接 URL 对应。
- `MEDEVIDENCE_MODELS_DIR` 指向宿主机模型目录。该目录在容器内以只读方式挂载，
  应用用 UID/GID 10001 运行，需要能读取模型并能写入状态卷。

```powershell
docker compose --env-file $ComposeEnv config --quiet
docker compose --env-file $ComposeEnv build app
docker compose --env-file $ComposeEnv up -d postgres redis
docker compose --env-file $ComposeEnv run --rm app python scripts/check_environment.py --cuda
docker compose --env-file $ComposeEnv run --rm maintenance python -m med_evidence_agentic_rag.persistence.migrations upgrade
docker compose --env-file $ComposeEnv up -d app
```

默认只向宿主机回环地址发布应用端口。可通过 `MEDEVIDENCE_PORT` 更换端口。
对外提供服务时，在入口代理设置 TLS。保留 PostgreSQL、Redis 和应用状态卷，
并在升级前记录镜像版本、数据库目标和状态挂载。外部环境文件及模型目录均在运行时挂载。
前端依赖使用锁文件构建，Python 依赖从 `pyproject.toml` 的版本范围解析。

### 健康、限时与邮件

| 接口 | 行为 |
|---|---|
| `GET /livez` | 服务进程能够响应 HTTP 请求时返回 200 |
| `GET /readyz` | PostgreSQL、checkpoint、Redis、Chroma 和 LLM 检查均有效时返回 200，其余状态返回 503 |
| `GET /api/health` | 保留返回 200 的兼容接口，JSON 的 `status` 字段反映就绪状态 |
| `GET /metrics` | 用 `DIAGNOSTICS_TOKEN` 启用独立 Bearer 鉴权，提供请求耗时、HTTP 状态类、活动资源、问答失败和启动恢复指标 |

就绪检查读取近期采样，每个依赖最多保留一个进行中的探针。LLM 探针只检查配置模型是否出现在
`/models`。应用启动加载本地 CUDA 模型；SMTP 和 MinerU 状态仅表示配置情况。
持续收到 checkpoint 连接失败时，检查数据库并重启应用以重建连接。
运行入口输出 JSON 日志，使用匹配的路由模板和经过验证的 `X-Request-ID`，缺失或无效时生成新值。
请求 ID 仅用于关联日志，请使用无业务含义的值。

启用 Compose 监控时，把与 `DIAGNOSTICS_TOKEN` 相同的单行值放入仓库外文件，
由 `MEDEVIDENCE_METRICS_TOKEN_FILE` 指定，并授予 Prometheus 容器用户读取权限。
运行 `docker compose --env-file $ComposeEnv --profile monitoring up -d prometheus`。
抓取与告警配置位于 `deploy/prometheus.yml` 和 `deploy/alerts.yml`。

默认 LLM 单次操作超时 60 秒、最多重试 1 次，Agent 图步骤上限 32，流式执行及正常结果保存的期限 300 秒，
问题长度上限 16000 个字符。对应环境变量见 `.env.example`。
执行超时会尝试保存部分回答、证据与轨迹。已经进入本地同步推理的线程需要退出后才能释放资源，
因此清理结束时间可能晚于执行期限。

SMTP 默认使用验证证书的 TLS，沿用已有发件人变量。使用 `SMTP_HOST`、`SMTP_PORT` 和
`SMTP_TRANSPORT` 可显式切换到 STARTTLS 或可信内部捕获服务的 `plain` 模式。
明文模式要求显式主机并且不会提交认证密码。应用保留待核实的发送状态，用户核实后可在会话页面
记录已送达或未发送。该操作只记录结果。

### 恢复与回滚

启动时应用会有限度地处理可恢复的操作。若需继续处理，停止唯一的应用和索引所有者，
显式设置进程环境中的 `POSTGRESQL_URL`、`CHROMA_PERSIST_DIRECTORY` 和 `CHROMA_COLLECTION_NAME`。
命令仅接受与 `DATA_ROOT` 一致的现有绝对目录，`--limit` 和 `--attempts` 可以控制扫描与重试上限。

```powershell
python -m med_evidence_agentic_rag.services.recovery --apply --uploads $UploadRoot --base $BaseRoot
docker compose --env-file $ComposeEnv run --rm maintenance python -m med_evidence_agentic_rag.services.recovery --apply --uploads /state/uploads --base /state/raw
```

多次清理失败的记录会保留为 `blocked`，可按 `operations.kind` 和 `error_code` 汇总排查。
对于未知邮件，核实外部接收情况再记录结果，恢复命令不会重发历史操作。

先停止应用、等待活动操作结束并断开其他数据库会话，再备份数据库和状态卷。
把 `raw/`、`uploads/`、`prompts/` 和独立的 Chroma 目录连同部署配置留在私有存储。
备份命令只写 PostgreSQL 自定义格式归档，同时输出 SHA-256。
请把摘要与可信归档来源、匹配的文件快照和应用版本一起记录在仓库外。
恢复使用可信归档和单独创建的空数据库；避免向该目标启动并发客户端。

```powershell
python scripts/maintenance.py backup --target $SourceTarget --archive $Archive --quiesced
python scripts/maintenance.py restore --target $RestoreTarget --archive $Archive --sha256 $Digest --apply
python scripts/maintenance.py validate-rollback --target $RestoreTarget --archive $Archive --sha256 $Digest --filesystem-snapshot $SnapshotRoot
```

每次执行前把 `POSTGRESQL_URL` 指向该次操作的数据库。`$SourceTarget`、`$RestoreTarget` 的格式为
`HOST:PORT/DATABASE`，必须与其匹配。需要相同版本或较新的 `pg_dump`、`pg_restore`、`psql`。
归档和快照使用仓库外的绝对路径。使用 Compose 执行这些命令时，通过 `run --rm -v`
把外部归档目录挂到一次性 `maintenance` 容器，并为本次操作注入对应的数据库配置。
若 Chroma 快照不在 `$SnapshotRoot/chroma`，使用 `--chroma-snapshot` 指定路径。
备份时同时记录应用进程可见的绝对路径，包括 `DATA_ROOT/raw`、`DATA_ROOT/uploads` 和
`CHROMA_PERSIST_DIRECTORY`。SQL 保存上传文件绝对路径，Chroma 索引也引用资料的绝对路径。
在隔离进程或容器中挂载恢复后的快照时，保留这些应用可见路径。容器宿主机上的目录可另选，
容器内的挂载目标须与备份时一致。更改 `DATA_ROOT` 只会改变运行时访问的目录，旧 SQL 记录和
Chroma 索引内的路径仍指向原位置。
回滚验证检查归档摘要、文件目录和当前代码的数据库版本；文件与 SQL 的逻辑一致性需在隔离实例验证。
验证后，用配对的应用版本和文件快照启动隔离实例，完成关键读写检查再切换流量。
Alembic schema 版本不会恢复业务数据或已发生的外发操作。

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
python scripts/check_preview.py --head
Push-Location frontend
npm test
npm run build
npm audit --audit-level=moderate
Pop-Location
```

自动化测试使用合成数据与隔离依赖。设置 `MEDEVIDENCE_TEST_POSTGRES_URL` 来启用独立 PostgreSQL
集成测试，验证数据库备份恢复还需要相应的客户端工具。设置 `MEDEVIDENCE_TEST_REDIS_CONTAINER` 来指定
Redis 原子操作测试的专用容器。公开内容检查默认扫描 Git 暂存内容，`--head` 扫描已提交内容。
前端继续运行依赖审计，结果随注册表中的安全公告更新。
普通测试不调用外部模型、邮件和解析服务；真实 GPU、外部服务与容量需在独立部署中验证。

### 合成端到端验证与负载

将 `MEDEVIDENCE_TEST_POSTGRES_URL` 和 `MEDEVIDENCE_TEST_REDIS_URL` 指向专用、可丢弃的测试服务。
PostgreSQL 测试账号需要创建和删除测试数据库的权限。运行
`python -m pytest -q tests/test_synthetic_deployment.py` 会启动本机 HTTP 和 SMTP 端点，
使用临时 Chroma 目录，经过登录、文本上传、检索、工具调用、邮件确认及离线恢复流程。
测试替换语言模型、嵌入与重排序推理，实际读写 SQL、LangGraph 检查点和 Redis。

在专用部署上设置环境变量 `EVAL_AGENT_TOKEN`（登录 JWT），可通过以下命令对每个请求创建独立会话，
读取回答并清理会话。`EVAL_LOAD_QUERY` 可选，默认使用合成问题。测试在达到请求数或时限时停止；
时限内已启动的请求可能额外等待创建响应或会话清理。命令输出目标指纹、并发峰值、错误及延迟摘要，
并按错误率和 p95 阈值设置退出码。真实容量需要使用目标部署及代表性流量单独测量。

```powershell
python -m med_evidence_agentic_rag.evaluation.load --allow-live --target $origin --concurrency 2 --requests 10 --duration 60 --max-error-rate 0 --max-p95 30
```

外部提供方探测需显式配置目标及凭据，并提供允许发送的文件或邮箱。每条命令都会进行实际调用。
模型探测使用指定 HTTPS 源站或回环地址的 `/v1/chat/completions`；解析探测使用官方 MinerU SDK；
邮件探测要求目标与 `SMTP_HOST` 匹配，并向指定收件人发送一封测试邮件。

```powershell
python scripts/verify_live.py model --allow-live --target $modelOrigin
python scripts/verify_live.py parser --allow-live --target mineru-official-sdk --input $syntheticDocument
python scripts/verify_live.py smtp --allow-live --target $env:SMTP_HOST --recipient $testRecipient
```

评估用例可设置 `forbidden_tools` 和 `expected_effects`。回放记录中的 `tools` 表示实际观察的调用，
`effect_counts` 单独记录由接收服务或持久化状态核实的数量。邮件草稿与 SMTP 接受应使用独立名称。
外发结果未知时将计数留空；HTTP 聊天适配器无法核实的外部副作用保持未评分。引用编号有效性衡量编号匹配，
证据对具体主张的支持需要单独核查。词面相似度与模型评分各自保留其来源。
