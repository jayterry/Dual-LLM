# SMS Guard（反注入 + 反詐騙）API/CLI

這個專案提供兩種使用方式：

- **API**：用 `FastAPI + Uvicorn` 提供 `/v1/sms/scan`，可選擇是否把「去毒摘要」入向量庫並用 RAG 回答。
- **CLI**：用 `agent.py` 在本機直接審查簡訊、入庫、再用檢索結果回答問題。

核心概念是「雙軌存放」：

- **archive（不可檢索）**：保存原文 + 警衛報告
- **indexed（可檢索）**：只保存 `sanitized_memory`（去指令化的安全摘要）

## 專案檔案概覽

- `server.py`：FastAPI 服務入口（手機可透過 `adb reverse` 呼叫）
- `agent.py`：CLI 入口（審查 → 入庫 → 檢索回答）
- `guard_llm.py`：LLM 警衛（反提示注入 + 反詐騙），輸出 `GuardReport`
- `memory_store.py`：`DualStore`（archive + Chroma 向量庫）
- `rag.py`：Ollama base URL / model / embedding 設定與 `get_embedding_model()`
- `logger_util.py`：logging 統一格式（含 `stage` 欄位）

## 環境需求

- Python 3.10+（註：部分 LangChain 依賴在 Python 3.14 可能會出現 Pydantic v1 相容性警告，但不一定影響執行）
- 本機可連到 Ollama（預設 `http://localhost:11434`）
- 需要有可用的：
  - **聊天模型**（預設 `qwen2.5:7b`，請先執行 `ollama pull qwen2.5:7b`）
  - **Embedding 模型**（預設 `nomic-embed-text`）

## 安裝

```bash
pip install -r requirements.txt
```

## 啟動 API

```bash
uvicorn server:app --host 127.0.0.1 --port 8787
```

### 使用 `.env.example`（建議）

此專案支援以環境變數設定模型與選配功能。建議做法：

1. 複製 `.env.example` 成 `.env`（`.env` 已在 `.gitignore`，不會被推上 Git）
2. 依你的電腦環境調整模型 / token / toxic DB 路徑
3. 再啟動 `uvicorn`

> 若你不想用 `.env`，也可以在 PowerShell 用 `$env:NAME="value"` 方式設定環境變數。

### 手機端（可選）

若你用 USB 連線並希望手機打到電腦本機：

```bash
adb reverse tcp:8787 tcp:8787
```

Base URL：

- `http://127.0.0.1:8787`

### API Token（可選）

若設定環境變數 `AGENT_API_TOKEN`，請求需帶：

- `Authorization: Bearer <token>`

## API 使用方式

### 健康檢查

- `GET /health`

### 審查簡訊（可選入庫 + 產生回答）

- `POST /v1/sms/scan?ingest=false|true&k=5`

Body（JSON）：

```json
{
  "text": "簡訊全文",
  "model": "qwen2.5:7b",
  "base_url": "http://localhost:11434"
}
```

回傳：

- `report`：警衛結構化報告（含 `sanitized_memory`）
- `message_id`：若 `ingest=true` 才會有
- `answer`：若 `ingest=true` 才會有（由檢索到的去毒記憶回答）

## CLI 使用方式

```bash
python agent.py --sms "你的簡訊全文"
```

常用參數：

- `--question`：要問 Agent 的問題
- `--k`：檢索 top-k
- `--model`：Ollama 聊天模型
- `--base-url`：Ollama base URL

## 設定（環境變數）

下表列出專案會用到的環境變數。你可以直接參考 `.env.example`。

| 變數 | 預設值 | 用途 | 是否必填 |
|---|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API 位址 | 否 |
| `OLLAMA_MODEL` | `qwen2.5:7b` | 聊天模型（警衛 + Agent 對答） | 否 |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding 模型（向量庫 / toxic 相似度） | 否 |
| `RAG_EMBEDDING_BACKEND` | `ollama` | Embedding 後端（目前僅支援 `ollama`） | 否 |
| `AGENT_API_TOKEN` | （無） | API token（若設定，API 需帶 `Authorization: Bearer ...`） | 否 |
| `TOXIC_DB_PATH` | （無） | Toxic DB JSON 路徑（設定後才啟用 `S_tox` 比對） | 否 |
| `CHAT_REUSE_REPORT_TTL_SEC` | `900` | Chat 審查快取 TTL（秒）；>0 會加速多輪問答 | 否 |
| `USER_PROFILE_DB_PATH` | `./user_profile_db.json` | 個人化 profile DB 路徑（本機執行期資料） | 否 |
| `USER_BURST_N` | `8` | 個人化：短時間大量通知（burst）門檻 | 否 |
| `TOXIC_COS_TAU` | `0.55` | Toxic cosine → 分數映射參數 | 否 |
| `TOXIC_COS_GAMMA` | `2.0` | Toxic cosine → 分數映射參數 | 否 |
| `FUSED_RISK_W_LLM` | `0.75` | 融合分數：R_LLM 權重 | 否 |
| `FUSED_RISK_W_TOX` | `0.25` | 融合分數：S_tox 權重 | 否 |
| `FUSED_RISK_BIAS` | `0.0` | 融合分數 bias | 否 |
| `LOG_LEVEL` | `INFO` | log 等級 | 否 |
| `LOG_FILE` | （無） | 若提供則額外輸出 log 到檔案 | 否 |

### 什麼檔案不該推上 Git（重要）

- `.env`：可能含 token / 個人機器設定
- `user_profile_db.json`：可能含使用者習慣與來源信任資訊（已在 `.gitignore`）
- `chroma*/`：向量庫執行期資料（已在 `.gitignore`）
- `toxic_db.json`：若是你本機建出的 embedding 產物，建議不要推（已在 `.gitignore`）；可改推 `toxic_seed.json` 並在本機用 `build_toxic_db.py` 產生
- `*.log`、Android `build/`、`.gradle/`：執行期/建置產物

### PowerShell 一行切換範例（Demo 用）

> 設定環境變數後通常需要「重啟 uvicorn」才會生效。

切換聊天模型（例：Qwen ↔ 其他模型）：

```powershell
$env:OLLAMA_MODEL="qwen2.5:7b"
```

指定 Ollama URL（例：不是預設 11434）：

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434"
```

開啟/關閉 Chat 審查快取（加速多輪追問）：

```powershell
$env:CHAT_REUSE_REPORT_TTL_SEC="900"   # 開啟（15 分鐘）
# $env:CHAT_REUSE_REPORT_TTL_SEC="0"   # 關閉
```

開啟 Toxic 比對（S_tox）（需提供 toxic_db.json 路徑）：

```powershell
$env:TOXIC_DB_PATH="toxic_db.json"
```

設定 API Token（可選）：

```powershell
$env:AGENT_API_TOKEN="change-me"
```

重啟 API（套用新設定）：

```powershell
uvicorn server:app --host 127.0.0.1 --port 8787
```

## 目前專案的整理建議（不改功能的前提）

- 新增 `README.md`、`.gitignore`（已補上）
- `requirements.txt` 補齊（已補上）
- 後續若要「更像正式套件」：建議把檔案移到 `sms_guard/` 套件目錄，並提供 `python -m sms_guard.api` / `python -m sms_guard.cli`

