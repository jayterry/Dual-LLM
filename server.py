"""
手機（USB + adb reverse）可呼叫的本機 API。

啟動（專案目錄）：
  pip install -r requirements.txt
  uvicorn server:app --host 127.0.0.1 --port 8787

手機端 Base URL（已執行 adb reverse tcp:8787 tcp:8787）：
  http://127.0.0.1:8787

可選：CHAT_REUSE_REPORT_TTL_SEC（預設 900）
  同一則 text+source 在秒數內重複「問 Agent」會重用審查 JSON，只再跑對答 LLM（較快）。
  先「送審」也會寫入快取，之後第一次問 Agent 可跳過警衛 LLM。
  設為 0 可關閉。

可選：設定環境變數 AGENT_API_TOKEN，則請求需帶
  Authorization: Bearer <token>
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from guard_llm import GuardReport, run_guard_universal
from logger_util import get_logger, setup_logging
from memory_store import DualStore
from rag import OLLAMA_BASE_URL, OLLAMA_MODEL, get_embedding_model
from toxic_match import fuse_risk, match_toxic
from user_profile import compute_user_risk, mark_blocked, mark_trusted, record_seen, update_observation

setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"), log_file=os.environ.get("LOG_FILE"))
log = get_logger("api.server")

app = FastAPI(title="SMS Guard API", version="0.1.0")

_STORE: Optional[DualStore] = None

# (過期時間 unix, report_dict) — 僅供 ingest=false 的 chat 重用，避免每句對話都跑警衛 LLM
_CHAT_REPORT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _chat_cache_ttl_sec() -> int:
    return max(0, int(os.environ.get("CHAT_REUSE_REPORT_TTL_SEC", "900")))


def _chat_report_cache_key(text: str, source: Optional[str]) -> str:
    h = hashlib.sha256()
    h.update((source or "").encode("utf-8"))
    h.update(b"\n")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _store_chat_report_cache(text: str, source: Optional[str], report_dict: dict[str, Any]) -> None:
    ttl = _chat_cache_ttl_sec()
    if ttl <= 0:
        return
    key = _chat_report_cache_key(text, source)
    now = time.time()
    _CHAT_REPORT_CACHE[key] = (now + ttl, copy.deepcopy(report_dict))
    if len(_CHAT_REPORT_CACHE) > 400:
        dead = [k for k, (exp, _) in _CHAT_REPORT_CACHE.items() if exp <= now]
        for k in dead[:320]:
            _CHAT_REPORT_CACHE.pop(k, None)


def _require_token(authorization: Optional[str] = Header(default=None)) -> None:
    token = os.environ.get("AGENT_API_TOKEN")
    if not token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="需要 Authorization: Bearer token")
    got = authorization[len("Bearer ") :].strip()
    if got != token:
        raise HTTPException(status_code=401, detail="token 不正確")


def _get_or_create_store() -> DualStore:
    global _STORE
    if _STORE is None:
        emb = get_embedding_model()
        _STORE = DualStore(embedding=emb, collection_prefix="sms_mem_api")
        log.info("已建立 DualStore collection=%s", _STORE.collection_name)
    return _STORE


class SmsScanBody(BaseModel):
    text: str = Field(..., min_length=1, description="簡訊或通知全文（不可信輸入）")
    model: Optional[str] = Field(default=None, description="Ollama 聊天模型，預設讀環境或 qwen2.5:7b")
    base_url: Optional[str] = Field(default=None, description="Ollama base URL")
    source: Optional[str] = Field(default=None, description="來源識別（例如通知 packageName）；用於 S_user/profile")


class SmsScanResponse(BaseModel):
    report: dict[str, Any]
    message_id: Optional[str] = None
    answer: Optional[str] = None


class ChatAskBody(BaseModel):
    # 要審查/要進入 RAG 的不可信文字（簡訊內容/通知內容）
    text: str = Field(..., min_length=1)
    # 使用者當下要問的問題
    question: str = Field(..., min_length=1)
    # 來源識別（建議用通知 packageName），用於 S_user
    source: Optional[str] = Field(default=None)
    ingest: bool = False
    k: int = Field(default=5, ge=1, le=50)
    model: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)


class ChatAskResponse(BaseModel):
    report: dict[str, Any]
    message_id: Optional[str] = None
    # 一律給字串，避免 App 端看到 answer: null（空則為 "" 或 fallback 摘要）
    answer: str = ""
    # True 表示本次未重跑警衛／毒樣／profile，僅重用快取的審查 JSON（較快）
    reused_report: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/sms/scan", response_model=SmsScanResponse)
def scan_sms(
    body: SmsScanBody,
    ingest: bool = False,
    k: int = 5,
    question: str = "這則訊息是否可疑或可能為詐騙？請用繁體中文分點說明理由與建議。",
    _: None = Depends(_require_token),
) -> SmsScanResponse:
    """
    警衛審查。若 ingest=true，會將去毒摘要入向量庫並用 RAG 產生 answer（較慢）。
    """
    model = body.model or OLLAMA_MODEL
    base_url = body.base_url or OLLAMA_BASE_URL

    log.info("POST /v1/sms/scan len=%s ingest=%s model=%s", len(body.text), ingest, model)
    report: GuardReport = run_guard_universal(text=body.text, model=model, base_url=base_url, temperature=0.0)
    report_dict = json.loads(report.to_json())

    # B 路：Toxic/情資庫相似度（可選；未設定或檔案不存在則略過）
    toxic_path = os.environ.get("TOXIC_DB_PATH", "").strip()
    m = match_toxic(raw_sms=body.text, toxic_path=toxic_path)
    fused = fuse_risk(r_llm_0_100=int(report.risk_score_total), s_tox_0_100=int(m.s_tox_0_100))
    report_dict["risk_fusion"] = {
        "r_llm": fused.r_llm_0_100,
        "s_tox": fused.s_tox_0_100,
        "risk_total": fused.risk_total_0_100,
        "toxic": {
            "enabled": bool(toxic_path),
            "path": toxic_path or None,
            "max_cosine": m.max_cosine,
            "best_id": m.best_id,
            "toxic_count": m.toxic_count,
            "embed_dim": m.embed_dim,
        },
    }
    # 讓欄位語意更一致：在 API 層提供合併後的總分（不改 GuardReport 本體 schema）
    report_dict["risk_score_total_fused"] = fused.risk_total_0_100

    # C 路：S_user（使用者個人化情境）— 先用電腦端 profile DB
    # 1) 記錄此來源曾出現（不等於信任，只是統計）
    seen_count = record_seen(source=body.source or "unknown")
    # 1.5) 更新「行為特徵 / 敏感動作 / 實體標籤」統計（不存原文）
    entity_tags: list[str] = []
    try:
        entity_tags = list(getattr(report.extracted, "claimed_entities", []) or [])
    except Exception:
        entity_tags = []
    update_observation(text=body.text, source=body.source or "unknown", entity_tags=entity_tags)
    ur = compute_user_risk(
        text=body.text,
        source=body.source,
        llm_verdict=report.verdict,
        risk_total_fused=int(fused.risk_total_0_100),
    )
    # 2) 個人化後的總分（小幅修正）
    risk_total_user_fused = int(max(0, min(100, int(fused.risk_total_0_100) + int(ur.delta_user))))
    report_dict["risk_user"] = {
        "source": body.source or "unknown",
        "source_key": ur.source_key,
        "seen_count": seen_count,
        "trusted": ur.trusted,
        "blocked": ur.blocked,
        "trust_status": ur.trust_status,
        "s_user": ur.s_user_0_100,
        "delta_user": ur.delta_user,
        "need_user_confirm": ur.need_user_confirm,
        "reasons": ur.reasons,
        "risk_total_user_fused": risk_total_user_fused,
    }
    report_dict["risk_score_total_user_fused"] = risk_total_user_fused

    message_id: Optional[str] = None
    answer: Optional[str] = None

    if ingest:
        # 延遲匯入：避免僅做 scan 時載入 LangChain Agent 相關依賴
        from agent import agent_answer

        store = _get_or_create_store()
        message_id = store.ingest_sms(raw_text=body.text, guard_report=report, meta={"source": "api"})
        retriever = store.as_retriever(k=k)
        answer = agent_answer(question=question, retriever=retriever, model=model, base_url=base_url, k=k)
        log.info("ingest 完成 message_id=%s", message_id)

    _store_chat_report_cache(body.text, body.source, report_dict)
    return SmsScanResponse(report=report_dict, message_id=message_id, answer=answer)


@app.post("/v1/chat/ask", response_model=ChatAskResponse)
def chat_ask(body: ChatAskBody, _: None = Depends(_require_token)) -> ChatAskResponse:
    """
    對答用 API：
    - 每次都先用 guard 做反注入/反詐審查（產生 sanitized_memory）
    - 若 ingest=true，則把去毒摘要入向量庫，並用 RAG（agent_answer）生成回答
    - ingest=false 時仍會跑 guard 與回傳 report，並用 agent_answer_from_report 依 JSON 證據生成回答（不寫入向量庫）
    - 若啟用 CHAT_REUSE_REPORT_TTL_SEC 且快取命中，可跳過警衛／毒樣／profile，只跑對答 LLM
    """
    model = body.model or OLLAMA_MODEL
    base_url = body.base_url or OLLAMA_BASE_URL

    log.info("POST /v1/chat/ask len(text)=%s ingest=%s model=%s", len(body.text), body.ingest, model)

    ttl = _chat_cache_ttl_sec()
    cache_key = _chat_report_cache_key(body.text, body.source)
    reused_report = False
    report: Optional[GuardReport] = None
    report_dict: dict[str, Any]

    now = time.time()
    cached = _CHAT_REPORT_CACHE.get(cache_key)
    if (
        not body.ingest
        and ttl > 0
        and cached is not None
        and cached[0] > now
    ):
        report_dict = copy.deepcopy(cached[1])
        reused_report = True
        log.info("POST /v1/chat/ask 重用審查快取 key_prefix=%s", cache_key[:12])
    else:
        report = run_guard_universal(
            text=body.text,
            model=model,
            base_url=base_url,
            temperature=0.0,
        )
        report_dict = json.loads(report.to_json())

        toxic_path = os.environ.get("TOXIC_DB_PATH", "").strip()
        m = match_toxic(raw_sms=body.text, toxic_path=toxic_path)
        fused = fuse_risk(r_llm_0_100=int(report.risk_score_total), s_tox_0_100=int(m.s_tox_0_100))
        report_dict["risk_fusion"] = {
            "r_llm": fused.r_llm_0_100,
            "s_tox": fused.s_tox_0_100,
            "risk_total": fused.risk_total_0_100,
            "toxic": {
                "enabled": bool(toxic_path),
                "path": toxic_path or None,
                "max_cosine": m.max_cosine,
                "best_id": m.best_id,
                "toxic_count": m.toxic_count,
                "embed_dim": m.embed_dim,
            },
        }
        report_dict["risk_score_total_fused"] = fused.risk_total_0_100

        seen_count = record_seen(source=body.source or "unknown")
        try:
            entity_tags = list(getattr(report.extracted, "claimed_entities", []) or [])
        except Exception:
            entity_tags = []
        update_observation(text=body.text, source=body.source or "unknown", entity_tags=entity_tags)

        ur = compute_user_risk(
            text=body.text,
            source=body.source,
            llm_verdict=report.verdict,
            risk_total_fused=int(fused.risk_total_0_100),
        )
        risk_total_user_fused = int(max(0, min(100, int(fused.risk_total_0_100) + int(ur.delta_user))))
        report_dict["risk_user"] = {
            "source": body.source or "unknown",
            "source_key": ur.source_key,
            "seen_count": seen_count,
            "trusted": ur.trusted,
            "blocked": ur.blocked,
            "trust_status": ur.trust_status,
            "s_user": ur.s_user_0_100,
            "delta_user": ur.delta_user,
            "need_user_confirm": ur.need_user_confirm,
            "reasons": ur.reasons,
            "risk_total_user_fused": risk_total_user_fused,
        }
        report_dict["risk_score_total_user_fused"] = risk_total_user_fused

        _store_chat_report_cache(body.text, body.source, report_dict)

    message_id: Optional[str] = None
    answer: Optional[str] = None

    if body.ingest:
        from agent import agent_answer, fallback_answer_from_report_dict

        assert report is not None  # ingest 前必跑完整審查，不會走快取
        store = _get_or_create_store()
        message_id = store.ingest_sms(raw_text=body.text, guard_report=report, meta={"source": body.source or "chat"})
        retriever = store.as_retriever(k=int(body.k))
        try:
            answer = agent_answer(
                question=body.question,
                retriever=retriever,
                model=model,
                base_url=base_url,
                temperature=0.0,
                k=int(body.k),
            )
        except Exception:
            log.exception("chat ingest：agent_answer 失敗，改用 fallback")
            answer = fallback_answer_from_report_dict(report_dict)
        if not (answer or "").strip():
            log.warning("chat ingest：answer 為空，改用 fallback")
            answer = fallback_answer_from_report_dict(report_dict)
    else:
        from agent import agent_answer_from_report, fallback_answer_from_report_dict

        payload = json.dumps(report_dict, ensure_ascii=False)
        try:
            answer = agent_answer_from_report(
                question=body.question,
                report_json=payload,
                model=model,
                base_url=base_url,
                temperature=0.0,
            )
        except Exception:
            log.exception("chat：agent_answer_from_report 失敗，改用 fallback")
            answer = fallback_answer_from_report_dict(report_dict)
        if not (answer or "").strip():
            log.warning("chat：agent 回傳空字串，改用 fallback")
            answer = fallback_answer_from_report_dict(report_dict)

    return ChatAskResponse(
        report=report_dict,
        message_id=message_id,
        answer=answer or "",
        reused_report=reused_report,
    )


class UserMarkBody(BaseModel):
    source: str = Field(..., min_length=1, description="來源識別（建議用通知 packageName）")


@app.post("/v1/user/trust")
def user_trust(body: UserMarkBody, _: None = Depends(_require_token)) -> dict[str, Any]:
    log.info("POST /v1/user/trust source=%s", body.source)
    return mark_trusted(source=body.source)


@app.post("/v1/user/block")
def user_block(body: UserMarkBody, _: None = Depends(_require_token)) -> dict[str, Any]:
    log.info("POST /v1/user/block source=%s", body.source)
    return mark_blocked(source=body.source)
