import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from guard_llm import GuardReport
from logger_util import get_logger


@dataclass
class ArchivedMessage:
    message_id: str
    created_at: float
    raw_text: str
    guard_report: dict[str, Any]
    meta: dict[str, Any]


def _as_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


class DualStore:
    """
    雙軌存放：
    - archive：原文 + 警衛報告（不可檢索）
    - indexed：只放 sanitized_memory（可檢索向量庫）
    """

    def __init__(
        self,
        *,
        embedding: Any,
        collection_prefix: str = "sms_mem",
        chroma_client: Optional[Any] = None,
    ) -> None:
        self._archive: list[ArchivedMessage] = []
        self._embedding = embedding
        self._chroma_client = chroma_client or chromadb.EphemeralClient()
        self._collection = f"{collection_prefix}_{uuid.uuid4().hex}"
        self._vectorstore: Optional[Chroma] = None

    @property
    def collection_name(self) -> str:
        return self._collection

    @property
    def archive_size(self) -> int:
        return len(self._archive)

    def list_archive(self) -> list[ArchivedMessage]:
        return list(self._archive)

    def _ensure_vectorstore(self) -> Chroma:
        if self._vectorstore is not None:
            return self._vectorstore
        # 建一個空集合；之後用 add_documents 增量寫入
        self._vectorstore = Chroma(
            client=self._chroma_client,
            collection_name=self._collection,
            embedding_function=self._embedding,
        )
        return self._vectorstore

    def ingest_sms(
        self,
        *,
        raw_text: str,
        guard_report: GuardReport,
        meta: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        依 GuardReport 決定是否建立可檢索記憶。

        - raw_text 一律進 archive（不可檢索）
        - indexed 只存 sanitized_memory（且依 verdict 決定是否寫入）
        """
        meta = meta or {}
        message_id = meta.get("message_id") or f"msg_{uuid.uuid4().hex}"
        log = get_logger("store.ingest")
        log.info("archive 寫入 raw_text（len=%s） message_id=%s verdict=%s", len(raw_text or ""), message_id, guard_report.verdict)

        archived = ArchivedMessage(
            message_id=str(message_id),
            created_at=time.time(),
            raw_text=raw_text,
            guard_report=asdict(guard_report),
            meta=meta,
        )
        self._archive.append(archived)

        # 決定 indexed 入庫策略
        safe_text = guard_report.sanitized_memory.strip()
        if not safe_text:
            log.warning("sanitized_memory 為空，略過向量入庫 message_id=%s", message_id)
            return archived.message_id

        if guard_report.verdict == "block":
            # 高風險：避免把原文/細節入可檢索庫，但允許存「極短風險摘要」，
            # 讓 Agent 仍可根據去毒後訊號給出反詐提醒，而不需要檢索原文。
            log.warning("verdict=block：改寫為極短隔離摘要後再入向量庫 message_id=%s", message_id)
            safe_text = safe_text[:220].rstrip() + ("…" if len(safe_text) > 220 else "")
            safe_text = f"（高風險隔離摘要）{safe_text}"

        doc = Document(
            page_content=safe_text,
            metadata={
                "message_id": archived.message_id,
                "risk_total": guard_report.risk_score_total,
                "injection_score": guard_report.injection_score,
                "scam_score": guard_report.scam_score,
                "verdict": guard_report.verdict,
                "labels": _as_json(guard_report.labels),
            },
        )
        vs = self._ensure_vectorstore()
        vs.add_documents([doc])
        log.info("向量入庫完成 message_id=%s collection=%s", message_id, self._collection)
        return archived.message_id

    def as_retriever(self, *, k: int = 5):
        vs = self._ensure_vectorstore()
        return vs.as_retriever(search_kwargs={"k": max(1, int(k))})

