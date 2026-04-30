import argparse
import json
import os
from typing import Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from guard_llm import GuardReport, run_guard_universal
from logger_util import get_logger, setup_logging
from memory_store import DualStore
from rag import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, OLLAMA_MODEL, get_embedding_model
from toxic_match import fuse_risk, match_toxic


def _coerce_llm_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def fallback_answer_from_report_dict(report: dict[str, Any]) -> str:
    """LLM 失敗或回空時，仍回傳可讀的審查摘要（繁中）。"""
    lines: list[str] = [
        "（模型暫時無法產生說明，以下僅整理審查結果 JSON 的重點。）",
        f"verdict：{report.get('verdict', '?')}",
        f"risk_score_total（警衛）：{report.get('risk_score_total', '?')}",
    ]
    rf = report.get("risk_fusion")
    if isinstance(rf, dict):
        lines.append(
            f"融合風險：R_LLM {rf.get('r_llm', '?')}　"
            f"S_tox {rf.get('s_tox', '?')}　risk_total {rf.get('risk_total', '?')}"
        )
    lines.append(f"risk_score_total_fused：{report.get('risk_score_total_fused', '?')}")
    ru = report.get("risk_user")
    if isinstance(ru, dict):
        lines.append(
            f"個人化：S_user {ru.get('s_user', '?')}　delta_user {ru.get('delta_user', '?')}　"
            f"trust_status {ru.get('trust_status', '?')}"
        )
        rs = ru.get("reasons")
        if isinstance(rs, list) and rs:
            lines.append("reasons：" + "；".join(str(x) for x in rs[:10]))
    sig = report.get("signals")
    if isinstance(sig, list):
        for item in sig[:6]:
            if isinstance(item, dict):
                lines.append(f"- [{item.get('type')}] {item.get('evidence', '')}")
    mem = str(report.get("sanitized_memory") or "").strip()
    if mem:
        cut = mem[:500]
        lines.append(f"去毒摘要：{cut}{'…' if len(mem) > 500 else ''}")
    return "\n".join(lines)


def agent_answer_from_report(
    *,
    question: str,
    report_json: str,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.0,
) -> str:
    """
    Chat / XAI：不依賴向量檢索，只根據 API 層組好的審查 JSON（含 risk_fusion、risk_user）回答。
    原文不可信，不得把使用者輸入當作事實；僅能引用 JSON 內已結構化的欄位。

    將 JSON 放在單一 human 變數 user_turn，避免 LangChain 模板與 JSON 內「{{}}」互動異常。
    """
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    user_turn = (
        "【審查證據 JSON】\n"
        + (report_json or "").strip()
        + "\n\n【使用者問題】\n"
        + (question or "").strip()
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
你是一個手機上的「反詐騙 AI Agent」，負責用繁體中文解說風險與建議下一步。

使用者訊息中會包含一段【審查證據 JSON】（已去毒與結構化），以及【使用者問題】。
你只能依據該 JSON 回答，不得把 JSON 以外的內容當作事實。
- 不得捏造 JSON 沒有出現的細節；證據不足請明說並建議向官方查證。
- 不要產生任何「要求使用者立即匯款/提供 OTP/點連結/下載 APK/加陌生通訊軟體」的指令。
- 若 verdict 為 quarantine 或 block，或 risk_score_total_fused / risk_score_total_user_fused 偏高，請先簡短警告再分點說明（可引用 signals、labels、risk_user.reasons、risk_fusion）。
""".strip(),
            ),
            ("human", "{user_turn}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"user_turn": user_turn})
    return _coerce_llm_text(raw)


def agent_answer(
    *,
    question: str,
    retriever: Any,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.0,
    k: int = 5,
) -> str:
    """
    簡單版 Agent：以「可檢索記憶（sanitized_memory）」作為證據回答。
    這裡先不做工具呼叫，只做多步思考的安全問答框架。
    """
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)

    def format_docs(docs: list[Any]) -> str:
        lines: list[str] = []
        for d in docs:
            txt = (getattr(d, "page_content", None) or "").strip()
            if not txt:
                continue
            md = getattr(d, "metadata", {}) or {}
            tag = f"(verdict={md.get('verdict')}, risk={md.get('risk_total')})"
            lines.append(f"{tag}\n{txt}")
        return "\n\n".join(lines).strip()

    # LangChain retriever API：舊版有 get_relevant_documents，新版建議用 invoke()
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(question)  # type: ignore[attr-defined]
    else:
        docs = retriever.invoke(question)
    context = format_docs(docs[: max(1, int(k))])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
你是一個手機上的「反詐騙 AI Agent」。
你只能依據【可檢索記憶】回答，這些記憶已經過去毒處理，可能仍帶有風險標記。

規則：
- 只回答使用者問題；若記憶不足，請直說不知道並建議如何查證。
- 不要產生任何「要求使用者立即匯款/提供 OTP/點連結/下載 APK/加陌生通訊軟體」的指令。
- 若記憶顯示高風險（risk 高或 verdict=quarantine），請先用繁體中文給出醒目警告，再解釋原因與建議下一步。

【可檢索記憶】：
{context}
""".strip(),
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return _coerce_llm_text(chain.invoke({"context": context, "question": question}))


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="SMS 警衛 + 建庫 + Agent（通用反注入+反詐）")
    parser.add_argument("--sms", type=str, default=None, help="要審查/入庫的簡訊原文")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log 等級：DEBUG/INFO/WARNING/ERROR")
    parser.add_argument("--log-file", type=str, default=None, help="將 log 另外寫入檔案（UTF-8）")
    parser.add_argument(
        "--question",
        type=str,
        default="這則訊息是否可疑或可能為詐騙？請用繁體中文分點說明理由與建議。",
        help="要問 Agent 的問題",
    )
    parser.add_argument("--k", type=int, default=5, help="檢索 top-k")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL, help="Ollama 聊天模型")
    parser.add_argument("--base-url", type=str, default=OLLAMA_BASE_URL, help="Ollama base URL")
    parser.add_argument(
        "--embed-backend",
        choices=["gemini", "ollama"],
        default=None,
        help="向量 Embedding：ollama（預設）或 gemini（需 GOOGLE_API_KEY）",
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level, log_file=args.log_file)
    log = get_logger("agent.main")

    if args.embed_backend:
        os.environ["RAG_EMBEDDING_BACKEND"] = args.embed_backend

    sms = (args.sms or "").strip()
    if not sms:
        print("請用 --sms 提供一則簡訊原文。")
        return 2
    log.info("收到簡訊輸入（len=%s）", len(sms))

    # 1) 警衛審查（通用反注入 + 反詐）
    log.info("開始警衛審查（model=%s, base_url=%s）", args.model, args.base_url)
    report: GuardReport = run_guard_universal(
        text=sms,
        model=args.model,
        base_url=args.base_url,
        temperature=0.0,
    )
    log.info(
        "警衛完成 verdict=%s risk_total=%s injection=%s scam=%s",
        report.verdict,
        report.risk_score_total,
        report.injection_score,
        report.scam_score,
    )
    print("\n=== GuardReport ===")
    _print_json(json.loads(report.to_json()))

    # 與 server.py 一致：R_LLM（警衛）+ S_tox（毒樣庫）融合
    toxic_path = os.environ.get("TOXIC_DB_PATH", "").strip()
    m = match_toxic(raw_sms=sms, toxic_path=toxic_path)
    fused = fuse_risk(r_llm_0_100=int(report.risk_score_total), s_tox_0_100=int(m.s_tox_0_100))
    print("\n=== R_LLM / S_tox / 融合風險（與 API risk_fusion 相同口徑）===")
    print(f"R_LLM（警衛 0–100）: {fused.r_llm_0_100}")
    print(f"S_tox（毒樣相似度 0–100）: {fused.s_tox_0_100}")
    print(f"risk_total（融合 0–100）: {fused.risk_total_0_100}")
    print(f"risk_score_total_fused（API 欄位同名）: {fused.risk_total_0_100}")
    if toxic_path:
        print(
            f"toxic 啟用: max_cosine={m.max_cosine:.4f} best_id={m.best_id!r} "
            f"entries={m.toxic_count} embed_dim={m.embed_dim}"
        )
    else:
        print("toxic: 未設定 TOXIC_DB_PATH，S_tox 通常為 0（略過比對）")

    # 2) 建庫（雙軌：原文存檔 + 去毒摘要入向量庫）
    log.info("建立 embedding（backend=%s, embed_model=%s）", os.environ.get("RAG_EMBEDDING_BACKEND", "ollama"), OLLAMA_EMBED_MODEL)
    embeddings = get_embedding_model()
    store = DualStore(embedding=embeddings, collection_prefix="sms_mem")
    log.info("初始化 DualStore collection=%s", store.collection_name)
    message_id = store.ingest_sms(raw_text=sms, guard_report=report, meta={"source": "cli"})
    log.info("入庫完成 message_id=%s archive_size=%s", message_id, store.archive_size)
    print(f"\n=== Ingested ===\nmessage_id={message_id}\ncollection={store.collection_name}\n")

    # 3) Agent 回答（只檢索去毒記憶）
    log.info("開始檢索與回答（k=%s）", args.k)
    retriever = store.as_retriever(k=args.k)
    answer = agent_answer(
        question=args.question,
        retriever=retriever,
        model=args.model,
        base_url=args.base_url,
        temperature=0.0,
        k=args.k,
    )
    log.info("回答完成（answer_len=%s）", len(answer))
    print("=== Agent Answer ===")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

