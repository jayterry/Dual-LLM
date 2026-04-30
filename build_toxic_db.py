from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any

from logger_util import get_logger, setup_logging
from rag import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, get_embedding_model
from toxic_match import canonicalize_sms_for_toxic

log = get_logger("toxic.build")


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_str_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []


def _embed_text(text: str) -> list[float]:
    emb = get_embedding_model()
    v = emb.embed_query(text)
    if not isinstance(v, list):
        return [float(x) for x in list(v)]
    return [float(x) for x in v]


def _build_item(seed: dict[str, Any], *, ollama_base_url: str, embed_model: str) -> dict[str, Any]:
    eid = str(seed.get("id") or "").strip()
    if not eid:
        raise ValueError("seed 條目缺少 id")

    payload = str(seed.get("payload") or "")
    if not payload.strip():
        raise ValueError(f"seed 條目缺少 payload：id={eid!r}")

    attack_type = str(seed.get("attack_type") or "unknown")
    ts = str(seed.get("timestamp") or _iso_now())

    canon = canonicalize_sms_for_toxic(payload)
    vec = _embed_text(canon)
    if not vec:
        raise RuntimeError(f"embedding 產生失敗：id={eid!r}")

    item: dict[str, Any] = {
        "id": eid,
        "timestamp": ts,
        "attack_type": attack_type,
        "payload": payload,
        "canonical_text": canon,
        "embedding_model": embed_model,
        "embedding_base_url": ollama_base_url,
        "embedding_vector": vec,
    }

    if "risk_score" in seed and seed.get("risk_score") is not None:
        item["risk_score"] = seed.get("risk_score")
    if "signals" in seed and seed.get("signals") is not None:
        item["signals"] = _as_str_list(seed.get("signals"))
    if "note" in seed and str(seed.get("note") or "").strip():
        item["note"] = str(seed.get("note")).strip()

    return item


def main() -> int:
    parser = argparse.ArgumentParser(description="將 toxic_seed.json 轉成 toxic_db.json（寫入 embedding_vector）。")
    parser.add_argument(
        "--seed",
        type=str,
        default="toxic_seed.json",
        help="seed JSON 路徑（JSON 陣列；也可直接給一個 {items:[]} 物件）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="toxic_db.json",
        help="輸出 toxic 資料庫 JSON 路徑",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log 等級：DEBUG/INFO/WARNING/ERROR")
    parser.add_argument("--log-file", type=str, default=None, help="可選：額外輸出 log 到檔案（UTF-8）")
    args = parser.parse_args()

    setup_logging(level=args.log_level, log_file=args.log_file)

    ollama_base_url = (os.environ.get("OLLAMA_BASE_URL") or OLLAMA_BASE_URL).strip()
    embed_model = (os.environ.get("OLLAMA_EMBED_MODEL") or OLLAMA_EMBED_MODEL).strip()
    log.info("準備產生 Toxic DB（embed_model=%s, base_url=%s）", embed_model, ollama_base_url)

    with open(args.seed, "r", encoding="utf-8") as f:
        data = json.load(f)

    seeds: list[dict[str, Any]]
    if isinstance(data, list):
        seeds = [x for x in data if isinstance(x, dict)]
    elif isinstance(data, dict) and isinstance(data.get("items"), list):
        seeds = [x for x in data["items"] if isinstance(x, dict)]
    else:
        print("seed 檔案格式不支援：請用 JSON 陣列，或 {items: []}。")
        return 2

    out: list[dict[str, Any]] = []
    for s in seeds:
        item = _build_item(s, ollama_base_url=ollama_base_url, embed_model=embed_model)
        out.append(item)
        log.info("已產生 toxic 條目 id=%s dim=%s", item["id"], len(item.get("embedding_vector", [])))

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    log.info("完成輸出 path=%s rows=%s", out_path, len(out))
    print(f"OK: wrote {len(out)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
