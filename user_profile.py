from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from logger_util import get_logger

log = get_logger("profile.db")

_LOCK = threading.Lock()


def _default_path() -> str:
    # 允許用環境變數覆寫；否則放在專案目錄同層
    return os.environ.get("USER_PROFILE_DB_PATH", "").strip() or os.path.join(os.getcwd(), "user_profile_db.json")


def _load_json(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:  # noqa: BLE001
        log.error("讀取 profile DB 失敗 path=%s err=%s", path, e)
        return {}


def _save_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _now_ts() -> float:
    return float(time.time())


def _norm_source(source: Optional[str]) -> str:
    s = (source or "").strip().lower()
    return s or "unknown"


def _ensure_schema(db: dict[str, Any]) -> dict[str, Any]:
    """
    升級/確保 profile DB schema 存在。

    目標 schema（四維度）：
    - sources: {source_id -> {...}}
    - categories: {category -> {...}}
    - temporal: {hour_hist[24], last_burst:{...}}
    - actions: global counters（不存原文）

    兼容舊版：
    - trusted_sources/blocked_sources/seen 會被映射到 sources[*].trust_status / seen_count / last_seen_at
    """
    if not isinstance(db, dict):
        db = {}

    # 新結構
    db.setdefault("sources", {})
    db.setdefault("categories", {})
    db.setdefault("temporal", {"hour_hist": [0] * 24, "last_burst": {"window_s": 3600, "count": 0, "started_at": 0.0}})
    db.setdefault(
        "actions",
        {
            "url_seen_count": 0,
            "otp_seen_count": 0,
            "apk_seen_count": 0,
            "money_terms_count": 0,
        },
    )

    # 遷移舊欄位（若存在）
    legacy_trusted = db.get("trusted_sources") if isinstance(db.get("trusted_sources"), dict) else {}
    legacy_blocked = db.get("blocked_sources") if isinstance(db.get("blocked_sources"), dict) else {}
    legacy_seen = db.get("seen") if isinstance(db.get("seen"), dict) else {}

    sources = db["sources"]
    if isinstance(sources, dict):
        for sid, meta in legacy_seen.items():
            if not isinstance(sid, str):
                continue
            key = _norm_source(sid)
            src = sources.get(key)
            if not isinstance(src, dict):
                src = {}
            if isinstance(meta, dict):
                src.setdefault("seen_count", int(meta.get("count") or 0))
                src.setdefault("last_seen_at", float(meta.get("last_seen_at") or 0.0))
                # 首見時間無法從舊版可靠推回，保守設為 last_seen
                src.setdefault("first_seen_at", float(meta.get("last_seen_at") or 0.0))
            sources[key] = src

        for sid in legacy_trusted.keys():
            if not isinstance(sid, str):
                continue
            key = _norm_source(sid)
            src = sources.get(key)
            if not isinstance(src, dict):
                src = {}
            src["trust_status"] = "TRUSTED"
            src.setdefault("trusted_at", _now_ts())
            sources[key] = src

        for sid in legacy_blocked.keys():
            if not isinstance(sid, str):
                continue
            key = _norm_source(sid)
            src = sources.get(key)
            if not isinstance(src, dict):
                src = {}
            src["trust_status"] = "BLOCKED"
            src.setdefault("blocked_at", _now_ts())
            sources[key] = src

    # 舊欄位保留（方便回溯），但新邏輯只使用新 schema
    return db


def _get_source(db: dict[str, Any], source: str) -> dict[str, Any]:
    sources = db.get("sources")
    if not isinstance(sources, dict):
        sources = {}
        db["sources"] = sources
    src = sources.get(source)
    if not isinstance(src, dict):
        src = {}
        sources[source] = src
    src.setdefault("trust_status", "UNKNOWN")  # TRUSTED|BLOCKED|UNKNOWN
    src.setdefault("entity_tags", [])
    src.setdefault("seen_count", 0)
    src.setdefault("first_seen_at", 0.0)
    src.setdefault("last_seen_at", 0.0)
    src.setdefault("url_seen_count", 0)
    src.setdefault("otp_seen_count", 0)
    src.setdefault("apk_seen_count", 0)
    src.setdefault("money_terms_count", 0)
    return src


def _hour_of(ts: float) -> int:
    try:
        lt = time.localtime(ts)
        return int(getattr(lt, "tm_hour", 0))
    except Exception:
        return 0


def _contains_sensitive_actions(text: str) -> dict[str, bool]:
    t = (text or "").lower()
    return {
        "has_url": ("http://" in t) or ("https://" in t) or ("www." in t),
        "has_otp": any(x in t for x in ("otp", "驗證碼", "一次性密碼")),
        "has_apk": any(x in t for x in ("apk", "安裝", "下載")),
        "has_money": any(x in t for x in ("匯款", "轉帳", "付款", "刷卡", "繳費")),
    }


def _add_entity_tags(src: dict[str, Any], tags: list[str]) -> None:
    if not tags:
        return
    cur = src.get("entity_tags")
    if not isinstance(cur, list):
        cur = []
    # 去重、限長（避免污染與爆長）
    out: list[str] = [str(x) for x in cur if str(x).strip()]
    for t in tags:
        s = str(t or "").strip()
        if not s:
            continue
        if s not in out:
            out.append(s)
        if len(out) >= 20:
            break
    src["entity_tags"] = out


@dataclass(frozen=True)
class UserRisk:
    s_user_0_100: int
    delta_user: int  # -15..+15
    need_user_confirm: bool
    reasons: list[str]
    source_key: str
    trusted: bool
    blocked: bool
    seen_count: int
    trust_status: str


def mark_trusted(*, source: str, path: Optional[str] = None) -> dict[str, Any]:
    p = path or _default_path()
    key = _norm_source(source)
    with _LOCK:
        db = _ensure_schema(_load_json(p))
        src = _get_source(db, key)
        src["trust_status"] = "TRUSTED"
        src["trusted_at"] = _now_ts()
        _save_json(p, db)
    return {"ok": True, "source": key}


def mark_blocked(*, source: str, path: Optional[str] = None) -> dict[str, Any]:
    p = path or _default_path()
    key = _norm_source(source)
    with _LOCK:
        db = _ensure_schema(_load_json(p))
        src = _get_source(db, key)
        src["trust_status"] = "BLOCKED"
        src["blocked_at"] = _now_ts()
        _save_json(p, db)
    return {"ok": True, "source": key}


def record_seen(*, source: str, path: Optional[str] = None) -> int:
    p = path or _default_path()
    key = _norm_source(source)
    with _LOCK:
        db = _ensure_schema(_load_json(p))
        now = _now_ts()
        src = _get_source(db, key)
        src["seen_count"] = int(src.get("seen_count") or 0) + 1
        src["last_seen_at"] = now
        if float(src.get("first_seen_at") or 0.0) <= 0.0:
            src["first_seen_at"] = now

        # temporal：小時直方圖
        temporal = db.get("temporal")
        if isinstance(temporal, dict):
            hh = temporal.get("hour_hist")
            if isinstance(hh, list) and len(hh) == 24:
                h = _hour_of(now)
                hh[h] = int(hh[h] or 0) + 1
                temporal["hour_hist"] = hh

            # burst：簡單 1 小時窗口計數
            lb = temporal.get("last_burst")
            if not isinstance(lb, dict):
                lb = {"window_s": 3600, "count": 0, "started_at": now}
            win = int(lb.get("window_s") or 3600)
            started = float(lb.get("started_at") or now)
            if now - started > float(win):
                lb = {"window_s": win, "count": 1, "started_at": now}
            else:
                lb["count"] = int(lb.get("count") or 0) + 1
            temporal["last_burst"] = lb
            db["temporal"] = temporal

        _save_json(p, db)
        return int(src["seen_count"])


def update_observation(
    *,
    text: str,
    source: str,
    entity_tags: Optional[list[str]] = None,
    path: Optional[str] = None,
) -> dict[str, Any]:
    """
    寫入「行為特徵」與「敏感動作統計」。
    注意：不存原文，只存統計/特徵，以降低污染面。
    """
    p = path or _default_path()
    key = _norm_source(source)
    now = _now_ts()
    flags = _contains_sensitive_actions(text)
    with _LOCK:
        db = _ensure_schema(_load_json(p))
        src = _get_source(db, key)

        # 行為統計
        if flags["has_url"]:
            src["url_seen_count"] = int(src.get("url_seen_count") or 0) + 1
        if flags["has_otp"]:
            src["otp_seen_count"] = int(src.get("otp_seen_count") or 0) + 1
        if flags["has_apk"]:
            src["apk_seen_count"] = int(src.get("apk_seen_count") or 0) + 1
        if flags["has_money"]:
            src["money_terms_count"] = int(src.get("money_terms_count") or 0) + 1

        # 全域 action 計數
        actions = db.get("actions")
        if isinstance(actions, dict):
            if flags["has_url"]:
                actions["url_seen_count"] = int(actions.get("url_seen_count") or 0) + 1
            if flags["has_otp"]:
                actions["otp_seen_count"] = int(actions.get("otp_seen_count") or 0) + 1
            if flags["has_apk"]:
                actions["apk_seen_count"] = int(actions.get("apk_seen_count") or 0) + 1
            if flags["has_money"]:
                actions["money_terms_count"] = int(actions.get("money_terms_count") or 0) + 1
            db["actions"] = actions

        # entity tags（只做弱記憶；避免污染：只在來源未封鎖時寫入）
        if src.get("trust_status") != "BLOCKED":
            _add_entity_tags(src, entity_tags or [])

        src["last_obs_at"] = now
        _save_json(p, db)
    return {"ok": True, "source": key, "flags": flags}


def compute_user_risk(
    *,
    text: str,
    source: Optional[str],
    llm_verdict: str,
    risk_total_fused: int,
    path: Optional[str] = None,
) -> UserRisk:
    """
    最小可行 S_user / delta_user：
    - 以 source（例如 packageName）是否「新、信任、封鎖」作為主要訊號
    - 以內容是否含 URL/OTP/匯款等敏感關鍵詞作為次要訊號
    - delta_user 限制在 [-15, +15]，避免個人化洗白高風險
    - 若 need_user_confirm=True，前端可跳出「是否信任來源」詢問
    """
    p = path or _default_path()
    key = _norm_source(source)
    t = (text or "").strip().lower()

    with _LOCK:
        db = _ensure_schema(_load_json(p))
        src = _get_source(db, key)
        trust_status = str(src.get("trust_status") or "UNKNOWN").upper()
        seen_count = int(src.get("seen_count") or 0)

        trusted = trust_status == "TRUSTED"
        blocked = trust_status == "BLOCKED"

        temporal = db.get("temporal") if isinstance(db.get("temporal"), dict) else {}
        lb = temporal.get("last_burst") if isinstance(temporal, dict) else {}
        burst_count = int(lb.get("count") or 0) if isinstance(lb, dict) else 0

    reasons: list[str] = []
    s_user = 0
    delta = 0
    need_confirm = False

    # 1) 來源信任狀態（最大影響，對應 ±15）
    if blocked:
        reasons.append("source.trust_status=BLOCKED(+15)")
        s_user = 95
        delta = 15
        need_confirm = False
    elif trusted:
        reasons.append("source.trust_status=TRUSTED(-15)")
        s_user = 10
        delta = -15
        need_confirm = False
    else:
        if seen_count <= 1:
            reasons.append("source.new_or_rare")
            s_user += 70
            delta += 6
            need_confirm = True
        else:
            reasons.append("source.seen_before_unknown")
            s_user += 40
            delta += 2

    # 敏感詞（只用於提高 need_confirm 與 s_user；不做洗白）
    sensitive_terms = ("http://", "https://", "otp", "驗證碼", "一次性密碼", "匯款", "轉帳", "付款", "下載", "apk", "點擊", "連結")
    if any(x in t for x in sensitive_terms):
        reasons.append("actions.contains_sensitive_terms")
        s_user = min(100, s_user + 15)
        delta = min(15, delta + 5)
        need_confirm = True

    # 2) temporal/burst：短時間大量通知（行為異常）
    if burst_count >= int(os.environ.get("USER_BURST_N", "8")):
        reasons.append("temporal.burst_detected")
        s_user = min(100, s_user + 10)
        delta = min(15, delta + 3)
        need_confirm = True

    # 安全護欄：高風險/封鎖時，個人化不得降低風險
    if str(llm_verdict).lower() == "block" or int(risk_total_fused) >= 85:
        if delta < 0:
            reasons.append("guardrail_no_downrank")
            delta = 0
        need_confirm = False  # 直接高風險，不用問「信任就放行」

    # clamp
    s_user = int(max(0, min(100, s_user)))
    delta = int(max(-15, min(15, delta)))

    return UserRisk(
        s_user_0_100=s_user,
        delta_user=delta,
        need_user_confirm=bool(need_confirm),
        reasons=reasons,
        source_key=key,
        trusted=bool(trusted),
        blocked=bool(blocked),
        seen_count=int(seen_count),
        trust_status=trust_status,
    )

