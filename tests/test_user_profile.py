import os

from user_profile import compute_user_risk, mark_trusted, record_seen


def test_user_risk_new_source_needs_confirm(tmp_path) -> None:
    p = tmp_path / "profile.json"
    os.environ["USER_PROFILE_DB_PATH"] = str(p)
    os.environ["USER_BURST_N"] = "999"  # 關掉 burst 影響，讓測試穩定

    # 第一次 seen 後 seen_count=1，仍視為 new_or_rare
    record_seen(source="com.test.app")
    ur = compute_user_risk(
        text="你好",
        source="com.test.app",
        llm_verdict="allow",
        risk_total_fused=10,
    )
    assert ur.trust_status == "UNKNOWN"
    assert ur.need_user_confirm is True
    assert ur.delta_user >= 6
    assert "source.new_or_rare" in ur.reasons


def test_user_risk_trusted_downrank_and_guardrail(tmp_path) -> None:
    p = tmp_path / "profile.json"
    os.environ["USER_PROFILE_DB_PATH"] = str(p)
    os.environ["USER_BURST_N"] = "999"

    mark_trusted(source="com.bank.app")

    # 一般低風險：TRUSTED 應該給 -15
    ur_ok = compute_user_risk(
        text="你的帳單已出爐",
        source="com.bank.app",
        llm_verdict="allow",
        risk_total_fused=30,
    )
    assert ur_ok.trust_status == "TRUSTED"
    assert ur_ok.delta_user == -15
    assert ur_ok.need_user_confirm is False

    # 高風險護欄：risk_total_fused >=85 時不得 downrank（delta 不能為負）
    ur_guard = compute_user_risk(
        text="你的帳戶異常",
        source="com.bank.app",
        llm_verdict="allow",
        risk_total_fused=90,
    )
    assert ur_guard.delta_user == 0
    assert "guardrail_no_downrank" in ur_guard.reasons

