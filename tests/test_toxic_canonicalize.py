from toxic_match import canonicalize_sms_for_toxic


def test_canonicalize_replaces_url_phone_and_numbers() -> None:
    raw = "請到 https://example.com/a?b=1 ，或 www.test.com ，手機 0912345678，代碼 123456"
    canon = canonicalize_sms_for_toxic(raw)
    assert "[URL]" in canon
    assert "[PHONE]" in canon
    assert "[NUM]" in canon
    # 原本的網址與電話不應該還保留
    assert "https://" not in canon.lower()
    assert "www." not in canon.lower()
    assert "0912345678" not in canon


def test_canonicalize_empty_is_empty() -> None:
    assert canonicalize_sms_for_toxic("") == ""
    assert canonicalize_sms_for_toxic("   ") == ""

