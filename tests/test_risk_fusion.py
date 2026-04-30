from toxic_match import fuse_risk


def test_fuse_risk_default_weights() -> None:
    # 預設：w_llm=0.75, w_tox=0.25, bias=0
    fused = fuse_risk(r_llm_0_100=80, s_tox_0_100=20)
    assert fused.r_llm_0_100 == 80
    assert fused.s_tox_0_100 == 20
    # 0.75*80 + 0.25*20 = 60 + 5 = 65
    assert fused.risk_total_0_100 == 65


def test_fuse_risk_clamped_0_100() -> None:
    assert fuse_risk(r_llm_0_100=-10, s_tox_0_100=0).risk_total_0_100 == 0
    assert fuse_risk(r_llm_0_100=200, s_tox_0_100=200).risk_total_0_100 == 100

