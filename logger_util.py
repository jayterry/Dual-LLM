import logging
from typing import Optional


class _StageDefaultFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # 第三方 logger（例如 httpx）不一定帶 stage；避免 Formatter KeyError
        if not hasattr(record, "stage"):
            setattr(record, "stage", "-")
        return True


class StageAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("stage", self.extra.get("stage", "-"))
        return msg, kwargs


def setup_logging(*, level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    設定全域 logging。
    - 預設輸出到 stderr
    - 若提供 log_file，會額外輸出到檔案
    """
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(lvl)

    # 避免重複加 handler（例如在 notebook / 重複呼叫）
    if getattr(root, "_agentproject_configured", False):
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(stage)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setLevel(lvl)
    sh.addFilter(_StageDefaultFilter())
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(lvl)
        fh.addFilter(_StageDefaultFilter())
        fh.setFormatter(fmt)
        root.addHandler(fh)

    setattr(root, "_agentproject_configured", True)


def get_logger(stage: str) -> StageAdapter:
    return StageAdapter(logging.getLogger("agentproject"), {"stage": stage})

