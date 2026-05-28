import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# supervision and cv2 may be broken in the test environment (cv2 attribute error).
# Stub only the modules that are broken or unavailable before importing evaluate.
# We must NOT stub mlflow or numpy — other test files use the real packages.
for _mod in ("cv2", "supervision", "supervision.metrics", "rfdetr", "tqdm"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# PIL.Image may also be broken via supervision's cv2 dependency chain; stub if absent.
for _mod in ("PIL", "PIL.Image"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import evaluate  # noqa: E402  (must come after stubs)


def test_load_model_large_passes_num_queries():
    # patch rfdetr module attribute — the local `from rfdetr import RFDETRLarge`
    # inside load_model picks up the mock because it reads from the same module object
    ckpt = Path("checkpoint.pth")
    with patch("rfdetr.RFDETRLarge") as mock_cls:
        evaluate.load_model("large", ckpt, num_queries=6000)
        mock_cls.assert_called_once_with(pretrain_weights=str(ckpt), num_queries=6000)


def test_load_model_base_passes_num_queries():
    ckpt = Path("checkpoint.pth")
    with patch("rfdetr.RFDETRBase") as mock_cls:
        evaluate.load_model("base", ckpt, num_queries=6000)
        mock_cls.assert_called_once_with(pretrain_weights=str(ckpt), num_queries=6000)


def test_load_model_omits_num_queries_when_none():
    ckpt = Path("checkpoint.pth")
    with patch("rfdetr.RFDETRLarge") as mock_cls:
        evaluate.load_model("large", ckpt, num_queries=None)
        _, call_kwargs = mock_cls.call_args
        assert "num_queries" not in call_kwargs


def test_load_model_unknown_variant_raises():
    with pytest.raises(ValueError, match="Unknown model variant"):
        evaluate.load_model("xlarge", Path("checkpoint.pth"))
