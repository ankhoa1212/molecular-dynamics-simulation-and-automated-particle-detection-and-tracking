import pytest

from train import build_model_kwargs


def test_includes_num_queries():
    assert build_model_kwargs({"variant": "large", "num_queries": 6000}) == {"num_queries": 6000}


def test_includes_amp_true():
    assert build_model_kwargs({"variant": "large", "amp": True}) == {"amp": True}


def test_includes_gradient_checkpointing_false():
    # False is not None — must be forwarded so rfdetr sees explicit False
    assert build_model_kwargs({"variant": "large", "gradient_checkpointing": False}) == {
        "gradient_checkpointing": False
    }


def test_includes_resolution_when_set():
    assert build_model_kwargs({"variant": "large", "resolution": 640}) == {"resolution": 640}


def test_excludes_null_resolution():
    assert "resolution" not in build_model_kwargs({"variant": "large", "resolution": None})


def test_excludes_variant():
    result = build_model_kwargs({"variant": "large", "num_queries": 6000})
    assert "variant" not in result


def test_full_12gb_profile():
    cfg = {
        "variant": "large",
        "num_queries": 6000,
        "amp": True,
        "gradient_checkpointing": True,
        "resolution": 640,
    }
    assert build_model_kwargs(cfg) == {
        "num_queries": 6000,
        "amp": True,
        "gradient_checkpointing": True,
        "resolution": 640,
    }
