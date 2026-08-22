import sys
import types

import pytest


def _neutralize_deeptrack_lazy_modules():
    """Defuse deeptrack's lazy_import submodule proxies after they've been
    poisoned by an `import deeptrack` somewhere in the suite.

    deeptrack==2.0.1 registers dozens of lazy_import proxy submodules
    (deeptrack.generators, deeptrack.models, deeptrack.pytorch, ...) in
    sys.modules as soon as `import deeptrack` runs anywhere in the process,
    even though this repo only ever uses deeptrack's non-tensorflow optics
    (Sphere/MieSphere/Brightfield, see render_brightfield.py). Any later
    generic module introspection that does getattr(module, "__file__") --
    e.g. hypothesis's local-constants scan, or its traceback-trimming
    escalation logic -- forces one of these proxies to fully load, which
    raises ImportError (tensorflow isn't installed) instead of the
    AttributeError such introspection expects, crashing unrelated code.

    Replace each still-lazy deeptrack submodule with a harmless stub
    carrying deeptrack's own (real, site-packages) __file__, so such
    introspection succeeds without ever touching functionality this repo
    doesn't use.
    """
    deeptrack = sys.modules.get("deeptrack")
    if deeptrack is None:
        return
    base_file = getattr(deeptrack, "__file__", None)
    for name, module in list(sys.modules.items()):
        if not name.startswith("deeptrack."):
            continue
        if type(module).__module__ == "lazy_import":
            stub = types.ModuleType(name)
            stub.__file__ = base_file
            sys.modules[name] = stub


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    _neutralize_deeptrack_lazy_modules()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _neutralize_deeptrack_lazy_modules()
