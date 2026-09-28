"""Face Swap Video for Windows — desktop port of the Android Face Swap Video 1.0 app."""
__version__ = "1.0.0"


def _install_matplotlib_stub():
    """mediapipe 1.0.x imports matplotlib.pyplot (drawing helpers only) while importing its package. The packaged
    app doesn't ship matplotlib (~60 MB), so register an empty stand-in BEFORE anything imports mediapipe — a failed
    first import would leave mediapipe half-initialised in sys.modules."""
    import sys
    if "matplotlib.pyplot" in sys.modules:
        return
    try:
        import importlib.util
        if importlib.util.find_spec("matplotlib") is not None:
            return
    except Exception:  # noqa: BLE001
        pass
    import types
    mpl = types.ModuleType("matplotlib"); plt = types.ModuleType("matplotlib.pyplot")
    mpl.pyplot = plt; sys.modules["matplotlib"] = mpl; sys.modules["matplotlib.pyplot"] = plt


_install_matplotlib_stub()
