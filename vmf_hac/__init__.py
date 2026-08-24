__all__ = ["VmfHAC", "vmf_hac"]


def __getattr__(name: str):
    if name in {"VmfHAC", "vmf_hac"}:
        from vmf_hac.core import VmfHAC, vmf_hac

        exports = {"VmfHAC": VmfHAC, "vmf_hac": vmf_hac}
        globals().update(exports)
        return exports[name]
    msg = f"module 'vmf_hac' has no attribute {name!r}"
    raise AttributeError(msg)
