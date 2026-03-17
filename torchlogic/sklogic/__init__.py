__all__ = ["LogicJEPAEncoder"]


def __getattr__(name):
    if name == "LogicJEPAEncoder":
        from .encoders import LogicJEPAEncoder
        return LogicJEPAEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
