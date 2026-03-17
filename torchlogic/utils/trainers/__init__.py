__all__ = [
    "BanditNRNTrainer",
    "BoostedBanditNRNTrainer",
    "AttnNRNTrainer",
    "LogicJEPATrainer",
    "LogicJEPALoss",
]


def __getattr__(name):
    if name == "BanditNRNTrainer":
        from .banditnrntrainer import BanditNRNTrainer
        return BanditNRNTrainer
    if name == "BoostedBanditNRNTrainer":
        from .boostedbanditnrntrainer import BoostedBanditNRNTrainer
        return BoostedBanditNRNTrainer
    if name == "AttnNRNTrainer":
        from .attnnrntrainer import AttnNRNTrainer
        return AttnNRNTrainer
    if name in {"LogicJEPATrainer", "LogicJEPALoss"}:
        from .logicjepatrainer import LogicJEPATrainer, LogicJEPALoss
        return {"LogicJEPATrainer": LogicJEPATrainer, "LogicJEPALoss": LogicJEPALoss}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
