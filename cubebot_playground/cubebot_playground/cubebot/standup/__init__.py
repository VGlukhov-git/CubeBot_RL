"""Stand-up training and lightweight hardware inference."""

__all__ = ["CubebotStandUp", "StandUpConfig"]


def __getattr__(name):
    # Keep MuJoCo and mjbatch out of Raspberry Pi inference imports.
    if name in __all__:
        from .environment import CubebotStandUp, StandUpConfig

        return {"CubebotStandUp": CubebotStandUp, "StandUpConfig": StandUpConfig}[name]
    raise AttributeError(name)
