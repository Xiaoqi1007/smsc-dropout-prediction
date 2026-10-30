"""Session-window configuration."""


def session_bounds(config):
    window = config.get("session_window")
    if window is None:
        return None
    if not isinstance(window, dict) or set(window) != {"min", "max"}:
        raise ValueError("session_window must contain integer min and max")
    lo, hi = window["min"], window["max"]
    if any(type(x) is not int for x in (lo, hi)) or not 0 <= lo <= hi:
        raise ValueError(
            "session_window must satisfy 0 <= min <= max with integer bounds"
        )
    if config.get("analysis_targets", config.get("targets")) != ["ever_dropout"]:
        raise ValueError("The session-window workflow predicts ever_dropout only")
    return lo, hi
