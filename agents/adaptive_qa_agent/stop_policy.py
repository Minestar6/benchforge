from dataclasses import dataclass


@dataclass
class StopDecision:
    should_stop: bool
    reason: str = ""


class StopPolicy:
    def __init__(self, adaptive_config):
        self.cfg = adaptive_config.stop

    def check(self, state) -> StopDecision:
        if state.round_in_mode >= state.max_rounds:
            return StopDecision(True, "max_rounds")
        if state.accepted_count >= state.target_candidates:
            return StopDecision(True, "target_reached")
        if state.consecutive_empty >= self.cfg.max_empty_rounds:
            return StopDecision(True, "max_empty_rounds")
        if state.failures >= self.cfg.max_failures:
            return StopDecision(True, "max_failures")
        return StopDecision(False)
