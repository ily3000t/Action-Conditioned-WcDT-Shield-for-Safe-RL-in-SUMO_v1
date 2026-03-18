from typing import List

from safe_rl.sim.actions import fallback_action_id, neighboring_actions


class CandidateActionGenerator:
    def __init__(self, candidate_count: int = 7):
        if candidate_count < 3:
            raise ValueError("candidate_count must be >= 3")
        self.candidate_count = candidate_count

    def generate(self, raw_action: int) -> List[int]:
        candidates = neighboring_actions(raw_action)
        if raw_action not in candidates:
            candidates.insert(0, raw_action)
        if fallback_action_id() not in candidates:
            candidates.append(fallback_action_id())

        # Keep deterministic ordering with raw action first.
        ordered = [raw_action] + [a for a in candidates if a != raw_action]
        deduped = []
        for action in ordered:
            if action not in deduped:
                deduped.append(action)
            if len(deduped) >= self.candidate_count:
                break
        return deduped
