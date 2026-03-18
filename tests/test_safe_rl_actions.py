import os

from safe_rl.sim.actions import action_distance, decode_action, encode_action, fallback_action_id
from safe_rl.shield.candidate_generator import CandidateActionGenerator


def test_action_encode_decode_roundtrip():
    for lon in (-1, 0, 1):
        for lat in (-1, 0, 1):
            action_id = encode_action(lon, lat)
            action = decode_action(action_id)
            assert action.longitudinal == lon
            assert action.lateral == lat


def test_fallback_is_decel_keep():
    action = decode_action(fallback_action_id())
    assert action.longitudinal == -1
    assert action.lateral == 0


def test_candidate_generation_size_and_raw_first():
    generator = CandidateActionGenerator(candidate_count=7)
    candidates = generator.generate(4)
    assert len(candidates) == 7
    assert candidates[0] == 4


def test_action_distance_non_negative():
    assert action_distance(0, 8) >= 0
