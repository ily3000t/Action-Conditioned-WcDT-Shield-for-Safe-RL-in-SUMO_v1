from safe_rl.data.types import InterventionRecord
from safe_rl.eval.evaluator import SafeRLEvaluator
from safe_rl.pipeline.telemetry import BufferTelemetryTracker, Stage3TelemetryTracker
from safe_rl.config.config import EvalConfig


class _FakeWriter:
    def __init__(self):
        self.records = []

    def add_scalar(self, tag, value, step):
        self.records.append((tag, float(value), int(step)))

    def values_for(self, tag):
        return [value for current_tag, value, _ in self.records if current_tag == tag]


class _FakePolicy:
    def predict(self, obs, deterministic=True):
        _ = (obs, deterministic)
        return 0


class _FakeEvalEnv:
    def __init__(self, episodes):
        self.episodes = episodes
        self.episode_idx = -1
        self.step_idx = 0
        self.reset_seeds = []

    def reset(self, seed=None, options=None):
        _ = options
        self.reset_seeds.append(seed)
        self.episode_idx += 1
        self.step_idx = 0
        return 0.0, {
            "episode_id": f"eval_ep_{self.episode_idx:04d}",
            "risky_mode": True,
            "scenario_source": "scenarios/highway_merge/highway_merge.sumocfg",
        }

    def step(self, action):
        _ = action
        info = dict(self.episodes[self.episode_idx][self.step_idx])
        self.step_idx += 1
        terminated = self.step_idx >= len(self.episodes[self.episode_idx])
        truncated = False
        return 0.0, float(info.get("reward", 0.0)), terminated, truncated, info


def test_stage3_telemetry_tracker_logs_stability_and_risk():
    writer = _FakeWriter()
    tracker = Stage3TelemetryTracker(writer)

    tracker.handle_session_event({"event": "episode_reset_started", "episode_id": "ep1"})
    tracker.on_step(1, {"episode_id": "ep1", "risk_raw": 0.8, "risk_final": 0.3, "intervened": True})
    tracker.handle_session_event({"event": "restart_real_session", "episode_id": "ep1"})
    tracker.handle_session_event({"event": "reset_load_failed", "episode_id": "ep1"})
    tracker.handle_session_event({"event": "fatal_step", "episode_id": "ep1"})
    tracker.handle_session_event({"event": "episode_completed", "episode_id": "ep1"})
    tracker.handle_session_event({"event": "episode_reset_started", "episode_id": "ep2"})
    tracker.handle_session_event({"event": "episode_reset_failed", "episode_id": "ep2"})

    assert writer.values_for("stability/reset_started_count")[-1] == 2.0
    assert writer.values_for("stability/reset_failed_count")[-1] == 1.0
    assert writer.values_for("stability/load_failed_count")[-1] == 1.0
    assert writer.values_for("stability/restart_count")[-1] == 1.0
    assert writer.values_for("stability/fatal_step_count")[-1] == 1.0
    assert writer.values_for("stability/episode_restart_count")[0] == 1.0
    assert writer.values_for("stability/episode_had_restart")[0] == 1.0
    assert writer.values_for("stability/episode_had_fatal_step")[0] == 1.0
    assert writer.values_for("stability/episode_had_reset_error")[-1] == 1.0
    assert writer.values_for("risk/step_raw")[-1] == 0.8
    assert writer.values_for("risk/step_final")[-1] == 0.3
    assert writer.values_for("risk/step_reduction")[-1] == 0.5
    assert writer.values_for("risk/episode_mean_reduction")[-1] == 0.5



def test_buffer_telemetry_tracker_logs_process_metrics():
    writer = _FakeWriter()
    tracker = BufferTelemetryTracker(writer)

    step_info = {
        "episode_id": "buffer_ep_1",
        "intervened": True,
        "risk_raw": 0.9,
        "risk_final": 0.4,
    }
    tracker.on_step(step_info)
    record = InterventionRecord(
        history_scene=[],
        raw_action=0,
        final_action=1,
        raw_risk=0.9,
        final_risk=0.4,
        reason="risk_threshold_exceeded",
        meta={"episode_id": "buffer_ep_1"},
    )
    tracker.on_push(
        record,
        {
            "size": 1.0,
            "mean_raw_risk": 0.9,
            "mean_final_risk": 0.4,
            "mean_risk_reduction": 0.5,
            "replacement_count": 1.0,
        },
    )
    tracker.on_episode_end("buffer_ep_1")

    assert writer.values_for("buffer/step_raw_risk")[-1] == 0.9
    assert writer.values_for("buffer/push_raw_risk")[-1] == 0.9
    assert writer.values_for("buffer/size")[-1] == 1.0
    assert writer.values_for("buffer/running_mean_risk_reduction")[-1] == 0.5
    assert writer.values_for("buffer/episode_pushes")[-1] == 1.0
    assert writer.values_for("buffer/episode_mean_risk_reduction")[-1] == 0.5



def test_evaluator_aggregates_risk_metrics_for_shielded_policy():
    env = _FakeEvalEnv(
        episodes=[
            [
                {
                    "collision": False,
                    "intervened": True,
                    "ego_speed": 10.0,
                    "risk_raw": 0.8,
                    "risk_final": 0.3,
                    "reward": 1.0,
                    "raw_action": 4,
                    "final_action": 1,
                    "replacement_happened": True,
                    "replacement_count": 1,
                    "replacement_same_as_raw_count": 0,
                    "fallback_action_count": 0,
                    "shield_called_steps": 1,
                    "shield_candidate_evaluated_steps": 1,
                    "shield_blocked_steps": 1,
                    "shield_replaced_steps": 1,
                },
                {
                    "collision": False,
                    "intervened": False,
                    "ego_speed": 12.0,
                    "risk_raw": 0.6,
                    "risk_final": 0.4,
                    "reward": 2.0,
                    "raw_action": 2,
                    "final_action": 2,
                    "replacement_happened": False,
                    "replacement_count": 0,
                    "replacement_same_as_raw_count": 0,
                    "fallback_action_count": 0,
                    "shield_called_steps": 1,
                    "shield_candidate_evaluated_steps": 1,
                    "shield_blocked_steps": 0,
                    "shield_replaced_steps": 0,
                },
            ]
        ]
    )
    evaluator = SafeRLEvaluator(EvalConfig())

    metrics = evaluator.evaluate_policy(env, _FakePolicy(), episodes=1, risky_mode=True, tb_writer=None, tb_prefix="shielded")

    assert metrics["mean_raw_risk"] == 0.7
    assert metrics["mean_final_risk"] == 0.35
    assert metrics["mean_risk_reduction"] == 0.35
    assert metrics["replacement_count"] == 1.0
    assert metrics["episode_details"][0]["replacement_count"] == 1
    assert metrics["episode_details"][0]["scenario_source"].endswith("highway_merge.sumocfg")



def test_evaluator_uses_proxy_risk_for_baseline_policy():
    env = _FakeEvalEnv(
        episodes=[
            [
                {
                    "collision": False,
                    "intervened": False,
                    "ego_speed": 10.0,
                    "risk_raw": 0.0,
                    "risk_final": 0.0,
                    "min_distance": 5.0,
                    "ttc": 2.0,
                    "reward": 1.0,
                    "raw_action": 4,
                    "final_action": 4,
                    "replacement_happened": False,
                    "replacement_count": 0,
                    "replacement_same_as_raw_count": 0,
                    "fallback_action_count": 0,
                    "shield_called_steps": 0,
                    "shield_candidate_evaluated_steps": 0,
                    "shield_blocked_steps": 0,
                    "shield_replaced_steps": 0,
                },
            ]
        ]
    )
    evaluator = SafeRLEvaluator(EvalConfig())

    metrics = evaluator.evaluate_policy(env, _FakePolicy(), episodes=1, risky_mode=True, tb_writer=None, tb_prefix="baseline")

    assert metrics["mean_raw_risk"] > 0.0
    assert metrics["mean_final_risk"] == metrics["mean_raw_risk"]
    assert metrics["mean_risk_reduction"] == 0.0



def test_evaluator_passes_paired_seeds_to_env_reset():
    env = _FakeEvalEnv(
        episodes=[
            [{"collision": False, "intervened": False, "ego_speed": 8.0, "risk_raw": 0.4, "risk_final": 0.4, "reward": 1.0, "raw_action": 0, "final_action": 0, "replacement_happened": False, "replacement_count": 0, "replacement_same_as_raw_count": 0, "fallback_action_count": 0, "shield_called_steps": 0, "shield_candidate_evaluated_steps": 0, "shield_blocked_steps": 0, "shield_replaced_steps": 0}],
            [{"collision": False, "intervened": True, "ego_speed": 9.0, "risk_raw": 0.7, "risk_final": 0.2, "reward": 2.0, "raw_action": 4, "final_action": 1, "replacement_happened": True, "replacement_count": 1, "replacement_same_as_raw_count": 0, "fallback_action_count": 0, "shield_called_steps": 1, "shield_candidate_evaluated_steps": 1, "shield_blocked_steps": 1, "shield_replaced_steps": 1}],
        ]
    )
    evaluator = SafeRLEvaluator(EvalConfig())

    metrics = evaluator.evaluate_policy(
        env,
        _FakePolicy(),
        episodes=2,
        risky_mode=True,
        tb_writer=None,
        tb_prefix="shielded",
        seeds=[11, 22],
    )

    assert env.reset_seeds == [11, 22]
    assert [detail["seed"] for detail in metrics["episode_details"]] == [11, 22]
