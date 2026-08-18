#!/usr/bin/env python3
"""Run one execution-only Gate-0 RoboTwin ACT evaluation.

This runner reuses RoboTwin's task setup/observation conversion and XPolicyLab's
ACT model. The only replacement is action selection between model queries.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml


ONE_CLOCK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ONE_CLOCK_ROOT / "src"))

from one_clock import ActionGroup, FixedChunkExecutor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ONE_CLOCK_ROOT / "configs/gate0_place_can_basket.yaml")
    parser.add_argument("--robotwin-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--strategy",
        choices=("global_fixed", "groupwise_fixed"),
        required=True,
    )
    parser.add_argument("--horizon", type=int, help="Global horizon for global_fixed.")
    parser.add_argument(
        "--group-horizons",
        type=str,
        help="Comma-separated group=horizon values for groupwise_fixed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Gate-0 config must be a mapping: {path}")
    return config


def parse_group_horizons(raw: str | None, defaults: dict[str, int]) -> dict[str, int]:
    if raw is None:
        return {name: int(value) for name, value in defaults.items()}
    result: dict[str, int] = {}
    for item in raw.split(","):
        name, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"group horizon must be name=value: {item!r}")
        result[name.strip()] = int(value)
    return result


def add_upstream_paths(robotwin_root: Path) -> None:
    for path in (
        robotwin_root,
        robotwin_root / "scripts",
        robotwin_root / "description" / "utils",
        robotwin_root / "XPolicyLab",
    ):
        path_string = str(path.resolve())
        if path_string not in sys.path:
            sys.path.insert(0, path_string)


def build_groups(config: dict[str, Any], horizons: dict[str, int]) -> tuple[ActionGroup, ...]:
    raw_groups = config["action_groups"]
    if set(raw_groups) != set(horizons):
        raise ValueError("action_groups and selected group horizons must have identical names")
    return tuple(
        ActionGroup(name, tuple(int(index) for index in raw_groups[name]), int(horizons[name]))
        for name in raw_groups
    )


def build_executor(
    config: dict[str, Any], strategy: str, horizon: int | None, group_horizons: dict[str, int]
) -> FixedChunkExecutor:
    action_dim = sum(len(indices) for indices in config["action_groups"].values())
    chunk_size = int(config["chunk_size"])
    if strategy == "global_fixed":
        if horizon is None:
            raise ValueError("--horizon is required for global_fixed")
        valid_horizons = {int(value) for value in config.get("global_horizons", [])}
        if valid_horizons and horizon not in valid_horizons:
            raise ValueError(f"global horizon must be one of {sorted(valid_horizons)}")
        groups = build_groups(config, {name: horizon for name in config["action_groups"]})
        return FixedChunkExecutor.global_fixed(
            action_dim=action_dim,
            chunk_size=chunk_size,
            horizon=horizon,
            groups=groups,
        )
    groups = build_groups(config, group_horizons)
    return FixedChunkExecutor.groupwise_fixed(
        action_dim=action_dim,
        chunk_size=chunk_size,
        groups=groups,
    )


def verify_action_dim(config: dict[str, Any]) -> int:
    """Check the configured groups against RoboTwin's own robot metadata."""

    from XPolicyLab.utils.process_data import get_robot_action_dim_info

    info = get_robot_action_dim_info(config["env_cfg_type"])
    upstream_dim = sum(info["arm_dim"]) + sum(info["ee_dim"])
    configured_dim = sum(len(indices) for indices in config["action_groups"].values())
    if configured_dim != upstream_dim:
        raise ValueError(
            f"configured action groups have dim {configured_dim}, "
            f"but RoboTwin metadata declares {upstream_dim}"
        )
    return upstream_dim


def load_act_model(robotwin_root: Path, config: dict[str, Any], checkpoint: Path):
    """Instantiate the upstream XPolicyLab ACT adapter without changing it."""

    deploy_path = robotwin_root / "XPolicyLab/policy/ACT/deploy.yml"
    deploy_config = yaml.safe_load(deploy_path.read_text(encoding="utf-8"))
    if not isinstance(deploy_config, dict):
        raise ValueError(f"ACT deploy config must be a mapping: {deploy_path}")

    action_dim = sum(len(indices) for indices in config["action_groups"].values())
    deploy_config.update(
        {
            "ckpt_dir": str(checkpoint.resolve()),
            "ckpt_name": str(checkpoint.resolve()),
            "env_cfg_type": config["env_cfg_type"],
            "action_type": config["action_type"],
            "action_dim": action_dim,
            "chunk_size": int(config["chunk_size"]),
            # Gate-0 compares execution commitments, so upstream temporal
            # aggregation must not mix predictions before our executor sees them.
            "temporal_agg": False,
        }
    )
    os.environ["ACT_ACTION_DIM"] = str(action_dim)
    from XPolicyLab.policy.ACT.model import Model

    return Model(deploy_config)


def query_full_act_chunk(act_model: Any) -> np.ndarray:
    """Use the official ACT inference call, then read its produced full chunk.

    XPolicyLab's current ``ACT.get_action`` returns only the selected row to its
    client while retaining ``all_actions`` internally. Resetting its ordinary
    cursor to zero for this explicit query exposes that same model-produced
    chunk without a second policy implementation or temporal aggregation.
    """

    act_model.t = 0
    act_model.get_action()
    normalized_chunk = act_model.all_actions[0].detach().cpu().numpy()
    return np.asarray(act_model.post_process(normalized_chunk))


def run_episode(
    *,
    official: Any,
    task_env: Any,
    task_args: dict[str, Any],
    act_wrapper: Any,
    executor: FixedChunkExecutor,
    config: dict[str, Any],
    episode_index: int,
    seed: int,
) -> tuple[bool, list[dict[str, object]]]:
    task_args = dict(task_args)
    task_args["eval_mode"] = True
    task_args["render_freq"] = 0
    task_args["eval_video_log"] = bool(config.get("eval_video_log", False))
    episode_info: dict[str, Any] = {"info": {}}

    if bool(config.get("expert_check", True)):
        task_env.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **task_args)
        episode_info = task_env.play_once()
        expert_ok = bool(task_env.plan_success and task_env.check_success())
        task_env.close_env()
        if not expert_ok:
            raise RuntimeError(f"official expert check failed for seed {seed}")

    task_env.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **task_args)
    instruction = official.build_instruction(
        task_args,
        episode_info,
        config.get("instruction_type", "seen"),
        int(config.get("episodes", 1)),
    )
    task_env.set_instruction(instruction=instruction)
    act_wrapper.reset()
    executor.reset()
    records: list[dict[str, object]] = []

    success = False
    try:
        while not official.is_episode_end(task_env):
            observation = task_env.get_obs()
            xpl_obs = official.robotwin_obs_to_xpolicylab(
                observation,
                instruction=task_env.get_instruction(),
                env_idx=0,
                frequency=int(config.get("frequency", 30)),
                task_env=task_env,
            )
            act_wrapper.update_obs(xpl_obs)
            decision = executor.step(lambda: query_full_act_chunk(act_wrapper.model))
            task_env.take_action(decision.action, action_type="qpos")
            records.append(decision.as_log_record())
        success = bool(task_env.eval_success)
    finally:
        task_env.close_env()

    return success, records


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.get("action_type", "joint") != "joint":
        raise ValueError("Gate-0 currently supports only the audited joint RoboTwin action schema")
    robotwin_root_value = args.robotwin_root or os.environ.get("ROBOTWIN_ROOT")
    if not robotwin_root_value:
        raise ValueError("--robotwin-root or ROBOTWIN_ROOT is required")
    robotwin_root = Path(robotwin_root_value)
    checkpoint_value = args.checkpoint or config.get("checkpoint")
    if not checkpoint_value:
        raise ValueError("--checkpoint or config checkpoint is required")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"ACT checkpoint directory does not exist: {checkpoint}")

    add_upstream_paths(robotwin_root)
    import eval_policy_xpolicylab as official

    task_name = str(config["task_name"])
    usr_args = {
        "task_name": task_name,
        "task_config": config.get("task_config", "demo_clean"),
        "policy_name": "ACT",
        "ckpt_setting": str(checkpoint),
        "action_type": config.get("action_type", "joint"),
        "seed": int(config.get("seed", 0)),
    }
    task_args, _ = official.load_task_args(usr_args)
    action_dim = verify_action_dim(config)
    group_horizons = parse_group_horizons(
        args.group_horizons,
        {name: int(value) for name, value in config["groupwise_horizons"].items()},
    )
    executor = build_executor(config, args.strategy, args.horizon, group_horizons)
    if executor.action_dim != action_dim:
        raise ValueError("executor action dimension does not match verified RoboTwin metadata")
    act_wrapper = load_act_model(robotwin_root, config, checkpoint)
    task_env = official.class_decorator(task_name)

    output_dir = args.output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    metadata = {
        "strategy": args.strategy,
        "global_horizon": args.horizon,
        "group_horizons": group_horizons,
        "action_groups": config["action_groups"],
        "task_name": task_name,
        "task_config": config.get("task_config", "demo_clean"),
        "action_type": config.get("action_type", "joint"),
        "chunk_size": int(config["chunk_size"]),
        "checkpoint": str(checkpoint.resolve()),
        "robotwin_root": str(robotwin_root.resolve()),
        "robotwin_commit": git_commit(robotwin_root),
        "xpolicylab_commit": git_commit(robotwin_root / "XPolicyLab"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    successes = 0
    with (output_dir / "steps.jsonl").open("w", encoding="utf-8") as log_file:
        for episode_index in range(int(config.get("episodes", 1))):
            success, records = run_episode(
                official=official,
                task_env=task_env,
                task_args=task_args,
                act_wrapper=act_wrapper,
                executor=executor,
                config=config,
                episode_index=episode_index,
                seed=int(config.get("seed", 0)) + episode_index,
            )
            successes += int(success)
            for record in records:
                record["episode"] = episode_index
                log_file.write(json.dumps(record) + "\n")

    episodes = int(config.get("episodes", 1))
    summary = {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
