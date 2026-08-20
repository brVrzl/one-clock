from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.dynamic_reliability_horizon.robotwin50_dataset.target_builder import (
    GROUP_INDICES,
    GROUP_NAMES,
    build_refresh_targets,
)
from experiments.dynamic_reliability_horizon.robotwin50_dataset.smoke_check import (
    PREDECLARED_TASKS,
    _snapshot_contract,
)
from experiments.dynamic_reliability_horizon.robotwin50_dataset.data_audit import _split
from experiments.dynamic_reliability_horizon.robotwin50_dataset.cache_builder import (
    ContractMismatch,
    audit_checkpoint_contract,
)


def test_robotwin_groups_are_verified_four_clock_partition() -> None:
    flattened = np.concatenate([GROUP_INDICES[name] for name in GROUP_NAMES])
    np.testing.assert_array_equal(flattened, np.arange(14))
    assert GROUP_INDICES["left_arm"].tolist() == [0, 1, 2, 3, 4, 5]
    assert GROUP_INDICES["left_gripper"].tolist() == [6]
    assert GROUP_INDICES["right_arm"].tolist() == [7, 8, 9, 10, 11, 12]
    assert GROUP_INDICES["right_gripper"].tolist() == [13]


def test_refresh_targets_keep_raw_distance_and_prefix_survival() -> None:
    chunks = np.zeros((3, 4, 14), dtype=np.float32)
    chunks[0, 1, 0] = 1.0
    targets = build_refresh_targets(
        chunks,
        np.asarray([0, 1, 2]),
        thresholds={name: 0.05 for name in GROUP_NAMES},
        chunk_size=4,
    )

    assert targets["censor_mask"][0, 0, 0] == 1
    assert targets["validity"][0, 0, 0] == 1
    assert targets["validity"][0, 0, 1] == 0
    assert targets["y_refresh"][0, 0, 0] == 1
    assert targets["y_refresh"][0, 0, 1] == 0
    assert targets["y_refresh"][0, 0, 2] == 0
    assert targets["raw_distances"][0, 0, 1] == 1.0
    assert np.isnan(targets["raw_distances"][2, 0, 1])
    assert targets["censor_mask"][2, 0, 1] == 0


def test_refresh_targets_use_recorded_frame_ids_and_censor_gaps() -> None:
    chunks = np.zeros((2, 3, 14), dtype=np.float32)
    targets = build_refresh_targets(
        chunks,
        np.asarray([10, 12]),
        thresholds={name: 0.05 for name in GROUP_NAMES},
        chunk_size=3,
    )
    assert targets["censor_mask"][0, 0, 0] == 1
    assert targets["censor_mask"][0, 0, 1] == 0
    assert targets["censor_mask"][0, 0, 2] == 1
    assert targets["validity"][0, 0, 2] == 1
    assert targets["y_refresh"][0, 0, 2] == 0


def test_smoke_contract_reports_state_mismatch_without_forwarding() -> None:
    dataset_info = {
        "features": {
            "observation.state": {"shape": [14]},
            "action": {"shape": [14]},
        }
    }
    config = {
        "input_features": {"observation.state": {"shape": [6]}},
        "output_features": {"action": {"shape": [14]}},
        "chunk_size": 50,
        "n_action_steps": 50,
        "n_obs_steps": 1,
        "num_steps": 10,
    }
    preprocessor = {
        "camera_rename_map": {
            "observation.images.cam_high": "observation.images.camera1",
            "observation.images.cam_left_wrist": "observation.images.camera2",
            "observation.images.cam_right_wrist": "observation.images.camera3",
        }
    }
    contract = _snapshot_contract(dataset_info, config, preprocessor)
    assert contract["mismatches"] == ["state shape dataset=[14] policy=[6]"]
    assert [item["task"] for item in PREDECLARED_TASKS] == [
        "move_can_pot",
        "pick_dual_bottles",
        "handover_block",
        "stack_blocks_two",
        "place_object_scale",
        "stack_blocks_three",
    ]


def test_episode_split_is_deterministic_task_bucket_holdout() -> None:
    rows = [
        {"episode_index": episode, "tasks": np.asarray([f"task-{episode // 2}"]) }
        for episode in range(20)
    ]
    first = _split(rows, seed=20260820)
    second = _split(rows, seed=20260820)
    assert first == second
    assert sorted(sum((values for values in first.values()), []), key=int) == [str(i) for i in range(20)]
    locations = {episode: split for split, values in first.items() for episode in values}
    for start in range(0, 20, 2):
        assert locations[str(start)] == locations[str(start + 1)]


def test_cache_contract_requires_policy_action_order_sidecar(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    (dataset_root / "meta").mkdir(parents=True)
    checkpoint.mkdir()
    names = [f"joint-{i}" for i in range(14)]
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.state": {"shape": [14]},
                    "action": {"shape": [14], "names": {"motors": names}},
                }
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "input_features": {"observation.state": {"shape": [14]}},
                "output_features": {"action": {"shape": [14]}},
                "chunk_size": 50,
                "n_action_steps": 50,
                "n_obs_steps": 1,
                "num_steps": 10,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "rename_observations_processor",
                        "config": {
                            "rename_map": {
                                "observation.images.cam_high": "observation.images.camera1",
                                "observation.images.cam_left_wrist": "observation.images.camera2",
                                "observation.images.cam_right_wrist": "observation.images.camera3",
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractMismatch, match="exact policy action ordering"):
        audit_checkpoint_contract(dataset_root, checkpoint)
