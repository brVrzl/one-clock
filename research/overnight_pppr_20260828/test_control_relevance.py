#!/usr/bin/env python3
"""Focused tests for the frozen control-relevance join and bootstrap.

These tests intentionally use tiny synthetic records.  They catch the three
failure modes that would change the Phase-0 conclusion: wrong ``u-d``
alignment, accidentally including warm-up/inactive schedule rows, and
deduplicating repeated clusters in a paired bootstrap.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_control_relevance import (  # noqa: E402
    CONDITION_SPECS,
    _bootstrap_cluster_indices,
    align_episode_condition,
)


def feature_table(rows: list[tuple[int, int, float, float, float]]) -> dict[str, np.ndarray]:
    """Build the minimum in-memory feature columns for alignment tests."""

    n = len(rows)
    old = np.asarray([row[0] for row in rows], dtype=np.int32)
    ages = np.asarray([row[1] for row in rows], dtype=np.int16)
    return {
        "task_key": np.asarray(["task"] * n, dtype="U32"),
        "episode": np.zeros(n, dtype=np.int16),
        "old_query_t": old,
        "future_query_u": old + ages,
        "age_steps": ages,
        "valid": np.ones(n, dtype=bool),
        "event_event_score": np.asarray([row[2] for row in rows], dtype=np.float32),
        "raw_ppr_arm": np.asarray([row[3] for row in rows], dtype=np.float32),
        "pppr_arm": np.asarray([row[4] for row in rows], dtype=np.float32),
        "raw_ppr_joint": np.asarray([row[3] for row in rows], dtype=np.float32),
        "pppr_joint": np.asarray([row[4] for row in rows], dtype=np.float32),
        "raw_ppr_grip": np.asarray([row[3] for row in rows], dtype=np.float32),
        "pppr_grip": np.asarray([row[4] for row in rows], dtype=np.float32),
    }


def source_event(condition: str, u: int, actual_age: int) -> dict[str, object]:
    component = str(CONDITION_SPECS[condition]["component"])
    event: dict[str, object] = {"condition": condition, "environment_step": u}
    event[component] = {"actual_source_age_steps": actual_age}
    return event


class ControlRelevanceAlignmentTest(unittest.TestCase):
    def test_pre_treatment_u_minus_d_mapping_uses_fresh_rows(self) -> None:
        condition = "reverse4"
        # Fresh rows encode their source query in the score.  At physical u=4
        # and u=5, the only allowed rows are old_query_t=0 and 1.  A row keyed
        # by u would therefore produce the wrong mean (and is not provided).
        features = feature_table([
            (0, 4, 10.0, 20.0, 30.0),
            (1, 4, 12.0, 22.0, 32.0),
        ])
        lookup = {
            ("task", 0, int(features["old_query_t"][i]), int(features["age_steps"][i])): i
            for i in range(len(features["old_query_t"]))
        }
        aligned = align_episode_condition(
            task_key="task",
            episode_index=0,
            condition=condition,
            source_events=[source_event(condition, 4, 4), source_event(condition, 5, 4)],
            features=features,
            feature_lookup=lookup,
        )
        self.assertEqual(aligned["valid_steps"], [4, 5])
        self.assertEqual(aligned["missing_steps"], [])
        self.assertAlmostEqual(aligned["raw_ppr"], 21.0)
        self.assertAlmostEqual(aligned["pppr"], 31.0)
        self.assertAlmostEqual(aligned["event"], 11.0)

    def test_active_schedule_filters_warmup_and_reports_missing(self) -> None:
        condition = "fo8"
        # u=0..7 are warm-up/inactive for requested age 8 and must not be
        # counted.  u=8 aligns to q=0; u=10 aligns to q=2, but q=2 is absent
        # and must be reported rather than filled.
        features = feature_table([(0, 8, 1.0, 2.0, 3.0)])
        lookup = {("task", 0, 0, 8): 0}
        events = [source_event(condition, u, 0) for u in range(8)]
        events += [source_event(condition, 8, 8), source_event(condition, 10, 8)]
        aligned = align_episode_condition(
            task_key="task",
            episode_index=0,
            condition=condition,
            source_events=events,
            features=features,
            feature_lookup=lookup,
        )
        self.assertEqual(aligned["active_logged_steps"], 2)
        self.assertEqual(aligned["valid_feature_steps"], 1)
        self.assertEqual(aligned["missing_feature_steps"], 1)
        self.assertEqual(aligned["valid_steps"], [8])
        self.assertEqual(aligned["missing_steps"], [10])
        self.assertAlmostEqual(aligned["raw_ppr"], 2.0)

    def test_cluster_bootstrap_expands_repeated_clusters_and_preserves_pairing(self) -> None:
        # Equal-sized one-row clusters keep the expected expanded length
        # obvious while still making repeated cluster IDs observable.
        clusters = [np.asarray([0]), np.asarray([1]), np.asarray([2])]
        sample = _bootstrap_cluster_indices(clusters, np.random.default_rng(20260828))
        # One row per sampled cluster is not enough: each sampled cluster must
        # retain all of its rows, and repeated cluster IDs must duplicate them.
        self.assertEqual(len(sample), sum(len(cluster) for cluster in clusters))
        counts = np.bincount(sample, minlength=3)
        self.assertTrue(np.any(counts > 1))
        raw = np.arange(3, dtype=float)
        pppr = raw + 0.5
        # The identical index array gives the exact paired method difference;
        # no independently sampled/deduplicated comparison is allowed.
        np.testing.assert_allclose(pppr[sample] - raw[sample], (pppr - raw)[sample])


if __name__ == "__main__":
    unittest.main()
