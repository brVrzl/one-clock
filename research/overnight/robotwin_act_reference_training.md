# RoboTwin ACT reference training — 2026-08-24

All runs in this file are `INFRASTRUCTURE-ONLY` and `NOT-PAPER-EVIDENCE`.
They are not task-success experiments.

## Protocol

- RoboTwin commit: `30954692d06ba7e89f7a6b76064f4062c488fa81`
- XPolicyLab commit: `c37109c500be67d0dea6b36bf7337bbd26e763cd`
- Dataset: official `TianxingChen/RoboTwin2.0`, `demo_clean`,
  `aloha_agilex`, 50 trajectories per task.
- ACT command family: pinned `XPolicyLab/policy/ACT/train.sh`, official
  defaults, seed 0, 6000 epochs, no early stopping or hyperparameter tuning.
- GPU 0 was excluded because it hosted an existing scientific process.

## beat_block_hammer

- Status: COMPLETE; GPU 1.
- Process log: external `overnight_20260824/logs/act_beat_block_hammer_train.log`.
- Data: `processed_data/demo_clean/beat_block_hammer/aloha_agilex-joint`,
  50 episodes.
- Action contract before training: qpos/action `(T, 14)`, six joints plus one
  end-effector value per arm; three camera streams; ACT chunk size 50 in the
  pinned training script.
- Checkpoints:
  - `.../checkpoints/demo_clean-beat_block_hammer-aloha_agilex-joint-0/policy_last.ckpt`
    (335,907,847 bytes; SHA256
    `7f3a058419b82464aeeb48d414a8b948eba55220ff5b4b82f16385a0383862fd`)
  - `.../policy_epoch_6000_seed_0.ckpt` (335,912,589 bytes; SHA256
    `961e7e3902e1af3731daaf15921b703c73ad7c9e00351b09cbab2dc8f889bd50`)
  - `dataset_stats.pkl` (7,922 bytes; SHA256
    `993667e6cd58e278652fa8591c55dd7548760311ee64f12c4bf98eace6471492`)
- Offline contract audit passed on one recorded observation from
  `episode_0.hdf5`: wrapper prefix loaded with zero missing/unexpected keys;
  output chunk `(1, 50, 14)`, padding head `(1, 50, 1)`, finite outputs;
  qpos `(1, 14)` and three camera tensors `(1, 3, 3, 480, 640)` after the
  recorded preprocessing. Normalization keys were qpos/action mean and
  standard deviation plus example qpos. No closed-loop action was sent.

## adjust_bottle

- Status: COMPLETE; GPU 2; official 6000-step protocol completed.
- Preprocessing is complete at
  `processed_data/demo_clean/adjust_bottle/aloha_agilex-joint`, 50 episodes.
- Process log: external `overnight_20260824/logs/act_adjust_bottle_train.log`.
- Checkpoints:
  - `.../checkpoints/demo_clean-adjust_bottle-aloha_agilex-joint-0/policy_last.ckpt`
    (335,907,847 bytes; SHA256
    `bba4e4357c2314de37664cbbaa552d8f9b8800c7a6cda17a948528648002c927`)
  - `.../policy_epoch_6000_seed_0.ckpt` (335,912,589 bytes; SHA256
    `74c3dd408c08cb8257cf49f5324f7028b14cb6c937401038d6fab4a1f2c1d7c5`)
  - `dataset_stats.pkl` (10,610 bytes; SHA256
    `871f255367613929739e6439987fc7e4a817f6f94bc754255b0dd9573532384f`)
- Offline contract audit passed on one recorded observation from
  `episode_0.hdf5`: wrapper prefix loaded with zero missing/unexpected keys;
  output chunk `(1, 50, 14)`, padding head `(1, 50, 1)`, finite outputs;
  qpos `(1, 14)` and three camera tensors `(1, 3, 3, 480, 640)`. No
  closed-loop action was sent.

## Evaluation boundary

No policy evaluation or closed-loop episode is authorized by this record.
