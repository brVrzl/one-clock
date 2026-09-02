#!/usr/bin/env python3
"""CPU/network-only download of the pinned Track-C candidate checkpoint."""

from huggingface_hub import snapshot_download


print(snapshot_download(
    repo_id="SidneyXie/pi05_robotwin",
    revision="e49e2ab6c11f07511573b67261bd129e88d0a416",
    local_dir="/home/wjq/research-assets/robotwin/checkpoints/SidneyXie_pi05_robotwin",
))
