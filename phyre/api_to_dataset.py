"""
according to the slotformer paper....

PHYRE (Bakhtin et al., 2019). We study the PHYRE-B tier in this paper, which consists of 25
templates of tasks. Tasks within the same template share similar initial configuration of the objects.
There are two evaluation settings, namely, within-template, where training and testing tasks come
from the same templates, and cross-template, where train-test tasks are from different templates. We
simulate the videos in 1 FPS as done in previous works (Bakhtin et al., 2019; Qi et al., 2020).
 We experiment on the within-template setting
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

import ffmpeg
import numpy as np
import phyre


def save_video(frames: np.ndarray, path: Path, fps: int) -> None:
    
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    (
        ffmpeg.input(    
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s=f"{w}x{h}",
            r=fps,
        )
        .output(
            str(path),
            vcodec="libx264",
            pix_fmt="yuv420p",
            preset="fast",
            movflags="+faststart",
        )
        .overwrite_output()
        .run(
            input=frames.tobytes(),
            capture_stdout=True,
            capture_stderr=True,
        )
    ) # T,H,W,3 uint8 RGB


def rollout_split(
    split_name: str,
    task_ids: Iterable[str],
    action_tier: str,
    output_dir: Path,
    actions_per_task: int,
    stride: int,
    max_steps: int,  # not supported anymore
    fps: int,
) -> dict:
    simulator = phyre.initialize_simulator(task_ids, action_tier)
    actions = simulator.build_discrete_action_space(actions_per_task)

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for task_index, task_id in enumerate(simulator.task_ids):
        for action_index, action in enumerate(actions):
            result = simulator.simulate_action(
                task_index,
                action,
                stride=stride,
                # max_steps=max_steps,
                need_images=True,
            )

            rgb = phyre.observations_to_float_rgb(result.images)
            frames = (np.asarray(rgb) * 255).astype(np.uint8)
            # frames = np.asarray(rgb, dtype=np.uint8)
            video_name = f"{task_id}_a{action_index}.mp4"
            video_path = split_dir / video_name
            save_video(frames, video_path, fps=fps)

            records.append(
                {
                    "video_path": video_name,
                    "num_frames": int(frames.shape[0]),
                    "task_id": task_id,
                    "action_index": action_index,
                    "status": str(result.status),
                }
            )

    return {"root_dir": str(split_dir), "records": records}


def convert(args: argparse.Namespace) -> None:
    # Ensure deterministic ordering/action sampling.
    random.seed(args.seed)
    np.random.seed(args.seed)
    if hasattr(phyre, "seed"):
        phyre.seed(args.seed)

    # eval_setups = args.setups or phyre.MAIN_EVAL_SETUPS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    

    # for args.setup in args.setup:
    (output_dir / args.setup).mkdir(parents=True, exist_ok=True)
    action_tier = phyre.eval_setup_to_action_tier(args.setup)
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(args.setup, args.fold_id)
    os.makedirs(args.setup, exist_ok=True)
    split_map = {
        "train": train_tasks,
        "validation": dev_tasks,
        "test": test_tasks,
    }
    if args.only_split:
        split_map = {args.only_split: split_map[args.only_split]}

    for split_name, split_tasks in split_map.items():
        print(f"[{args.setup}] Converting {split_name} with {len(split_tasks)} tasks...")
        data = rollout_split(
            split_name=split_name,
            task_ids=split_tasks,
            action_tier=action_tier,
            output_dir=output_dir / args.setup,
            actions_per_task=args.actions_per_task,
            stride=args.stride,
            max_steps=args.max_steps,
            fps=args.fps,
        )
        
        json_path =  args.setup / f"{split_name}.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  wrote {json_path} with {len(data['records'])} videos")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PHYRE tasks to a video dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../../../scratch/phyre_videos",
        help="Where to save videos.",
    )
    parser.add_argument("--fold-id", type=int, default=0, help="PHYRE fold id to export.")
    parser.add_argument("--actions-per-task",type=int,default=8,help="Number of discrete actions to sample per task." )
    parser.add_argument("--max-steps", type=int, default=30, help="Max simulator steps per rollout.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride when simulating.")
    parser.add_argument("--fps", type=int, default=30, help="FPS used when writing videos.")
    parser.add_argument("--setup",type=str,default="ball_within_template",help="Eval args.setups to export")
    parser.add_argument( "--only-split",type=str,choices=["train", "validation", "test"],help="Export a single split instead of all three.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    convert(args)
