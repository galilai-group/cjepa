"""Benchmark script comparing wrapped TIMM ViT with masking vs baseline.

Usage:
    python benchmark_masked_vit.py --batch-size 32 --drop-ratio 0.75
"""

import argparse
import time
import torch
import torch.nn as nn
import timm
import numpy as np
import gc

from stable_pretraining.data.transforms import PatchMasking
from stable_pretraining.backbone import EfficientMaskedTimmViT, TeacherStudentWrapper


class BaselineViT(nn.Module):
    """Standard TIMM ViT that processes all patches (including masked ones)."""

    def __init__(self, model_name="vit_base_patch16_224", pretrained=False):
        super().__init__()
        self.vit = timm.create_model(
            model_name, pretrained=pretrained, num_classes=1000
        )

    def forward(self, x):
        return self.vit(torch.nan_to_num(x, nan=0.0))


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(
        f"Device: {device} | Batch: {args.batch_size} | Drop ratio: {args.drop_ratio}"
    )

    # Create masked data
    print("\nGenerating masked data...")
    masking = PatchMasking(
        patch_size=args.patch_size,
        drop_ratio=args.drop_ratio,
        source="image",
        target="image",
        fill_value=float("nan") if args.use_nan else 0.0,
    )

    data = []
    for _ in range(args.num_iterations):
        batch = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
        masked_batch = torch.stack([masking({"image": img})["image"] for img in batch])
        data.append(masked_batch.to(device))

    # Create models
    print("Creating models...")
    baseline = BaselineViT(args.model_name, args.pretrained).train()
    vit = timm.create_model(
        args.model_name, pretrained=args.pretrained, num_classes=1000
    )
    wrapped = EfficientMaskedTimmViT(vit).train()
    if args.teacher_student:
        wrapped = TeacherStudentWrapper(
            wrapped,
            warm_init=True,
            base_ema_coefficient=0.994,
            final_ema_coefficient=0.998,
        )
        baseline = TeacherStudentWrapper(
            baseline,
            warm_init=True,
            base_ema_coefficient=0.994,
            final_ema_coefficient=0.998,
        )
    wrapped = wrapped.to(device)
    baseline = baseline.to(device)

    # Benchmark function
    def benchmark(model, name):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        gc.collect()

        # Warmup
        for batch in data[:3]:
            if args.teacher_student:
                output = model.forward_student(batch)
            else:
                output = model(batch)
            loss = output.mean()
            loss.backward()
            model.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Benchmark
        timings = []
        for batch in data:
            start = time.perf_counter()
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            if args.teacher_student:
                output = model.forward_student(batch)
            else:
                output = model(batch)
            loss = output.mean()
            loss.backward()
            model.zero_grad()

            if device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event) / 1000)
            else:
                timings.append(time.perf_counter() - start)

        mem = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0
        )
        return {
            "mean_time": np.mean(timings),
            "std_time": np.std(timings),
            "throughput": args.batch_size / np.mean(timings),
            "memory_mb": mem,
        }

    # Run benchmarks
    print(f"\nBenchmarking ({args.num_iterations} iterations)...")
    baseline_results = benchmark(baseline, "Baseline")
    wrapped_results = benchmark(wrapped, "Wrapped")

    # Print results
    print(f"\n{'=' * 70}")
    print(f"{'Metric':<25} {'Baseline':<20} {'Wrapped':<20} {'Ratio':<10}")
    print(f"{'-' * 70}")
    print(
        f"{'Time (s)':<25} {baseline_results['mean_time']:.4f} ± {baseline_results['std_time']:.4f}"
        f"{'':<6} {wrapped_results['mean_time']:.4f} ± {wrapped_results['std_time']:.4f}"
        f"{'':<6} {baseline_results['mean_time'] / wrapped_results['mean_time']:.2f}x"
    )
    print(
        f"{'Throughput (img/s)':<25} {baseline_results['throughput']:.2f}"
        f"{'':<14} {wrapped_results['throughput']:.2f}"
        f"{'':<14} {wrapped_results['throughput'] / baseline_results['throughput']:.2f}x"
    )

    if device.type == "cuda":
        print(
            f"{'Memory (MB)':<25} {baseline_results['memory_mb']:.2f}"
            f"{'':<14} {wrapped_results['memory_mb']:.2f}"
            f"{'':<14} {baseline_results['memory_mb'] / wrapped_results['memory_mb']:.2f}x"
        )
        print(
            f"{'Memory saved (MB)':<25} {'':<20} {baseline_results['memory_mb'] - wrapped_results['memory_mb']:.2f}"
        )

    print(f"{'=' * 70}")
    speedup = baseline_results["mean_time"] / wrapped_results["mean_time"]
    print(
        f"\nWrapped is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}, "
        f"processing {(1 - args.drop_ratio) * 100:.0f}% of patches"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark wrapped ViT with masking")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-iterations", type=int, default=100, help="Iterations")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size")
    parser.add_argument(
        "--teacher-student", action="store_true", help="Use teacher student wrapper"
    )
    parser.add_argument("--drop-ratio", type=float, default=0.75, help="Masking ratio")
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-nan", action="store_true", help="Use NaN for masking")

    args = parser.parse_args()
    main(args)
