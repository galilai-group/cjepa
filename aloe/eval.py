"""Evaluate ALOE on CLEVRER validation data or write a test submission."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from .data import load_vocab, make_dataset, make_loader
from .rollout import resolve_device
from .train import ALOEModule, batch_accuracy


def add_predictions(results, batch, output, label_to_answer):
    cls_logits = output["cls_answer_logits"]
    if cls_logits is not None:
        answers = cls_logits.argmax(-1).cpu().tolist()
        scenes = batch["cls_scene_index"].cpu().tolist()
        questions = batch["cls_question_id"].cpu().tolist()
        for scene, question, answer in zip(scenes, questions, answers, strict=True):
            results[scene][question] = {
                "question_id": question,
                "answer": label_to_answer[answer],
            }

    mc_logits = output["mc_answer_logits"]
    if mc_logits is not None:
        answers = (mc_logits > 0).cpu().tolist()
        flags = batch["mc_flag"].cpu()
        choice_ids = batch["mc_choice_id"].cpu()
        scenes = batch["mc_scene_index"].cpu().tolist()
        questions = batch["mc_question_id"].cpu().tolist()
        for index, (scene, question) in enumerate(zip(scenes, questions, strict=True)):
            mask = flags == index
            choices = [
                {
                    "choice_id": int(choice),
                    "answer": "correct" if answer else "wrong",
                }
                for choice, answer in zip(
                    choice_ids[mask].tolist(),
                    torch.tensor(answers)[mask].tolist(),
                    strict=True,
                )
            ]
            results[scene][question] = {
                "question_id": question,
                "choices": choices,
            }


@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg: DictConfig):
    split = cfg.eval.split
    if split not in {"val", "test"}:
        raise ValueError("eval.split must be 'val' or 'test'")
    checkpoint = Path(cfg.eval.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"ALOE checkpoint not found: {checkpoint}")
    device = resolve_device(cfg.eval.device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    module = ALOEModule(saved["hyper_parameters"]["config"])
    module.load_state_dict(saved["state_dict"])
    model = module.model.to(device).eval()
    dataset = make_dataset(cfg, split)
    loader = make_loader(dataset, cfg, train=False)

    vocab = load_vocab()
    label_to_answer = {value: key for key, value in vocab["a_vocab"].items()}
    results = defaultdict(dict)
    totals = {
        name: [0, 0]
        for name in (
            "descriptive",
            "multiple-choice",
            "explanatory",
            "predictive",
            "counterfactual",
        )
    }

    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"ALOE {split}")):
            if (
                cfg.eval.limit_batches is not None
                and batch_index >= cfg.eval.limit_batches
            ):
                break
            device_batch = {key: value.to(device) for key, value in batch.items()}
            output = model(device_batch)
            add_predictions(results, batch, output, label_to_answer)
            if split == "val":
                for name, (correct, total) in batch_accuracy(
                    device_batch, output
                ).items():
                    totals[name][0] += correct
                    totals[name][1] += total

    serialized = [
        {
            "scene_index": scene,
            "questions": [questions[key] for key in sorted(questions)],
        }
        for scene, questions in sorted(results.items())
    ]
    output_path = Path(cfg.eval.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(serialized, file)
    print(f"Predictions: {output_path}")
    if split == "val":
        for name, (correct, total) in totals.items():
            message = (
                f"{name:>16}: {correct / total:.4f} ({correct}/{total})"
                if total
                else f"{name:>16}: n/a"
            )
            print(message)


if __name__ == "__main__":
    run()
