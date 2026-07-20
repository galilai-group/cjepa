import json

import h5py
import numpy as np
import torch

from aloe.data import CLEVRERQuestions, collate_questions
from aloe.model import ALOE
from aloe.rollout import extend_slots


def test_aloe_h5_question_pipeline(tmp_path):
    slots_path = tmp_path / "slots.h5"
    slots = np.random.default_rng(0).normal(size=(160, 2, 4)).astype(np.float32)
    with h5py.File(slots_path, "w") as file:
        file.create_dataset("slots", data=slots)
        file.create_dataset("episode_idx", data=np.full(160, 10_000))
        file.create_dataset("step_idx", data=np.arange(160))
        file.create_dataset("ep_len", data=np.asarray([160], np.int32))
        file.create_dataset("ep_offset", data=np.asarray([0], np.int64))

    questions = [
        {
            "scene_index": 10_000,
            "video_filename": "video_10000.mp4",
            "questions": [
                {
                    "question_id": 0,
                    "question": "Are there any objects?",
                    "question_type": "descriptive",
                    "answer": "yes",
                },
                {
                    "question_id": 1,
                    "question": "What will happen next?",
                    "question_type": "predictive",
                    "choices": [
                        {
                            "choice_id": 0,
                            "choice": "the objects collide",
                            "answer": "correct",
                        },
                        {
                            "choice_id": 1,
                            "choice": "the objects exit",
                            "answer": "wrong",
                        },
                    ],
                },
            ],
        }
    ]
    questions_root = tmp_path / "questions"
    questions_root.mkdir()
    with (questions_root / "val.json").open("w") as file:
        json.dump(questions, file)

    dataset = CLEVRERQuestions(
        slots_path,
        questions_root,
        split="val",
        sample_frames=3,
        observed_frames=16,
        max_question_len=5,
        max_choice_len=4,
    )
    batch = collate_questions([dataset[0], dataset[1]])
    assert batch["cls_video_emb"].shape == (1, 3, 2, 4)
    assert batch["mc_video_emb"].shape == (1, 3, 2, 4)

    model = ALOE(
        num_slots=2,
        slot_dim=4,
        sample_frames=3,
        max_question_len=5,
        max_choice_len=4,
        input_dim=4,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        mlp_dim=8,
    )
    output = model(batch)
    assert output["cls_answer_logits"].shape == (1, 22)
    assert output["mc_answer_logits"].shape == (2,)
    assert torch.isfinite(model.loss(batch, output)["loss"])


def test_cjepa_slot_rollout_interleaves_frame_offsets():
    class Increment(torch.nn.Module):
        history_frames = 2
        pred_frames = 1

        def predict(self, history, inference=False):
            assert inference
            return history[:, -1:] + 1

    slots = torch.arange(8, dtype=torch.float32).view(1, 8, 1, 1)
    result = extend_slots(Increment(), slots, target_frames=12, frameskip=2)
    assert result.shape == (1, 12, 1, 1)
    assert result[:, 8:, 0, 0].tolist() == [[7.0, 8.0, 8.0, 9.0]]
