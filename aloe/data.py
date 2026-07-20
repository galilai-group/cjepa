"""CLEVRER question and HDF5-slot input pipeline for ALOE."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers HDF5 compression filters
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

QUESTION_REVISION = "412337d0210bf98cee2ca90c3586ab2ea7ca519e"
QUESTION_URL = (
    "https://raw.githubusercontent.com/galilai-group/cjepa/"
    f"{QUESTION_REVISION}/dataset/clevrer/questions/{{split}}.json"
)
SPLIT_OFFSET = {"train": 0, "val": 10_000, "test": 15_000}
SUBTYPE = {
    "descriptive": 0,
    "explanatory": 1,
    "predictive": 2,
    "counterfactual": 3,
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url)
    mode = "wb"
    offset = 0
    if temporary.exists():
        offset = temporary.stat().st_size
        request.add_header("Range", f"bytes={offset}-")
        mode = "ab"
    with urllib.request.urlopen(request) as response:
        if offset and response.status != 206:
            offset, mode = 0, "wb"
        total = response.headers.get("Content-Length")
        total = offset + int(total) if total else None
        with temporary.open(mode) as file, tqdm(
            total=total,
            initial=offset,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            while chunk := response.read(8 * 1024 * 1024):
                file.write(chunk)
                progress.update(len(chunk))
    temporary.replace(destination)


def ensure_questions(root: str | Path, split: str) -> Path:
    """Download the original repository's CLEVRER annotations on first use."""
    path = Path(root).expanduser().resolve() / f"{split}.json"
    if not path.exists():
        print(f"Downloading CLEVRER {split} questions to {path}")
        _download(QUESTION_URL.format(split=split), path)
    return path


def load_vocab() -> dict:
    with (Path(__file__).parent / "vocab.json").open() as file:
        return json.load(file)


class CLEVRERQuestions(Dataset):
    """Join CLEVRER questions to episode slots in a stable-worldmodel H5 file."""

    def __init__(
        self,
        slots_path: str | Path,
        questions_root: str | Path,
        *,
        split: str,
        sample_frames: int = 25,
        observed_frames: int = 128,
        max_question_len: int = 20,
        max_choice_len: int = 12,
        shuffle_slots: bool = False,
    ):
        if split not in SPLIT_OFFSET:
            raise ValueError(f"Unknown CLEVRER split: {split}")
        self.slots_path = Path(slots_path).expanduser().resolve()
        if not self.slots_path.exists():
            raise FileNotFoundError(f"Slot dataset not found: {self.slots_path}")
        self.split = split
        self.sample_frames = sample_frames
        self.observed_frames = observed_frames
        self.max_question_len = max_question_len
        self.max_choice_len = max_choice_len
        self.shuffle_slots = shuffle_slots
        self.frame_offset = observed_frames // sample_frames
        self._file = None

        with h5py.File(self.slots_path, "r") as file:
            if "slots" not in file:
                raise KeyError(f"{self.slots_path} does not contain a 'slots' column")
            self.lengths = np.asarray(file["ep_len"][:], dtype=np.int64)
            self.offsets = np.asarray(file["ep_offset"][:], dtype=np.int64)
            if "episode_idx" in file:
                ids = np.asarray(file["episode_idx"][self.offsets], dtype=np.int64)
            else:
                ids = np.arange(len(self.lengths), dtype=np.int64)
            self.slot_shape = tuple(file["slots"].shape[1:])

        # Canonical scene ids are preferred. The ordinal aliases also make old
        # subset H5 files (whose val ids started at zero) readable.
        self.scene_to_episode = {int(scene): i for i, scene in enumerate(ids)}
        base = SPLIT_OFFSET[split]
        self.scene_to_episode.update(
            {
                base + i: i
                for i in range(len(self.lengths))
                if base + i not in self.scene_to_episode
            }
        )

        vocab = load_vocab()
        self.q_vocab = vocab["q_vocab"]
        self.answer_to_label = vocab["a_vocab"]
        self.label_to_answer = {
            value: key for key, value in self.answer_to_label.items()
        }
        annotation = ensure_questions(questions_root, split)
        with annotation.open() as file:
            scenes = json.load(file)
        self.questions = self._prepare_questions(scenes)
        if not self.questions:
            raise ValueError(
                f"No {split} questions match episodes in {self.slots_path}. "
                "Check the episode_idx values."
            )

    def _tokenize(self, text: str, length: int):
        words = text.lower().replace("?", "").split(" ")
        try:
            tokens = [self.q_vocab[word] for word in words if word]
        except KeyError as error:
            raise KeyError(
                f"Unknown CLEVRER question token {error.args[0]!r}"
            ) from error
        if len(tokens) > length:
            raise ValueError(
                f"Question has {len(tokens)} tokens; configured maximum is {length}"
            )
        padding = np.ones(length, dtype=np.bool_)
        padding[: len(tokens)] = False
        tokens.extend([self.q_vocab["PAD"]] * (length - len(tokens)))
        return np.asarray(tokens, np.int64), padding

    def _prepare_questions(self, scenes: list[dict]) -> list[dict]:
        output = []
        for scene in scenes:
            scene_id = int(scene["scene_index"])
            if scene_id not in self.scene_to_episode:
                continue
            for question in scene["questions"]:
                question_type = question["question_type"]
                item = {
                    "scene_index": scene_id,
                    "question_id": int(question["question_id"]),
                    "subtype": SUBTYPE[question_type],
                }
                if question_type == "descriptive":
                    tokens, padding = self._tokenize(
                        question["question"],
                        self.max_question_len + self.max_choice_len,
                    )
                    item.update(
                        type=0,
                        tokens=tokens,
                        padding=padding,
                        label=self.answer_to_label.get(question.get("answer", ""), -1),
                    )
                else:
                    q_tokens, q_padding = self._tokenize(
                        question["question"], self.max_question_len
                    )
                    tokens, padding, labels, choice_ids = [], [], [], []
                    for choice in question["choices"]:
                        c_tokens, c_padding = self._tokenize(
                            choice["choice"], self.max_choice_len
                        )
                        tokens.append(np.concatenate([q_tokens, c_tokens]))
                        padding.append(np.concatenate([q_padding, c_padding]))
                        answer = choice.get("answer")
                        labels.append(
                            -1 if answer is None else int(answer == "correct")
                        )
                        choice_ids.append(int(choice["choice_id"]))
                    item.update(
                        type=1,
                        tokens=np.stack(tokens),
                        padding=np.stack(padding),
                        labels=np.asarray(labels, np.int64),
                        choice_ids=np.asarray(choice_ids, np.int64),
                    )
                output.append(item)
        return output

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.slots_path, "r")
        return self._file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self):
        file = getattr(self, "_file", None)
        if file is not None:
            file.close()

    def __len__(self):
        return len(self.questions)

    def _slots(self, item: dict) -> torch.Tensor:
        episode = self.scene_to_episode[item["scene_index"]]
        length, offset = int(self.lengths[episode]), int(self.offsets[episode])
        max_start = self.observed_frames - (self.sample_frames - 1) * self.frame_offset
        if length < self.observed_frames or max_start <= 0:
            raise ValueError(
                f"Episode {item['scene_index']} has {length} frames; "
                f"ALOE requires at least {self.observed_frames}"
            )
        start = np.random.randint(max_start) if self.split == "train" else 0
        if item["subtype"] == SUBTYPE["predictive"] and length > 150:
            start += length - self.observed_frames
        indices = offset + start + np.arange(self.sample_frames) * self.frame_offset
        slots = torch.from_numpy(np.asarray(self.file["slots"][indices], np.float32))
        if self.shuffle_slots:
            slots = slots[:, torch.randperm(slots.shape[1])]
        return slots

    def __getitem__(self, index):
        item = self.questions[index]
        output = {
            "scene_index": item["scene_index"],
            "question_id": item["question_id"],
            "q_subtype": item["subtype"],
            "q_type": item["type"],
            "q_tokens": torch.from_numpy(item["tokens"]),
            "q_pad_mask": torch.from_numpy(item["padding"]),
            "video_emb": self._slots(item),
        }
        if item["type"] == 0:
            output["a_label"] = item["label"]
        else:
            output["a_label"] = torch.from_numpy(item["labels"])
            output["mc_choice_id"] = torch.from_numpy(item["choice_ids"])
        return output


def collate_questions(items: list[dict]) -> dict[str, torch.Tensor]:
    cls = [item for item in items if item["q_type"] == 0]
    mc = [item for item in items if item["q_type"] == 1]
    example_slots = items[0]["video_emb"]
    token_len = items[0]["q_pad_mask"].shape[-1]

    def stack(group, key, shape, dtype):
        if group:
            values = [item[key] for item in group]
            return torch.stack(
                [
                    value if torch.is_tensor(value) else torch.tensor(value)
                    for value in values
                ]
            )
        return torch.empty(shape, dtype=dtype)

    if mc:
        mc_tokens = torch.cat([item["q_tokens"] for item in mc])
        mc_padding = torch.cat([item["q_pad_mask"] for item in mc])
        mc_labels = torch.cat([item["a_label"] for item in mc])
        mc_choice_ids = torch.cat([item["mc_choice_id"] for item in mc])
        mc_flag = torch.cat(
            [torch.full_like(item["a_label"], i) for i, item in enumerate(mc)]
        )
    else:
        mc_tokens = torch.empty((0, token_len), dtype=torch.long)
        mc_padding = torch.empty((0, token_len), dtype=torch.bool)
        mc_labels = torch.empty(0, dtype=torch.long)
        mc_choice_ids = torch.empty(0, dtype=torch.long)
        mc_flag = torch.empty(0, dtype=torch.long)

    return {
        "cls_video_emb": stack(
            cls, "video_emb", (0, *example_slots.shape), example_slots.dtype
        ),
        "cls_q_tokens": stack(cls, "q_tokens", (0, token_len), torch.long),
        "cls_q_pad_mask": stack(cls, "q_pad_mask", (0, token_len), torch.bool),
        "cls_label": stack(cls, "a_label", (0,), torch.long),
        "cls_scene_index": stack(cls, "scene_index", (0,), torch.long),
        "cls_question_id": stack(cls, "question_id", (0,), torch.long),
        "mc_video_emb": stack(
            mc, "video_emb", (0, *example_slots.shape), example_slots.dtype
        ),
        "mc_q_tokens": mc_tokens,
        "mc_q_pad_mask": mc_padding,
        "mc_label": mc_labels,
        "mc_flag": mc_flag,
        "mc_choice_id": mc_choice_ids,
        "mc_subtype": stack(mc, "q_subtype", (0,), torch.long),
        "mc_scene_index": stack(mc, "scene_index", (0,), torch.long),
        "mc_question_id": stack(mc, "question_id", (0,), torch.long),
    }


def make_dataset(cfg, split: str) -> CLEVRERQuestions:
    return CLEVRERQuestions(
        cfg.data[split],
        cfg.data.questions,
        split=split,
        sample_frames=cfg.data.sample_frames,
        observed_frames=cfg.data.observed_frames,
        max_question_len=cfg.data.max_question_len,
        max_choice_len=cfg.data.max_choice_len,
        shuffle_slots=cfg.data.shuffle_slots if split == "train" else False,
    )


def make_loader(dataset, cfg, *, train: bool):
    workers = int(cfg.loader.num_workers)
    kwargs = {
        "batch_size": cfg.loader.batch_size if train else cfg.loader.eval_batch_size,
        "shuffle": train,
        "num_workers": workers,
        "pin_memory": cfg.loader.pin_memory,
        "collate_fn": collate_questions,
        "drop_last": train,
    }
    if workers:
        kwargs["persistent_workers"] = cfg.loader.persistent_workers
    return torch.utils.data.DataLoader(dataset, **kwargs)
