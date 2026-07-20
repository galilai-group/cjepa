# C-JEPA

Official implementation of **Causal-JEPA: Learning World Models through
Object-Level Latent Interventions**.

[Heejeong Nam](https://hazel-heejeong-nam.github.io/),
[Quentin Le Lidec](https://quentinll.github.io/),
[Lucas Maes](https://lucasmaes.bearblog.dev/),
[Yann LeCun](https://yann.lecun.com/), and
[Randall Balestriero](https://randallbalestriero.github.io/).

[Paper](https://arxiv.org/abs/2602.11389) ·
[Project page](https://hazel-heejeong-nam.github.io/cjepa/) ·
[Checkpoints](https://huggingface.co/HazelNam/CJEPA)

<!-- ![C-JEPA architecture](static/architecture.png) -->

C-JEPA masks complete object trajectories in latent space, then predicts the
masked history and future objects from the remaining context. This repository
contains that contribution, thin data/training/planning entry points, and the
downstream CLEVRER ALOE evaluation. Environment management, HDF5 loading,
training infrastructure, environments, and MPC come from released
`stable-worldmodel` and `stable-pretraining` packages.

## Setup

Run:

```bash
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates the locked `.venv` and installs inference-only VideoSAUR
inside it; no third-party training tree is added to the visible repository.

Datasets, downloaded encoder checkpoints, and model checkpoints live below
`$STABLEWM_HOME` (default: `~/.stable_worldmodel`):

```bash
export STABLEWM_HOME=/path/with/enough/storage  # optional
```

## Data

All data uses the current stable-worldmodel HDF5 schema. Dataset names resolve
under `$STABLEWM_HOME/datasets`.

### Push-T
* Follow [https://github.com/lucas-maes/le-wm#data](https://github.com/lucas-maes/le-wm#data) to download the Push-T dataset.

### CLEVRER: download and convert

This single command downloads all official CLEVRER video archives and writes
the train/validation/test H5 files:

```bash
python scripts/prepare_clevrer.py
```
Note that the CLEVRER dataset is large (≈ 20 GB) and the conversion takes time.

## Object Encoder Checkpoints

Object-centric encoder *training* is intentionally not part of this repo.
When raw-pixel training or slot extraction first runs, the matching checkpoint
is downloaded automatically to
`$STABLEWM_HOME/artifacts/object-encoders/`:

We currently support VideoSAUR checkpoints only.

| Dataset | Encoder checkpoint |
|---|---|
| CLEVRER | [VideoSAUR](https://huggingface.co/HazelNam/CJEPA/blob/main/clevrer_videosaur_model.ckpt) |
| PushT | [VideoSAUR](https://huggingface.co/HazelNam/CJEPA/blob/main/pusht_videosaur_model.ckpt) |

Override the path when needed:

```bash
python train.py data.encoder.checkpoint=/path/to/model.ckpt
```


## Speed-Up Training

Pre-extract slots once to speed up training. This uses native
stable-worldmodel H5 slot representations:

```bash
python scripts/extract_slots.py \
  "$STABLEWM_HOME/datasets/clevrer_train.h5" \
  "$STABLEWM_HOME/datasets/clevrer_train_slots.h5" \
  --dataset clevrer
```

```bash
python scripts/extract_slots.py \
  "$STABLEWM_HOME/datasets/pusht_expert_train.h5" \
  "$STABLEWM_HOME/datasets/pusht_expert_train_slots.h5" \
  --dataset pusht
```




## Training

CLEVRER :

```bash
python train.py data=clevrer
```

Use a pre-extracted-slot file with the same command and one override:

```bash
python train.py data.train=clevrer_train_slots.h5 \
  data.val=clevrer_val_slots.h5
```

PushT:

```bash
python train.py data=pusht
```
or for pre-extracted slots:

```bash
python train.py data.train=pusht_expert_train_slots.h5 
```

Checkpoints are written to
`$STABLEWM_HOME/checkpoints/cjepa/<dataset>/`. The `_object.ckpt` file is
directly loadable by `stable_worldmodel.policy.AutoCostModel`.

## Planning

Set `policy` to a checkpoint path relative to
`$STABLEWM_HOME/checkpoints`, without the `_object.ckpt` suffix:

```bash
python eval.py policy=cjepa/pusht
```

All planning settings are in `config/eval.yaml`; there are no values embedded
in `eval.py`. For a quick check:

```bash
python eval.py policy=cjepa/pusht eval.num_eval=1 eval.eval_budget=2 \
  eval.goal_offset=2 plan.horizon=1 plan.receding_horizon=1 \
  solver.num_samples=2 solver.topk=1 solver.n_steps=1 eval.video=false
```

## CLEVRER visual reasoning (ALOE)

The downstream ALOE pipeline lives entirely in `aloe/` and uses one config,
`aloe/config.yaml`. It reads the same H5 slot files as C-JEPA; the old
SlotFormer/NeRV code and pickle conversion are not needed. CLEVRER question
annotations are downloaded automatically on the first train/eval run. Its
model and evaluation behavior were recovered from the previous repository
snapshot and the original [SlotFormer](https://github.com/pairlab/SlotFormer)
ALOE implementation.

First extract VideoSAUR slots for all three CLEVRER splits:

```bash
for split in train val test; do
  python scripts/extract_slots.py \
    "$STABLEWM_HOME/datasets/clevrer_${split}.h5" \
    "$STABLEWM_HOME/datasets/clevrer_${split}_slots.h5" \
    --dataset clevrer
done
```

Then run the complete C-JEPA → ALOE path:

```bash
python -m aloe.rollout
python -m aloe.train
python -m aloe.eval
```

By default, rollout loads
`$STABLEWM_HOME/checkpoints/cjepa/clevrer/cjepa_object.ckpt`, extends every
slot sequence from 128 to 160 frames, and writes
`clevrer_{split}_slots_rollout.h5`. ALOE writes `best.ckpt` below
`$STABLEWM_HOME/checkpoints/aloe/`, and evaluation writes validation
predictions plus per-question-type accuracy.

Use another C-JEPA checkpoint or create a CLEVRER test submission with one
override:

```bash
python -m aloe.rollout rollout.checkpoint=/path/to/cjepa_object.ckpt
python -m aloe.eval eval.split=test \
  eval.output="$STABLEWM_HOME/results/aloe_test.json"
```

## Repository layout

```text
cjepa.py                  C-JEPA architecture and stable-worldmodel adapter
encoder.py                checkpoint download and frozen VideoSAUR inference
train.py                  shared pixel/slot trainer
eval.py                   PushT planning
config/                   one train config, two data presets, one eval config
scripts/prepare_clevrer.py download-to-H5 CLEVRER converter
scripts/extract_slots.py  H5 pixels-to-slots converter
aloe/                     C-JEPA rollout, ALOE train/eval, and one config
tests/                    fast model and H5 integration tests
```

## Architecture compatibility

Restoring ALOE does not change the C-JEPA architecture. The ALOE dimensions,
question/choice tokens, slot tokens, positional encoding, and 12-layer
transformer setup match the previous pipeline; only the old NeRV transformer
container is replaced by PyTorch's equivalent `TransformerEncoder`. No legacy
ALOE checkpoint shim is included.

# Misc

The tested releases are:

- `stable-worldmodel==0.1.1`
- `stable-pretraining==0.1.7`
