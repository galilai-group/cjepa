from datasets import load_from_disk
import pickle as pkl
import os
import torch
import numpy as np
from tqdm import trange, tqdm

slotpath="/your/own/path/pusht_expert_slots_pushtnoise_videosaur_lr1e-4_w03_step=100000.pkl"
save_dir = "/your/own/path"
train_meta = load_from_disk("~/.stable_worldmodel/pusht_expert_train")
val_meta = load_from_disk("~/.stable_worldmodel/pusht_expert_val")

with open(slotpath, "rb") as f:
    data = pkl.load(f) 

train_vid_num = len(data['train'])
val_vid_num = len(data['val'])
train_action_dict = {}
val_action_dict = {}
train_proprio_dict = {}
val_proprio_dict = {}
train_state_dict = {}
val_state_dict = {}

action_dim = val_meta[0]['action'].shape[0]
proprio_dim = val_meta[0]['proprio'].shape[0]
state_dim = val_meta[0]['state'].shape[0]

for i in trange(train_vid_num):
    key = f"{i}_pixels.mp4"
    num_frames = data["train"][key].shape[0]
    train_action_dict[key] = np.zeros((num_frames, action_dim), dtype=np.float32)
    train_proprio_dict[key] = np.zeros((num_frames, proprio_dim), dtype=np.float32)
    train_state_dict[key] = np.zeros((num_frames, state_dim), dtype=np.float32)
for i in trange(val_vid_num):
    key = f"{i}_pixels.mp4"
    num_frames = data["val"][key].shape[0]
    val_action_dict[key] = np.zeros((num_frames, action_dim), dtype=np.float32)
    val_proprio_dict[key] = np.zeros((num_frames, proprio_dim), dtype=np.float32)
    val_state_dict[key] = np.zeros((num_frames, state_dim), dtype=np.float32)

for rows in tqdm(train_meta):
    episode = int(rows['episode_idx'])
    frame = int(rows['step_idx'])
    key = f"{episode}_pixels.mp4"
    train_action_dict[key][frame] = rows['action']
    train_proprio_dict[key][frame] = rows['proprio']
    train_state_dict[key][frame] = rows['state']

for rows in val_meta:
    episode = int(rows['episode_idx'])
    frame = int(rows['step_idx'])
    key = f"{episode}_pixels.mp4"
    val_action_dict[key][frame] = rows['action']
    val_proprio_dict[key][frame] = rows['proprio']
    val_state_dict[key][frame] = rows['state']


action_file = {
    "train": train_action_dict,
    "val": val_action_dict
}

proprio_file = {
    "train": train_proprio_dict,
    "val": val_proprio_dict
}

state_file = {
    "train": train_state_dict,
    "val": val_state_dict
}

with open(os.path.join(save_dir, "pusht_expert_action_meta.pkl"), "wb") as f:
    pkl.dump(action_file, f)
with open(os.path.join(save_dir, "pusht_expert_proprio_meta.pkl"), "wb") as f:
    pkl.dump(proprio_file, f)
with open(os.path.join(save_dir, "pusht_expert_state_meta.pkl"), "wb") as f:
    pkl.dump(state_file, f) 