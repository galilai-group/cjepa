import os 
import numpy as np
import pickle as pkl
import glob 
from tqdm import tqdm

train_base_dir = '/cs/data/people/hnam16/train_slots'
val_base_dir = '/cs/data/people/hnam16/val_slots'
save_dir = '/cs/data/people/hnam16/'

# all data
train_data_list = glob.glob(os.path.join(train_base_dir, '*.npy'))
train_data_list = sorted(train_data_list)


val_data_list = glob.glob(os.path.join(val_base_dir, '*.npy'))
val_data_list = sorted(val_data_list)

# load val data
val_data = {}
for data_path in tqdm(val_data_list):
    name = 'video_' + data_path.split('/')[-1].split('.')[0] + '.mp4'
    np_data = np.load(data_path)
    val_data[name] = np_data

train_data = {}
for data_path in tqdm(train_data_list):
    name = 'video_' + data_path.split('/')[-1].split('.')[0] + '.mp4'
    np_data = np.load(data_path)
    train_data[name] = np_data

data = {
    'train': train_data,
    'val': val_data
}
# save val data
with open(os.path.join(save_dir, 'phyre_savi_slots.pkl'), 'wb') as f:
    pkl.dump(data, f)