from datasets import load_dataset, Video
import json
import base64
from torchcodec.decoders import VideoDecoder
import numpy as np
import torch
import cv2

dataset = load_dataset("CausalVerse/CausalVerse_Video_Robotics_Kitchen", cache_dir="/cs/data/people/hnam16/causalverse/kitchen/data")
print(dataset)
ds = dataset["robotics_kitchen"]
a = ds['metavalue'][0]

meta = json.loads(ds['metavalue'][0])
video_bytes = base64.b64decode(ds["videos"][0][0])
decoder = VideoDecoder(video_bytes, dimension_order='NHWC')
frames = [decoder[i]for i in range(len(decoder))]
video = torch.stack(frames, dim=0).float()
with open("dataset/causalverse/kitchen_metavalues_sample.json", "w") as f:
    json.dump(meta, f, indent=4)

len_npz = len(ds['npz_data'])
print("len npz data:", len_npz)
npz = json.loads(ds['npz_data'][0])



saveframes = video.cpu().numpy().astype(np.uint8)

hh, ww = saveframes.shape[1:3]
out = cv2.VideoWriter(
    "sample.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    10,
    (ww, hh)
)

for f in saveframes:
    out.write(f[..., ::-1])  # RGB → BGR

out.release()