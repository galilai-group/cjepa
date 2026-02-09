from datasets import load_dataset, Video
import json
import base64
from torchcodec.decoders import VideoDecoder
import numpy as np
import torch
import cv2
import tempfile
import subprocess
import os
from tqdm import tqdm

fps=25
bitrate="5M"
basedir = "/cs/data/people/hnam16/causalverse/kitchen/videos_mp4/"
if not os.path.exists(basedir):
    os.makedirs(basedir)

dataset = load_dataset("CausalVerse/CausalVerse_Video_Robotics_Kitchen", cache_dir="/cs/data/people/hnam16/causalverse/kitchen/data")
print(dataset)
videos = dataset["robotics_kitchen"]["videos"]

for video_idx, video_data in enumerate(tqdm(videos)):
    for view_name, view_idx in zip(['agent_view', 'front_view', 'side_view'], [0,2,4]):
        video_bytes = base64.b64decode(video_data[view_idx])
        decoder = VideoDecoder(video_bytes, dimension_order='NHWC')
        frames = [decoder[i] for i in range(len(decoder))]
        video = torch.stack(frames, dim=0)
        
        saveframes = video.cpu().numpy().astype(np.uint8)
        hh, ww = saveframes.shape[1:3]

        assert saveframes.ndim == 4 and saveframes.shape[-1] == 3, f"frames shape must be (T,H,W,3), got {saveframes.shape}"
        T, H, W, _ = saveframes.shape
        outpath = os.path.join(basedir, view_name, f"video_{video_idx:05d}_{view_name}.mp4")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        # ffmpeg: raw rgb24 -> h264 mp4
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-b:v", bitrate,
            "-maxrate", bitrate,
            "-bufsize", "10M",
            outpath,
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            proc.stdin.write(saveframes.tobytes())
            proc.stdin.close()
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"ffmpeg failed with code {ret}")
        finally:
            if proc.poll() is None:
                proc.kill()

            