import os
import glob
import hickle as hkl
import numpy as np
from dataset.PYHRE.phyre_vistool import observations_to_float_rgb, observations_to_uint8_rgb
import subprocess

dirs = "/cs/data/people/hnam16/PHYRE/PHYRE_1fps_p100n400/full/*/*/*_image.hkl"
output_dir = "/cs/data/people/hnam16/PHYRE/rgb_videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
fps=25
bitrate="5M"

videos = glob.glob(dirs)

for video_path in videos:
    super, sub = video_path.split('/')[-3:-1]
    
    data = np.array(hkl.load(video_path),  dtype=np.int16)
    rgb_data = observations_to_uint8_rgb(data)
    assert rgb_data.ndim == 4 and rgb_data.shape[-1] == 3, f"frames shape must be (T,H,W,3), got {rgb_data.shape}"

    T, H, W, _ = rgb_data.shape
    new_name = f"{super}_{sub}_" + video_path.split('/')[-1].replace('_image.hkl', '.mp4')
    outpath = os.path.join(output_dir,super,sub, new_name)
    if not os.path.exists(os.path.dirname(outpath)):    
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
        proc.stdin.write(rgb_data.tobytes())
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with code {ret}")
    finally:
        if proc.poll() is None:
            proc.kill()

        