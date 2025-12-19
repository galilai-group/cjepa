PYTHONPATH=$(pwd)/videosaur

python videosaur/inference.py \
    checkpoint="/cs/data/people/hnam16/.stable_worldmodel/artifacts/oc_ckpts/oc_ckpt.ckpt" \
    input.path="/cs/data/people/hnam16/data/clevrer/videos/video_08000.mp4" \
    output.save_path="result_videosaur/video_08000_masks_hf.mp4"