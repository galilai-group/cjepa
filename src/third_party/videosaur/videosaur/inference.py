import argparse
import torch
from torchvision.io import read_video, read_image
from omegaconf import OmegaConf
from src.third_party.videosaur.videosaur import configuration, models
from src.third_party.videosaur.videosaur.data.transforms import CropResize, Normalize, Resize, build_inference_transform
import os
import numpy as np
import imageio
from torchvision import transforms as tvt
from src.third_party.videosaur.videosaur.visualizations import mix_inputs_with_masks, draw_segmentation_masks_on_image, color_map, overlay_inputs_with_masks
import matplotlib.pyplot as plt


def load_model_from_checkpoint(checkpoint_path: str, config_path: str):
    config = configuration.load_config(config_path)
    model = models.build(config.model, config.optimizer)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, config

def prepare_video(video_path: str, transfom_config=None):
    # Load video
    video, _, _ = read_video(video_path)
    video = video.float() / 255.0
    #change size of the video to 224x224
    video_vis = video.permute(0, 3, 1, 2)
    video_vis = tvt.Resize((transfom_config.input_size, transfom_config.input_size))(video_vis)
    video_vis = video_vis.permute(1, 0, 2, 3)

    
    if transfom_config:
        tfs = build_inference_transform(transfom_config)
        video = video.permute(3, 0, 1, 2)
        video = tfs(video).permute(1, 0, 2, 3)
     # Add batch dimension
    inputs = {"video": video.unsqueeze(0), 
              "video_visualization": video_vis.unsqueeze(0)}
    return inputs


def prepare_image(image_path: str, transfom_config=None):
    image = read_image(image_path)
    image = image.float() / 255.0
    resize = CropResize(dataset_type="image", 
                        crop_type="short_side_resize_central", 
                        size=transfom_config.input_size, 
                        resize_mode="bilinear")
    image_vis =resize(image)
    
    if transfom_config:
        tfs = build_inference_transform(transfom_config)
        image = tfs(image)
     # Add batch dimension
    inputs = {"image": image.unsqueeze(0), 
              "image_visualization": image_vis.unsqueeze(0)}
    return inputs



def main(config):
    # Load the model from checkpoint
    
    model, _ = load_model_from_checkpoint(config.checkpoint, config.model_config)
    model.initializer.n_slots = config.n_slots
    # Prepare the video dict
    if config.input.type == "video":
        prepare_inputs = prepare_video
    elif config.input.type == "image":
        prepare_inputs = prepare_image
        
    inputs = prepare_inputs(config.input.path, config.input.transforms)
    # Perform inference
    with torch.no_grad():
        outputs = model(inputs) # inputs.video = torch.Size([1, 128, 3, 196, 196]) inputs.video_vis = torch.Size([1, 3, 128, 196, 196])
        aux_outputs = model.aux_forward(inputs, outputs)
    if config.input.type=="video" and config.output.save_path:
        # Save the results
        save_dir = os.path.dirname(config.output.save_path)
        os.makedirs(save_dir, exist_ok=True)
        # Original grid video (input + per-slot masks)
        masked_video_frames = mix_inputs_with_masks(inputs, outputs)
        with imageio.get_writer(config.output.save_path, fps=30) as writer:
            for frame in masked_video_frames:
                writer.append_data(frame)
        writer.close()

        # Single overlay video with colored slot attentions
        overlay_frames = overlay_inputs_with_masks(inputs, outputs)
        base, ext = os.path.splitext(config.output.save_path)
        overlay_save_path = getattr(config.output, "overlay_save_path", None)
        if overlay_save_path is None:
            overlay_save_path = f"{base}_overlay{ext or '.mp4'}"
        os.makedirs(os.path.dirname(overlay_save_path), exist_ok=True)
        with imageio.get_writer(overlay_save_path, fps=30) as overlay_writer:
            for frame in overlay_frames:
                overlay_writer.append_data(frame)
        overlay_writer.close()
    elif config.input.type=="image" and config.output.save_path:
        save_dir = os.path.dirname(config.output.save_path)
        os.makedirs(save_dir, exist_ok=True)
        masks = aux_outputs["decoder_masks_hard"][0].bool()
        cmap = color_map(masks.shape[0])
        image = (inputs["image_visualization"]*256)[0].type(torch.uint8)
        mixed_image = draw_segmentation_masks_on_image(image, masks, colors=cmap)
        # Save the results
        plt.imsave(config.output.save_path, mixed_image.permute(1, 2, 0).numpy())
    print("Inference completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on a single MP4 video.")
    parser.add_argument("--config", default="configs/inference/movi_c.yml", help="Configuration to run")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)
