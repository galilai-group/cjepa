import os
import pickle as pkl
import torch
from train.train_causalwm_from_clevrer_slot import setup_distributed, is_main_process, setup_wandb, get_data, get_world_model, rollout_video_slots
import hydra
import logging
from custom_models.cjepa_predictor import MaskedSlotPredictor
from aloe_scripts.visualize import build_model, build_method

data_dir = "/cs/data/people/hnam16/data/clevrer_for_savi/videos/train/video_08000-09000/video_08001/mp4"
cjepa_ckpt = "/cs/data/people/hnam16/.stable_worldmodel/121p_final_predictor.ckpt"
savi_model_ckpt = "/cs/data/people/hnam16/savi_pretrained/clevrer_savi_reproduce_20260109_102641_LR0.0001/savi/epoch/model_8.pth"
video_feature = "/cs/data/people/hnam16/data/modified_extraction/clevrer_savi_reproduced.pkl"
collision_frames = [16,40]

with open(video_feature, "rb") as f:
    video_feature = pkl.load(f)

target_video = video_feature["video_08001"]

@hydra.main(version_base=None, config_path="../configs", config_name="config_train_causal_clevrer_slot")
def run(cfg):

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    predictor = MaskedSlotPredictor(
        num_slots=7,  
        slot_dim=128,  
        history_frames=6,  
        pred_frames=10,  
        num_masked_slots=1,  # to test attention
        seed=cfg.seed,  
        depth=cfg.predictor.get("depth", 6),
        heads=cfg.predictor.get("heads", 16),
        dim_head=cfg.predictor.get("dim_head", 64),
        mlp_dim=cfg.predictor.get("mlp_dim", 2048),
        dropout=cfg.predictor.get("dropout", 0.1),
    )

    predictor = predictor.to(device)
    state_dict = torch.load(cjepa_ckpt, map_location=device)
    predictor.load_state_dict(state_dict)


    data = []
    for d in data:
        out, masked_indices, attention_dict = predictor.attention_probing(d.to(device), layer_idx=-1)

    savi_model = build_model(params)
    state_dict = torch.load(savi_model_ckpt, map_location="cpu")
    missing, unexpected = savi_model.load_state_dict(state_dict['state_dict'], strict=False)
    assert len(missing) == 0, f"Missing keys when loading pretrained weights: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys when loading pretrained weights: {unexpected}"

    info = 'savi'
    ckp_path = os.path.join('temp', info)       
    method = build_method(
        model=savi_model,
        datamodule=datamodule,
        params=params,
        ckp_path=ckp_path,
        local_rank=0,
        use_ddp=False,
        use_fp16=True,
    )

    method.model.eval()
    for out_slice in out:
        # visualize predictions
        pred_post_recon_img, pred_post_recons, pred_post_masks, _ = method.model.decode(out_slice.flatten(0, 1))
        original_saveframe = video[timestamp]  # 3,64,64
        combined_saveframe = recon_combined[timestamp] # 3,64, 64
        recon_saveframe = [recons[timestamp][i] for i in range(recons.shape[1])] # list of 3,64,64
        masks_saveframe = [masks[timestamp][i] for i in range(masks.shape[1])] # list of 3,64,64
        
        original_saveframe = video[timestamp]  # 3,64,64
        combined_saveframe = recon_combined[timestamp] # 3,64, 64
        recon_saveframe = [recons[timestamp][i] for i in range(recons.shape[1])] # list of 3,64,64
        masks_saveframe = [masks[timestamp][i] for i in range(masks.shape[1])] # list of 3,64,64
        #save torch tensor as image
        save_dir = os.path.join('savi_visualize', args.ckpt.split('/')[-1][:-4], f'video_{i.item()}')
        mkdir_or_exist(save_dir)
        save_image(original_saveframe, os.path.join(save_dir, f'original_frame_{timestamp}.png'))
        save_image(combined_saveframe, os.path.join(save_dir, f'combined_frame_{timestamp}.png'))
        for idx, m in enumerate(recon_saveframe):
            save_image(m, os.path.join(save_dir, f'recon_{idx}_frame_{timestamp}.png'))
        for idx, m in enumerate(masks_saveframe):
            save_image(m, os.path.join(save_dir, f'mask_{idx}_frame_{timestamp}.png'))

        imgs = video.type_as(recon_combined)
        save_video = method._make_video_grid(imgs, recon_combined, recons, masks)
        # results.append(save_video)

        video = method._convert_video([save_video], caption=None)




