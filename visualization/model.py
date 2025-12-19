from pathlib import Path
import torch
from loguru import logger as logging
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

import stable_pretraining as spt
import stable_worldmodel as swm
from custom_models.dinowm_oc import OCWM
from videosaur.videosaur import  models
from custom_models.dinowm_causal import CausalWM
from custom_models.cjepa_predictor import MaskedSlotPredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_OC_model_from_checkpoint(cfg):

    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path must be specified in config!")
    ckpt_path = Path(cfg.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    logging.info(f"Loading checkpoint from: {ckpt_path}")

    model = models.build(cfg.model, cfg.dummy_optimizer, None, None)
    encoder = model.encoder 
    slot_attention = model.processor 
    initializer = model.initializer
    embedding_dim = cfg.videosaur.SLOT_DIM 
    num_patches = cfg.videosaur.NUM_SLOTS

    if cfg.training_type == "wm":
        embedding_dim += cfg.dinowm.proprio_embed_dim + cfg.dinowm.action_embed_dim  # Total embedding size
    logging.info(f"Patches: {num_patches}, Embedding dim: {embedding_dim}")
    
    # For VideoWM, no action/proprio encoders
    predictor = swm.wm.dinowm.CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg.dinowm.history_size,
        dim=embedding_dim,
        **cfg.predictor,
    )

    # Build action and proprioception encoders
    if cfg.training_type == "video":    
        action_encoder = None
        proprio_encoder = None
        logging.info(f"[Video Only] Action encoder: None, Proprio encoder: None")
    else :
        effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
        action_encoder = swm.wm.dinowm.Embedder(in_chans=effective_act_dim, emb_dim=cfg.dinowm.action_embed_dim)
        proprio_encoder = swm.wm.dinowm.Embedder(in_chans=cfg.dinowm.proprio_dim, emb_dim=cfg.dinowm.proprio_embed_dim)
        logging.info(f"Action dim: {effective_act_dim}, Proprio dim: {cfg.dinowm.proprio_dim}")
    
    world_model = OCWM(
        encoder=spt.backbone.EvalOnly(encoder),
        slot_attention=spt.backbone.EvalOnly(slot_attention),
        initializer = spt.backbone.EvalOnly(initializer),
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        history_size=cfg.dinowm.history_size,
        num_pred=cfg.dinowm.num_preds,
    )
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    ckpt_state = checkpoint.model.state_dict()
    
    # Load into world_model
    missing, unexpected = world_model.load_state_dict(ckpt_state, strict=False)
    logging.info(f"Missing keys: {len(missing) if missing else 0}")
    logging.info(f"Unexpected keys: {len(unexpected) if unexpected else 0}")
    
    # Set to eval model
    world_model.eval()
    
    return world_model


def load_causal_model_from_checkpoint(cfg):

    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path must be specified in config!")
    ckpt_path = Path(cfg.checkpoint_path)
    print(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    logging.info(f"Loading checkpoint from: {ckpt_path}")

    model = models.build(cfg.model, cfg.dummy_optimizer, None, None)
    encoder = model.encoder 
    slot_attention = model.processor 
    initializer = model.initializer
    embedding_dim = cfg.videosaur.SLOT_DIM 
    num_patches = cfg.videosaur.NUM_SLOTS

    if cfg.training_type == "wm":
        embedding_dim += cfg.dinowm.proprio_embed_dim + cfg.dinowm.action_embed_dim  # Total embedding size
    logging.info(f"Patches: {num_patches}, Embedding dim: {embedding_dim}")
    
    # For VideoWM, no action/proprio encoders
    predictor = MaskedSlotPredictor(
        num_slots=num_patches,  # S: number of slots
        slot_dim=embedding_dim,  # 64 or higher if action/proprio included
        history_frames=cfg.dinowm.history_size,  # T: history length
        pred_frames=cfg.dinowm.num_preds,  # number of future frames to predict
        num_masked_slots=cfg.get("num_masked_slots", 2),  # M: number of slots to mask
        seed=cfg.seed,  # for reproducible masking
        depth=cfg.predictor.get("depth", 6),
        heads=cfg.predictor.get("heads", 16),
        dim_head=cfg.predictor.get("dim_head", 64),
        mlp_dim=cfg.predictor.get("mlp_dim", 2048),
        dropout=cfg.predictor.get("dropout", 0.1),
    )

    # Build action and proprioception encoders
    if cfg.training_type == "video":    
        action_encoder = None
        proprio_encoder = None
        logging.info(f"[Video Only] Action encoder: None, Proprio encoder: None")
    else :
        effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
        action_encoder = swm.wm.dinowm.Embedder(in_chans=effective_act_dim, emb_dim=cfg.dinowm.action_embed_dim)
        proprio_encoder = swm.wm.dinowm.Embedder(in_chans=cfg.dinowm.proprio_dim, emb_dim=cfg.dinowm.proprio_embed_dim)
        logging.info(f"Action dim: {effective_act_dim}, Proprio dim: {cfg.dinowm.proprio_dim}")
    
    world_model = CausalWM(
        encoder=spt.backbone.EvalOnly(encoder),
        slot_attention=spt.backbone.EvalOnly(slot_attention),
        initializer = spt.backbone.EvalOnly(initializer),
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        history_size=cfg.dinowm.history_size,
        num_pred=cfg.dinowm.num_preds,
    )
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    ckpt_state = checkpoint.model.state_dict()
    
    # Load into world_model
    missing, unexpected = world_model.load_state_dict(ckpt_state, strict=False)
    logging.info(f"Missing keys: {len(missing) if missing else 0}")
    logging.info(f"Unexpected keys: {len(unexpected) if unexpected else 0}")
    
    # Set to eval model
    world_model.eval()
    
    return world_model