
from pathlib import Path
import hydra
import torch
import torchvision
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import numpy as np

import stable_pretraining as spt
import stable_worldmodel as swm
from custom_models.dinowm_reg import DINOWM_REG
from transformers import AutoModel



from utils.eval_metrics import rankme, frechet_joint_distance, feature_rollout_degradation


from utils.eval import EvalFramework
from utils.global_visualization import visualize

DINO_PATCH_SIZE = 14


def load_model_from_checkpoint(cfg):

    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path must be specified in config!")
    
    ckpt_path = Path(cfg.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    logging.info(f"Loading checkpoint from: {ckpt_path}")
    
    # Load DINO encoder using videosaur's model builder
    encoder = AutoModel.from_pretrained("facebook/dinov2-with-registers-small") # yes cls, .last
    embedding_dim = encoder.config.hidden_size
    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    
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

    
    world_model = DINOWM_REG(
        encoder=spt.backbone.EvalOnly(encoder),
        predictor=predictor,
        action_encoder=None,
        proprio_encoder=None,
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



def evaluate_videowm(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    world_model = load_model_from_checkpoint(cfg)
    world_model = world_model.to(device)
    
    cache_dir = cfg.get("cache_dir", None)
    img_size = (cfg.image_size // cfg.patch_size) * DINO_PATCH_SIZE
    evaluation = EvalFramework(
        world_model, 
        cfg.dataset_name, 
        cache_dir, 
        img_size,
        metrics=cfg.metrics,
        device=device
    )

    eval_dataset = evaluation.pull_eval_data(
        cfg.n_steps,
        cfg.frameskip,
        seed=cfg.seed,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    
    num_batches = cfg.get('num_batches', None)
    if cfg.metrics.rankme.enabled:
        min_num_batches = cfg.metrics.rankme.min_batch_size // cfg.batch_size + 1
        if num_batches is None or num_batches < min_num_batches:
            logging.info(f"RankMe needs at least {min_num_batches} batches : [CFG] {num_batches} >> [Updated] {min_num_batches}")
            num_batches = min_num_batches
    
    pred_embeddings_all, target_embeddings_all = evaluation.get_eval_embeddings(
        eval_loader,
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        num_batches=num_batches
    )

    
    logging.info(f"Total accumulated embeddings: {pred_embeddings_all.shape}")

    results = evaluation.calculate_metrics( 
        pred_embeddings_all, 
        target_embeddings_all,
        rollout_loader=eval_loader if cfg.metrics.feature_rollout_degradation.enabled else None,
        rollout_batches=cfg.get('rollout_batches', None),
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds
    )
    
    # Save results
    output_dir = Path(cfg.get('output_dir', './eval_results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "eval_results.txt"
    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("VideoWM Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint: {cfg.checkpoint_path}\n")
        f.write(f"Dataset: {cfg.dataset_name} ({cfg.get('split', 'validation')})\n")
        f.write(f"Total samples: {pred_embeddings_all.shape[0]}\n")
        f.write(f"Embedding dimension: {pred_embeddings_all.shape[1]}\n\n")
        
        for metric_name, metric_value in results.items():
            if metric_name == 'rollout_degradation':
                f.write(f"\nFeature Rollout Degradation:\n")
                f.write(f"  Degradation Rate: {metric_value['degradation_rate']:.4f}\n")
                f.write(f"  MSE per step: {metric_value['mse_per_step']}\n")
                f.write(f"  Cosine Similarity per step: {metric_value['cosine_sim_per_step']}\n")
            else:
                f.write(f"{metric_name}: {metric_value:.4f}\n")
    
    logging.info(f"\nResults saved to: {results_file}")
    
    # Optional: save embeddings for further analysis
    if cfg.get('save_embeddings', False):
        embeddings_file = output_dir / "embeddings.pt"
        torch.save({
            'pred_embeddings': pred_embeddings_all,
            'target_embeddings': target_embeddings_all,
        }, embeddings_file)
        logging.info(f"Embeddings saved to: {embeddings_file}")
    

    if cfg.get('visualization', {}).get('enabled', False):
        visualize(pred_embeddings_all, cfg, output_dir)
    
    return results


@hydra.main(version_base=None, config_path="../configs", config_name="config_test")
def run(cfg):
    """Entry point for evaluation."""
    logging.info("VideoWM Evaluation")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    results = evaluate_videowm(cfg)
    
    logging.info("\n" + "=" * 60)
    logging.info("Evaluation Complete")
    logging.info("=" * 60)


if __name__ == "__main__":
    run()
