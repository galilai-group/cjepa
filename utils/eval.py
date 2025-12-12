from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger as logging
from torch.utils.data import DataLoader
import stable_pretraining as spt
import stable_worldmodel as swm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils.eval_metrics import rankme, frechet_joint_distance, feature_rollout_degradation


DINO_PATCH_SIZE=14


class EvalFramework():
    def __init__(
            self, 
            model,
            dataset_name, 
            cache_dir, 
            img_size,
            metrics: Dict[str, Any]
            ):
        super().__init__()

        self.model = model
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir 
        self.img_size = img_size 
        self.metrics = metrics

        if self.dataset_name == "clevrer_train":
            self.type="video"
        else:
            raise ValueError(f"Eval dataset not recognized: {self.dataset_name}")
        

    def _get_img_pipeline(key, target, img_size=224):
        return spt.data.transforms.Compose(
            spt.data.transforms.ToImage(
                **spt.data.dataset_stats.ImageNet,
                source=key,
                target=target,
            ),
            spt.data.transforms.Resize(img_size, source=key, target=target),
            spt.data.transforms.CenterCrop(img_size, source=key, target=target),
        )

    def _norm_col_transform(dataset, col="pixels"):
        """Normalize column to zero mean, unit variance."""
        data = dataset[col][:]
        mean = data.mean(0).unsqueeze(0)
        std = data.std(0).unsqueeze(0)
        return lambda x: (x - mean) / std
    
    def forward(self, batch, cfg):    
        with torch.no_grad():
            batch = self.model.encode(
                batch,
                target="embed",
                pixels_key="pixels"
            )
            embedding = batch["embed"][:, :cfg.dinowm.history_size, :, :]  # (B, history_size, patches, D)
            pred_embedding = self.model.predict(embedding)  # (B, T-1, patches, dim)
            target_embedding = batch["embed"][:, cfg.dinowm.num_preds:, :, :]  # (B, T-1, patches, dim)
        return pred_embedding, target_embedding
    
    def pull_eval_data(self, n_steps, frameskip):
        """Setup dataset with image transforms and normalization."""
        dataset = swm.data.VideoDataset(
            self.dataset_name,
            num_steps=n_steps,
            frameskip=frameskip,
            transform=None,
            cache_dir=self.cache_dir,
        )
        if self.type is not "video":
            norm_action_transform = self._norm_col_transform(dataset.dataset, "action")
            norm_proprio_transform = self._norm_col_transform(dataset.dataset, "proprio")
            transform = spt.data.transforms.Compose(
                *[self._get_img_pipeline(f"{col}.{i}", f"{col}.{i}", self.img_size) for col in ["pixels"] for i in range(n_steps)],
                spt.data.transforms.WrapTorchTransform(
                    norm_action_transform,
                    source="action",
                    target="action",
                ),
                spt.data.transforms.WrapTorchTransform(
                    norm_proprio_transform,
                    source="proprio",
                    target="proprio",
                ),
            )
        else:
            transform = spt.data.transforms.Compose(*[self._get_img_pipeline(f"{col}.{i}", f"{col}.{i}", img_size) for col in ["pixels"] for i in range(n_steps)])
        dataset.transform = transform
        rnd_gen = torch.Generator().manual_seed(cfg.seed)
        _, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )
        logging.info(f"Eval Size: {len(val_set)}")

        return val_set # usage: eval_loader = DataLoader(val_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)


    def get_eval_embeddings(self, eval_loader, num_batches: None):
        # Accumulate embeddings across batches
        all_pred_embeddings = []
        all_target_embeddings = []
        
        logging.info("Starting evaluation on validation set...")
        for batch_idx, batch in enumerate(eval_loader):
            if num_batches is not None and batch_idx >= num_batches:
                logging.info(f"Stopping after {num_batches} batches")
                break
            
            # Move batch to device
            batch['pixels'] = batch['pixels'].to(device)
            
            # Extract embeddings from predictor, take first prediction
            pred_emb, target_emb = self.forward(batch, self.model, cfg)
            pred_step0 = pred_emb[:, 0, :, :]  # (B, num_patches, D)
            target_step0 = target_emb[:, 0, :, :]  # (B, num_patches, D)
            
            # Reshape: flatten spatial dimension
            B, P, D = pred_step0.shape
            pred_flat = pred_step0.reshape(B * P, D)  # (B*P, D)
            target_flat = target_step0.reshape(B * P, D)  # (B*P, D)
            
            all_pred_embeddings.append(pred_flat.cpu())
            all_target_embeddings.append(target_flat.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                logging.info(f"Processed batch {batch_idx + 1}/{len(eval_loader)}")
        
        # Concatenate all batches
        pred_embeddings_all = torch.cat(all_pred_embeddings, dim=0)  # (N_total, D)
        target_embeddings_all = torch.cat(all_target_embeddings, dim=0)  # (N_total, D)

        return pred_embeddings_all, target_embeddings_all
    
    def calculate_metrics(
            self, 
            pred_embeddings_all, 
            target_embeddings_all,
            rollout_loader=None,
            rollout_batches=None
            ):
        # Compute metrics
        results = {}
        
        if self.metrics.rankme.enabled:
            rankme_score = rankme(pred_embeddings_all)
            results['rankme'] = rankme_score
            logging.info(f"  RankMe: {rankme_score:.4f} (range: [1, {pred_embeddings_all.shape[1]}], higher is better)")
        
        if self.metrics.frechet_joint_distance.enabled:
            frechet_dist = frechet_joint_distance(target_embeddings_all, pred_embeddings_all)
            results['frechet_distance'] = frechet_dist
            logging.info(f"  Fréchet Distance: {frechet_dist:.4f} (range: [0, ∞), lower is better)")
        
        if self.metrics.feature_rollout_degradation.enabled:
            all_pred_rollouts = [[] for _ in range(self.metrics.feature_rollout_degradation.num_rollout_steps)]
            all_target_rollouts = [[] for _ in range(self.metrics.feature_rollout_degradation.num_rollout_steps)]
            
            for batch_idx, batch in enumerate(rollout_loader):
                if rollout_batches is not None and batch_idx >= rollout_batches:
                    break
                
                batch['pixels'] = batch['pixels'].to(device)
                for step_idx in range(self.metrics.feature_rollout_degradation.num_rollout_steps):
                    pred_emb, target_emb = self.forward(batch, cfg)
                    pred_step0 = pred_emb[:, 0, :, :]  # (B, num_patches, D)
                    target_step0 = target_emb[:, 0, :, :]  # (B, num_patches, D)
                    
                    # Reshape: flatten spatial dimension
                    B, P, D = pred_step0.shape
                    pred_step0 = pred_step0.reshape(B * P, D)  # (B*P, D)
                    target_step0 = target_step0.reshape(B * P, D)  # (B*P, D)
                
                    all_pred_rollouts[step_idx].append(pred_step0.cpu())
                    all_target_rollouts[step_idx].append(target_step0.cpu())

                    # update todo
                    batch = None
            # Concatenate across batches
            pred_rollouts = [torch.cat(preds, dim=0) if preds else torch.tensor([]) for preds in all_pred_rollouts]
            target_rollouts = [torch.cat(tgts, dim=0) if tgts else torch.tensor([]) for tgts in all_target_rollouts]
            
            degradation = feature_rollout_degradation(pred_rollouts, target_rollouts)
            results['rollout_degradation'] = degradation
            logging.info(f"  Rollout Degradation Rate: {degradation['degradation_rate']:.4f}")
            logging.info(f"    MSE progression: {[f'{m:.4f}' for m in degradation['mse_per_step']]}")
            logging.info(f"    Cosine Similarity: {[f'{c:.4f}' for c in degradation['cosine_sim_per_step']]}")
        return results

