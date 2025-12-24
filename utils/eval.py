from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from sklearn.decomposition import PCA
from loguru import logger as logging
from torch.utils.data import DataLoader
import stable_pretraining as spt
import stable_worldmodel as swm
from utils.eval_metrics import rankme, frechet_joint_distance, feature_rollout_degradation


DINO_PATCH_SIZE=14


class EvalFramework():
    def __init__(
        self,
        model: torch.nn.Module,
        dataset_name: str,
        cache_dir: Optional[str],
        img_size: int,
        metrics: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        """Initialize evaluation framework.
        
        Args:
            model: World model to evaluate
            dataset_name: Name of dataset (e.g., 'clevrer_train')
            cache_dir: Cache directory for dataset
            img_size: Image size for processing
            metrics: Dictionary of metric configurations
            device: Device to use (defaults to cuda if available)
        """
        self.model = model
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.img_size = img_size
        self.metrics = metrics
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Determine dataset type
        if self.dataset_name == "clevrer_train":
            self.type = "video"
        else:
            raise ValueError(f"Eval dataset not recognized: {self.dataset_name}")

    @staticmethod
    def _get_img_pipeline(key: str, target: str, img_size: int = 224):
        return spt.data.transforms.Compose(
            spt.data.transforms.ToImage(
                **spt.data.dataset_stats.ImageNet,
                source=key,
                target=target,
            ),
            spt.data.transforms.Resize(img_size, source=key, target=target),
            spt.data.transforms.CenterCrop(img_size, source=key, target=target),
        )
    @staticmethod
    def _get_img_pipeline_minimal(key: str, target: str, img_size: int = 224):
        return spt.data.transforms.Compose(
            spt.data.transforms.ToImage(
                **spt.data.dataset_stats.ImageNet,
                source=key,
                target=target,
            ),
            spt.data.transforms.Resize(img_size, source=key, target=target),
        )

    @staticmethod
    def _norm_col_transform(dataset, col: str = "pixels"):
        data = dataset[col][:]
        mean = data.mean(0).unsqueeze(0)
        std = data.std(0).unsqueeze(0)
        return lambda x: (x - mean) / std

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        history_size: int,
        num_preds: int,
        use_inference_function: bool=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the world model.
        
        Args:
            batch: Input batch with 'pixels' key
            history_size: Number of frames in history
            num_preds: Number of frames to predict
            
        Returns:
            Tuple of (pred_embedding, target_embedding, full_embedding)
                - pred_embedding: (B, num_preds, num_patches, D) predicted embeddings
                - target_embedding: (B, num_preds, num_patches, D) ground truth embeddings
                - full_embedding: (B, T, num_patches, D) all embeddings including history
        """
        with torch.no_grad():
            batch = self.model.encode(
                batch,
                target="embed",
                pixels_key="pixels"
            )
            # Extract history and future embeddings
            full_embedding = batch["embed"]  # (B, T, patches, D)
            embedding = full_embedding[:, :history_size, :, :]  # (B, history_size, patches, D)
            pred_embedding = self.model.predict(embedding, use_inference_function)[:, -1, :, :].unsqueeze(1)  # (B, num_preds, patches, D)
            target_embedding = full_embedding[:, history_size:history_size + num_preds, :, :]  # (B, num_preds, patches, D)
        
        return pred_embedding, target_embedding, full_embedding
    
    def pull_eval_data(
        self,
        n_steps: int,
        frameskip: int,
        seed: int,
    ):
        """Setup dataset with image transforms and normalization.
        
        Args:
            n_steps: Number of timesteps in each sample
            frameskip: Frame skip between samples
            seed: Random seed for reproducibility
            
        Returns:
            Validation dataset with applied transforms
        """

        val_set = swm.data.VideoDataset(
            self.dataset_name+"_val",
            num_steps=n_steps,
            frameskip=frameskip,
            transform=None,
            cache_dir=self.cache_dir,
        )
        # Apply transforms to all steps
        if "clevrer" in self.dataset_name:
            transform = spt.data.transforms.Compose(
                *[self._get_img_pipeline_minimal(f"{col}.{i}", f"{col}.{i}", self.img_size) for col in ["pixels"] for i in range(n_steps)],
            )
            val_set.transform = transform
        else :

            val_transform = spt.data.transforms.Compose(
                *[self._get_img_pipeline(f"{col}.{i}", f"{col}.{i}", self.img_size) for col in ["pixels"] for i in range(n_steps)],
                spt.data.transforms.WrapTorchTransform(
                    self._norm_col_transform(val_set.dataset, "action"),
                    source="action",
                    target="action",
                ),
                spt.data.transforms.WrapTorchTransform(
                    self._norm_col_transform(val_set.dataset, "proprio"),
                    source="proprio",
                    target="proprio",
                ),
            )
            val_set.transform = val_transform
        
        # Split into train/val using seed for reproducibility
        rnd_gen = torch.Generator().manual_seed(seed)
        logging.info(f"Eval Size: {len(val_set)}")

        return val_set
    
    def get_eval_embeddings(
        self,
        eval_loader: DataLoader,
        history_size: int,
        num_preds: int,
        num_batches: Optional[int] = None,
        use_inference_function: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings from validation set.
        
        Args:
            eval_loader: DataLoader for evaluation dataset
            history_size: Number of frames in history
            num_preds: Number of frames to predict
            num_batches: Optional limit on number of batches to process
            
        Returns:
            Tuple of (pred_embeddings_all, target_embeddings_all)
                - pred_embeddings_all: (N_total, D) predicted embeddings
                - target_embeddings_all: (N_total, D) ground truth embeddings
        """
        # Accumulate embeddings across batches
        all_pred_embeddings = []
        all_target_embeddings = []
        
        logging.info("Starting evaluation on validation set...")
        for batch_idx, batch in enumerate(eval_loader):
            if num_batches is not None and batch_idx >= num_batches:
                logging.info(f"Stopping after {num_batches} batches")
                break
            
            # Move batch to device
            batch['pixels'] = batch['pixels'].to(self.device)
            
            pred_emb, target_emb, _ = self.forward(batch, history_size, num_preds, use_inference_function)
            pred_step = pred_emb[:, 0, :, :]  # (B, num_patches, D)
            target_step = target_emb[:, 0, :, :]  # (B, num_patches, D) 
            
            # Reshape: flatten spatial dimension
            B, P, D = pred_step.shape
            pred_flat = pred_step.reshape(B * P, D)  # (B*P, D)
            target_flat = target_step.reshape(B * P, D)  # (B*P, D)
            
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
        pred_embeddings_all: torch.Tensor,
        target_embeddings_all: torch.Tensor,
        rollout_loader: Optional[DataLoader] = None,
        rollout_batches: Optional[int] = None,
        history_size: Optional[int] = None,
        num_preds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics.
        
        Args:
            pred_embeddings_all: Predicted embeddings (N, D)
            target_embeddings_all: Target embeddings (N, D)
            rollout_loader: Optional DataLoader for rollout metric
            rollout_batches: Optional limit on rollout batches
            history_size: Number of history frames (required for rollout)
            num_preds: Number of prediction steps (required for rollout)
            
        Returns:
            Dictionary of computed metrics
        """
        # Compute metrics
        results = {}
        
        if self.metrics.rankme.enabled:
            rankme_score = rankme(pred_embeddings_all)
            results['rankme'] = rankme_score
            logging.info(f"  RankMe: {rankme_score:.4f} (range: [1, {pred_embeddings_all.shape[1]}], higher is better)")
        
        if self.metrics.frechet_joint_distance.enabled:
            # Raw FJD (original dimension)
            frechet_dist = frechet_joint_distance(target_embeddings_all, pred_embeddings_all)
            results['frechet_distance'] = frechet_dist
            logging.info(f"  Fréchet Distance (raw): {frechet_dist:.4f} (range: [0, ∞), lower is better)")
            
            # Reduced dimension FJD for inter-model comparison
            reduce_dim = getattr(self.metrics.frechet_joint_distance, 'reduce_dim', None)
            if reduce_dim is not None and reduce_dim > 0:
                frechet_dist_reduced = self._compute_fjd_with_joint_pca(
                    pred_embeddings_all, target_embeddings_all, reduce_dim
                )
                results['frechet_distance_reduced'] = frechet_dist_reduced
                results['frechet_distance_reduce_dim'] = reduce_dim
                logging.info(f"  Fréchet Distance (PCA-{reduce_dim}): {frechet_dist_reduced:.4f}")
        
        if self.metrics.feature_rollout_degradation.enabled:
            if rollout_loader is None:
                logging.warning("Feature rollout degradation enabled but no rollout_loader provided. Skipping.")
            elif history_size is None or num_preds is None:
                logging.warning("Feature rollout degradation requires history_size and num_preds. Skipping.")
            else:
                degradation = self._compute_rollout_degradation(
                    rollout_loader,
                    rollout_batches,
                    history_size,
                    num_preds
                )
                results['rollout_degradation'] = degradation
                logging.info(f"  Rollout Degradation Rate: {degradation['degradation_rate']:.4f}")
                logging.info(f"    MSE progression: {[f'{m:.4f}' for m in degradation['mse_per_step']]}")
                logging.info(f"    Cosine Similarity: {[f'{c:.4f}' for c in degradation['cosine_sim_per_step']]}")
        
        return results

    def _compute_fjd_with_joint_pca(
        self,
        pred_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        reduce_dim: int,
    ) -> float:
        """Compute FJD with joint PCA dimensionality reduction.
        
        To ensure fair comparison between different models, we perform PCA on
        the concatenation of pred and target embeddings. This ensures both
        distributions are projected onto the same principal components,
        making the FJD comparable across models with different embedding dimensions.
        
        Args:
            pred_embeddings: Predicted embeddings (N, D)
            target_embeddings: Target embeddings (N, D)
            reduce_dim: Target dimension after PCA reduction
            
        Returns:
            FJD computed on the reduced dimension embeddings
        """
        # Convert to numpy
        pred_np = pred_embeddings.numpy() if isinstance(pred_embeddings, torch.Tensor) else pred_embeddings
        target_np = target_embeddings.numpy() if isinstance(target_embeddings, torch.Tensor) else target_embeddings
        
        # Concatenate for joint PCA fitting
        # Shape: (2N, D) - this ensures both distributions share the same PCA basis
        joint_embeddings = np.concatenate([pred_np, target_np], axis=0)
        
        # Fit PCA on joint embeddings
        n_components = min(reduce_dim, joint_embeddings.shape[1], joint_embeddings.shape[0])
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(joint_embeddings)
        
        # Transform both using the same fitted PCA
        pred_reduced = pca.transform(pred_np)  # (N, reduce_dim)
        target_reduced = pca.transform(target_np)  # (N, reduce_dim)
        
        # Log explained variance for debugging
        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        logging.info(f"    Joint PCA: {n_components} components, {explained_var:.1f}% variance explained")
        
        # Convert back to torch and compute FJD
        pred_reduced_torch = torch.from_numpy(pred_reduced).float()
        target_reduced_torch = torch.from_numpy(target_reduced).float()
        
        return frechet_joint_distance(target_reduced_torch, pred_reduced_torch)

    def _compute_rollout_degradation(
        self,
        rollout_loader: DataLoader,
        rollout_batches: Optional[int],
        history_size: int,
        num_preds: int,
    ) -> Dict[str, Any]:
        """Compute feature rollout degradation metric.
        
        Performs autoregressive rollout: at each step, uses predicted embeddings
        as input to the model to generate the next prediction. This measures
        how well the model maintains prediction quality over multiple steps.
        
        Args:
            rollout_loader: DataLoader for rollout evaluation
            rollout_batches: Optional limit on number of batches
            history_size: Number of frames in history
            num_preds: Number of frames to predict per step
            
        Returns:
            Dictionary with degradation metrics including:
                - mse_per_step: MSE at each rollout step
                - cosine_sim_per_step: Cosine similarity at each step
                - degradation_rate: Overall degradation rate
        """
        num_rollout_steps = self.metrics.feature_rollout_degradation.num_rollout_steps
        all_pred_rollouts = [[] for _ in range(num_rollout_steps)]
        all_target_rollouts = []
        
        logging.info(f"Computing rollout degradation over {num_rollout_steps} steps...")
        
        for batch_idx, batch in enumerate(rollout_loader):
            if rollout_batches is not None and batch_idx >= rollout_batches:
                break
            
            batch['pixels'] = batch['pixels'].to(self.device)
            _, _, full_embedding = self.forward(batch, history_size, num_preds)
            embedding = full_embedding[:, :history_size, :, :] 
            target_embedding = full_embedding[:, -num_rollout_steps:, :, :]
            all_target_rollouts.append(target_embedding.cpu())  # (B, num_rollout_steps, P, D)

        
            for t in range(num_rollout_steps):
            # Extract embeddings from predictor, take LAST prediction (inference-style)
                pred_embedding = self.model.predict(embedding)[:, -1, :, :].unsqueeze(1)  # (B, num_preds, patches, D)
                pred_step = pred_embedding[:, 0, :, :]  # (B, num_patches, D)
                embedding = torch.cat([embedding, pred_step.unsqueeze(1)], dim=1)[:, -history_size:, :, :]  # (B, history_size, P, D), update history

                all_pred_rollouts[t].append(pred_step.cpu())
            if (batch_idx + 1) % max(1, len(rollout_loader) // 10) == 0:
                logging.info(f"  Processed {batch_idx + 1} rollout batches")
        
        # Concatenate across batches
        pred_rollouts = [
            torch.cat(preds, dim=0) if preds else torch.tensor([])
            for preds in all_pred_rollouts
        ] # shape: (num_rollout_steps, rollout_batches * B, P, D)
        target_rollouts = torch.cat(all_target_rollouts, dim=0)  # (rollout_batches * B, num_rollout_steps, P, D)
        target_rollouts = target_rollouts.permute(1, 0, 2, 3)  # (num_rollout_steps, rollout_batches * B, P, D)

        # Compute degradation metrics
        degradation = feature_rollout_degradation(pred_rollouts, target_rollouts)
        
        return degradation

