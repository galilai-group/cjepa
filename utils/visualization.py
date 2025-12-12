from pathlib import Path
from loguru import logger as logging
import torch
from utils.latent_trajectory import (
    plot_latent_energy_trajectory,
    plot_phase_space,
    plot_slot_interaction_trajectory,
)
from utils.interaction_dynamics import (
    plot_vector_field_prediction,
    plot_temporal_similarity_matrix,
    plot_temporal_difference_heatmap,
)
from utils.eval import extract_predictor_embeddings

def visualize(pred_embeddings_all, cfg, output_dir):
        
    # Create visualization directory
    viz_dir = cfg.get('visualization_dir', None)
    if viz_dir is None:
        viz_dir = output_dir / "visualizations"
    else:
        viz_dir = Path(viz_dir)
    
    if cfg.get('visualization', {}).get('enabled', False):
        viz_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Visualization directory: {viz_dir}")
    
    logging.info("\n" + "=" * 60)
    logging.info("Generating Dynamics Visualizations...")
    logging.info("=" * 60)
    

    if cfg.visualization.get('latent_trajectory', {}).get('enabled', False):
        try:
            logging.info("Generating Latent Energy Trajectory...")
            plot_latent_energy_trajectory(
                pred_embeddings_all,
                output_path=viz_dir,
                video_name=f"validation_all_batches",
            )
            logging.info("✓ Latent trajectory visualization complete")
        except Exception as e:
            logging.warning(f"Failed to generate latent trajectory: {e}")
    
    # 2. Phase Space Plot (State Norm vs Temporal Derivative)
    if cfg.visualization.get('phase_space', {}).get('enabled', False):
        try:
            logging.info("Generating Phase Space Plot...")
            plot_phase_space(
                pred_embeddings_all,
                output_path=viz_dir,
                video_name=f"validation_all_batches",
            )
            logging.info("✓ Phase space visualization complete")
        except Exception as e:
            logging.warning(f"Failed to generate phase space plot: {e}")
    
    # 3. Vector Field of Future Predictions (JEPA-style)
    if cfg.visualization.get('vector_field', {}).get('enabled', False):
        try:
            logging.info("Generating Vector Field Plot...")
            # Need predicted embeddings for vector field
            # Re-run evaluation to get predictions
            all_pred_rollouts = [[] for _ in range(2)]
            all_pred_embeddings_rollout = [[] for _ in range(2)]
            
            for batch_idx, batch in enumerate(eval_loader):
                if num_batches is not None and batch_idx >= num_batches:
                    break
                
                batch['pixels'] = batch['pixels'].to(device)
                pred_emb, _ = extract_predictor_embeddings(batch, world_model, cfg)
                
                # Collect step 0 and step 1
                for step_idx in range(min(2, pred_emb.shape[1])):
                    B, P, D = pred_emb[:, step_idx, :, :].shape
                    flat = pred_emb[:, step_idx, :, :].reshape(B * P, D)
                    all_pred_embeddings_rollout[step_idx].append(flat.cpu())
            
            if all_pred_embeddings_rollout[0] and all_pred_embeddings_rollout[1]:
                step0 = torch.cat(all_pred_embeddings_rollout[0], dim=0)
                step1 = torch.cat(all_pred_embeddings_rollout[1], dim=0)
                
                # Ensure same length
                min_len = min(len(step0), len(step1))
                step0 = step0[:min_len]
                step1 = step1[:min_len]
                
                plot_vector_field_prediction(
                    embeddings=step0,
                    predictions=step1,
                    output_path=viz_dir,
                    video_name="validation_all_batches",
                )
                logging.info("✓ Vector field visualization complete")
        except Exception as e:
            logging.warning(f"Failed to generate vector field plot: {e}")
    
    # 4. Temporal Self-Similarity Matrix
    if cfg.visualization.get('temporal_similarity', {}).get('enabled', False):
        try:
            logging.info("Generating Temporal Similarity Matrix...")
            plot_temporal_similarity_matrix(
                pred_embeddings_all,
                output_path=viz_dir,
                video_name="validation_all_batches",
            )
            logging.info("✓ Temporal similarity visualization complete")
        except Exception as e:
            logging.warning(f"Failed to generate similarity matrix: {e}")
    
    # 5. Temporal Distance Matrix
    if cfg.visualization.get('temporal_distance', {}).get('enabled', False):
        try:
            logging.info("Generating Temporal Distance Matrix...")
            window_size = cfg.visualization.get('temporal_distance', {}).get('window_size', 1)
            plot_temporal_difference_heatmap(
                pred_embeddings_all,
                output_path=viz_dir,
                video_name="validation_all_batches",
                window_size=window_size,
            )
            logging.info("✓ Temporal distance visualization complete")
        except Exception as e:
            logging.warning(f"Failed to generate distance matrix: {e}")
    
    logging.info(f"\nAll visualizations saved to: {viz_dir}")