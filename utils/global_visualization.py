from pathlib import Path
from loguru import logger as logging
import torch
from utils.latent_trajectory import plot_phase_space

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

    if cfg.visualization.get('phase_space', {}).get('enabled', False):
        try:
            logging.info("Generating Phase Space Plot...")
            plot_phase_space(
                pred_embeddings_all,
                output_path=viz_dir,
                video_name=f"validation_batches",
            )
            logging.info("✓ Phase space visualization complete")
        except Exception as e:
            logging.warning(f"Failed to generate phase space plot: {e}")
    

    
    logging.info(f"\nAll visualizations saved to: {viz_dir}")