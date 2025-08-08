# cell_tracking_trackastra.py
# Contains the logic to run cell tracking using the Trackastra model
# and parse its output into the format compatible with the TUI tracker.
# V3 (This Update): Fixed label mapping issue by creating proper correspondence between
# Trackastra output and original segmentation labels.

import logging
import tempfile
import shutil
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

try:
    import torch
    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc
    TRACKASTRA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Trackastra or its dependencies not found. Trackastra tracking will be unavailable. Error: {e}")
    TRACKASTRA_AVAILABLE = False


def _parse_trackastra_output(ctc_tracks_df, tracked_masks, original_masks=None):
    """
    Parses the output from trackastra.tracking.graph_to_ctc to generate
    a trajectory DataFrame (trj) and a cell lineage dictionary.

    Args:
        ctc_tracks_df (pd.DataFrame): The DataFrame returned by graph_to_ctc.
        tracked_masks (np.ndarray): The final tracked masks from graph_to_ctc.
        original_masks (np.ndarray, optional): The original input masks to map labels.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The trajectory DataFrame (trj).
            - dict: The cell lineage dictionary.
    """
    if ctc_tracks_df is None or ctc_tracks_df.empty:
        logging.error("Trackastra returned an empty or invalid tracks DataFrame.")
        return pd.DataFrame(), {}

    # Ensure DataFrame has the correct column names
    ctc_tracks_df.columns = ["particle", "t_start", "t_end", "parent_particle"]

    # 1. Build lineage from the DataFrame
    logging.info("Parsing Trackastra DataFrame for lineage information.")
    cell_lineage = {}
    for _, row in ctc_tracks_df.iterrows():
        parent_id = row['parent_particle']
        child_id = row['particle']
        if parent_id != 0:
            if parent_id not in cell_lineage:
                cell_lineage[parent_id] = []
            cell_lineage[parent_id].append(child_id)

    # 2. Build the per-frame trj DataFrame from the tracked masks
    logging.info("Generating per-frame trajectory data from Trackastra's tracked masks.")
    all_props = []
    
    # Create a mapping from tracked labels to original labels if original_masks provided
    label_mapping = {}
    if original_masks is not None:
        logging.info("Creating label mapping from tracked masks to original masks.")
        logging.info(f"Tracked masks shape: {tracked_masks.shape}, Original masks shape: {original_masks.shape}")
        for t in range(min(tracked_masks.shape[2], original_masks.shape[2])):  # Both are now in (height, width, frames) format
            tracked_frame = tracked_masks[:, :, t]
            original_frame = original_masks[:, :, t]
            
            # For each tracked label, find the most overlapping original label
            for tracked_label in np.unique(tracked_frame)[1:]:  # Skip background (0)
                tracked_mask = (tracked_frame == tracked_label)
                if not np.any(tracked_mask):
                    continue
                    
                # Find overlapping original labels
                overlapping_original = original_frame[tracked_mask]
                unique_original, counts = np.unique(overlapping_original[overlapping_original > 0], return_counts=True)
                
                if len(unique_original) > 0:
                    # Use the most frequent original label
                    best_original_label = unique_original[np.argmax(counts)]
                    label_mapping[(t, tracked_label)] = best_original_label
                    # Debug: Log some mapping examples
                    if len(label_mapping) <= 5:  # Only log first few mappings
                        logging.info(f"Label mapping: tracked_label {tracked_label} -> original_label {best_original_label} (frame {t})")
                else:
                    # If no overlap found, set to 0 (will trigger centroid fallback)
                    label_mapping[(t, tracked_label)] = 0
                    logging.warning(f"No overlap found for tracked_label {tracked_label} in frame {t}, setting to 0")
    
    # Extract properties from tracked masks
    for t in range(tracked_masks.shape[2]):  # Now using shape[2] since masks are in (height, width, frames) format
        frame_mask = tracked_masks[:, :, t]
        if frame_mask.max() == 0:  # Skip empty frames
            continue

        properties = ('label', 'centroid', 'area', 'equivalent_diameter', 'perimeter', 'orientation', 'solidity',
                      'eccentricity')

        try:
            props_df = pd.DataFrame(regionprops_table(frame_mask, properties=properties))
            props_df['frame'] = t
            # Fix coordinate mapping to match the original tracking code
            # In skimage.regionprops: centroid[0] = row (y), centroid[1] = column (x)
            props_df.rename(columns={
                'label': 'particle',
                'centroid-0': 'y',  # Row coordinate (height)
                'centroid-1': 'x'   # Column coordinate (width)
            }, inplace=True)
            
            # Add original mask label mapping
            if label_mapping:
                props_df['original_mask_label'] = props_df.apply(
                    lambda row: label_mapping.get((int(row['frame']), int(row['particle'])), 0), 
                    axis=1
                )
                # Debug: Log some mapping examples
                if len(props_df) > 0:
                    sample_mappings = props_df[['particle', 'original_mask_label']].head(3)
                    logging.info(f"Sample label mappings for frame {t}: {sample_mappings.to_dict('records')}")
            else:
                # If no original masks provided, set to 0 (will trigger centroid fallback)
                logging.warning("No original masks provided for label mapping. Setting original_mask_label to 0.")
                props_df['original_mask_label'] = 0
                
            all_props.append(props_df)
        except Exception as e:
            logging.warning(f"Could not extract regionprops from frame {t}. Error: {e}")

    if not all_props:
        logging.warning("No properties could be extracted from any frame of the tracked masks.")
        return pd.DataFrame(), cell_lineage

    trj = pd.concat(all_props, ignore_index=True)

    # Add parent information to the main trj dataframe
    parent_map = ctc_tracks_df.set_index('particle')['parent_particle']
    trj['parent_particle'] = trj['particle'].map(parent_map).fillna(0).astype(int)

    # Log mapping statistics
    if label_mapping:
        mapped_count = sum(1 for v in label_mapping.values() if v != 0)
        total_count = len(label_mapping)
        logging.info(f"Label mapping statistics: {mapped_count}/{total_count} labels successfully mapped to original masks.")
        
        # Debug: Show some examples of successful and failed mappings
        successful_mappings = [(k, v) for k, v in label_mapping.items() if v != 0]
        failed_mappings = [(k, v) for k, v in label_mapping.items() if v == 0]
        
        if successful_mappings:
            logging.info(f"Sample successful mappings: {successful_mappings[:3]}")
        if failed_mappings:
            logging.warning(f"Sample failed mappings: {failed_mappings[:3]}")
    
    logging.info(
        f"Successfully parsed Trackastra output. Found {len(trj)} detections across {trj['particle'].nunique()} tracks.")

    return trj, cell_lineage


def track_with_trackastra(main_app_state):
    """
    Main function to execute tracking using Trackastra.

    Args:
        main_app_state (dict): The main state dictionary of the application.

    Returns:
        tuple: A tuple containing:
            - trj (pd.DataFrame): The resulting trajectory DataFrame.
            - col_tuple (dict): An empty dict for compatibility.
            - col_weights (dict): An empty dict for compatibility.
            - cell_lineage (dict): The resulting lineage dictionary.
    """
    if not TRACKASTRA_AVAILABLE:
        raise ImportError("Trackastra library is not installed or available.")

    logging.info("========== Starting Tracking with Trackastra ==========")

    imgs = main_app_state.get('raw_imgs')
    masks = main_app_state.get('raw_masks')
    params = main_app_state.get('params', {})

    if imgs is None or masks is None:
        raise ValueError("Raw images or masks are not loaded.")

    # --- 1. Get Trackastra parameters from UI ---
    model_name = params.get('Trackastra Model', ('general_2d',))[0]
    linking_mode = params.get('Trackastra Linking Mode', ('greedy',))[0]
    device = params.get('Trackastra Device', ('cuda',))[0]

    logging.info(f"Trackastra parameters: Model='{model_name}', Mode='{linking_mode}', Device='{device}'")

    # --- 2. Prepare data for Trackastra ---
    logging.info(f"Original image shape: {imgs.shape}. Transposing for Trackastra.")
    imgs_for_trackastra = np.transpose(imgs, (2, 0, 1))
    masks_for_trackastra = np.transpose(masks, (2, 0, 1))
    logging.info(f"Transposed image shape for Trackastra: {imgs_for_trackastra.shape}")
    logging.info(f"Original masks shape: {masks.shape}, Transposed masks shape: {masks_for_trackastra.shape}")

    # Keep the original masks for label mapping (use original orientation, not transposed)
    original_masks_for_mapping = masks.copy()  # Use original orientation for label mapping

    temp_dir = tempfile.mkdtemp()
    logging.info(f"Created temporary directory for Trackastra mask output: {temp_dir}")

    trj, cell_lineage = pd.DataFrame(), {}

    try:
        # --- 3. Load model and run tracking ---
        logging.info("Loading pretrained Trackastra model...")
        model = Trackastra.from_pretrained(model_name, device=device)

        logging.info("Running Trackastra tracking...")
        track_graph = model.track(imgs_for_trackastra, masks_for_trackastra, mode=linking_mode, max_distance=128)
        
        # Print track_graph information for debugging
        print(f"Track graph nodes: {len(track_graph.nodes())}")
        print(f"Track graph edges: {len(track_graph.edges())}")
        print(f"Track graph type: {type(track_graph)}")
        
        # Check if track_graph has any attributes
        if hasattr(track_graph, 'graph'):
            print(f"Track graph attributes: {track_graph.graph}")
        
        logging.info(f"Track graph: {len(track_graph.nodes())} nodes, {len(track_graph.edges())} edges")

        logging.info("Converting Trackastra graph to CTC format...")
        # graph_to_ctc returns the tracks DataFrame and the final masks
        ctc_tracks_df, masks_tracked = graph_to_ctc(
            track_graph,
            masks_for_trackastra,
            outdir=temp_dir,  # Still provide outdir for masks
        )

        # --- 4. Parse the output with original masks for label mapping ---
        # Note: masks_tracked is in transposed format (frames, height, width)
        # We need to transpose it back to original format (height, width, frames) for consistency
        logging.info(f"Tracked masks shape before transpose: {masks_tracked.shape}")
        masks_tracked_original_orientation = np.transpose(masks_tracked, (1, 2, 0))
        logging.info(f"Tracked masks shape after transpose: {masks_tracked_original_orientation.shape}")
        
        trj, cell_lineage = _parse_trackastra_output(
            ctc_tracks_df, 
            masks_tracked_original_orientation, 
            original_masks=original_masks_for_mapping
        )

    except Exception as e:
        logging.error(f"An error occurred during Trackastra tracking: {e}", exc_info=True)
        return pd.DataFrame(), {}, {}, {}
    finally:
        # --- 5. Clean up temporary directory ---
        try:
            shutil.rmtree(temp_dir)
            logging.info(f"Removed temporary directory: {temp_dir}")
        except OSError as e:
            logging.error(f"Error removing temporary directory {temp_dir}: {e}")

    return trj, {}, {}, cell_lineage
