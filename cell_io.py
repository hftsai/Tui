# Author: ImagineA / Andrei Rares
# Date: 2018-08-18
# Modified by Gemini to fix int64 saving error for ID masks.
# Further modified by Gemini to remove invalid 'errors' argument from sort_values.
# Further modified by Gemini to correct GIF saving method with imageio.v3.
# V1 Update: Added cell count output to experiment parameters.
# V2 Update (User Request): Added saving of lineage relationship table and track plots.
# V3 Update (User Request): Added saving of raw ID mask animation.
# V4 (Gemini): Added saving of command output (log) for tracking operations.
# V5 (Gemini & User Request): Added saving of results in CTC RES format and enabling evaluation button.
# V6 (User Request): Save comprehensive tracking statistics to a JSON file.
# V7 (Gemini): Removed calculated_stats from save_results signature,
#                           implemented res_track.txt generation, and uses main_app_state for stats.
# V8 (Gemini): Filter saved res_track.txt and masks by cell_visibility.
# V9 (Gemini): Addressed SettingWithCopyWarning by explicitly copying DataFrame.
# V10 (Gemini): Fixed parent linking for res_track.txt when parents are hidden.
# V11 (Gemini): Forced stats calculation on save and changed stats file to .txt.
# V12 (Gemini): Added tracking mode to the saved results folder name.
# V13 (Gemini): Integrated automatic CTC evaluation into the save process.
# V15 (Gemini): Expanded and corrected trajectory enrichment for ILP/Trackastra modes.
# V16 (Gemini - This Update): Standardized the output columns for tracks.csv.

import numpy as np
import os
from os.path import join, split, normpath, exists, basename
from datetime import datetime
import glob
import logging
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import QMessageBox, QApplication, QProgressDialog
from PyQt5.QtCore import Qt
import tifffile
from skimage.measure import regionprops_table
import imageio.v3 as iio
import yaml  # Add this import at the top with others
import traceback
import time

from cell_drawing import create_colorized_masks, create_colorized_tracks, create_track_overview, create_raw_id_mask_animation_frames
from cell_evaluation import run_ctc_evaluation_api
from ctc_metrics.metrics import op_clb
from cell_track_saving import save_singular_tracks, save_mitosis_tracks, save_fusion_tracks


def read_img_sequence(path, file_extension):
    """Read a sequence of images using imageio instead of PIMS."""
    file_pattern = join(path, f'*.{file_extension}')
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise IOError(f"No files found matching pattern: {file_pattern}")
    
    logging.info(f"Loading {len(files)} images from {path}...")
    
    # Read first image to get dimensions
    first_image = iio.imread(files[0])
    height, width = first_image.shape[:2]
    num_frames = len(files)
    
    # Pre-allocate array for better performance
    if len(first_image.shape) == 2:  # Grayscale
        images = np.zeros((height, width, num_frames), dtype=first_image.dtype)
    else:  # Color
        images = np.zeros((height, width, first_image.shape[2], num_frames), dtype=first_image.dtype)
    
    # Load images with progress logging
    for i, file_path in enumerate(files):
        if i % max(1, num_frames // 10) == 0:  # Log progress every 10% or at least once
            logging.info(f"Loading image {i+1}/{num_frames} ({((i+1)/num_frames)*100:.1f}%)")
        
        try:
            img = iio.imread(file_path)
            if len(img.shape) == 2:  # Grayscale
                images[:, :, i] = img
            else:  # Color
                images[:, :, :, i] = img
        except Exception as e:
            logging.error(f"Error loading image {file_path}: {e}")
            raise
    
    logging.info(f"Successfully loaded {num_frames} images with shape {images.shape}")
    return images


def _enrich_trajectory_if_needed(trj, id_masks, main_app_state):
    """
    Checks for geometric properties in the trajectory and calculates them from masks if missing.
    This is common for ILP or Trackastra outputs which may only provide core tracking data.
    """
    if trj is None or trj.empty:
        return trj

    # Define a more comprehensive set of geometric properties to ensure exists
    required_cols = [
        'area', 'equivalent_diameter', 'perimeter', 'major_axis_length',
        'minor_axis_length', 'orientation', 'solidity', 'eccentricity'
    ]
    missing_cols = [col for col in required_cols if col not in trj.columns]

    if not missing_cols:
        logging.info("Trajectory already contains all required geometric properties. No enrichment needed.")
        return trj

    if id_masks is None:
        logging.warning("Cannot enrich trajectory with geometric properties: ID masks are not available.")
        trj_copy = trj.copy()
        for col in missing_cols:
            trj_copy[col] = np.nan
        return trj_copy

    logging.info(f"Enriching trajectory with missing geometric properties: {missing_cols}")
    with pg.BusyCursor():
        all_props_list = []
        num_frames = id_masks.shape[2]
        # Dynamically define properties to measure based on what's missing
        properties_to_measure = ('label', 'centroid') + tuple(missing_cols)

        for frame_idx in range(num_frames):
            mask_slice = id_masks[:, :, frame_idx]
            if np.any(mask_slice > 0):
                try:
                    props = regionprops_table(mask_slice, properties=properties_to_measure)
                    props_df = pd.DataFrame(props)
                    props_df['frame'] = frame_idx
                    all_props_list.append(props_df)
                except Exception as e:
                    logging.warning(f"Could not calculate regionprops for frame {frame_idx}. Error: {e}")

        if not all_props_list:
            logging.warning("Enrichment failed: No properties could be extracted from masks.")
            trj_copy = trj.copy()
            for col in missing_cols:
                trj_copy[col] = np.nan
            return trj_copy

        props_df_all_frames = pd.concat(all_props_list, ignore_index=True)
        props_df_all_frames.rename(columns={
            'label': 'particle', 'centroid-0': 'y_new', 'centroid-1': 'x_new'}, inplace=True)

        trj_copy = trj.copy()
        for col in ['frame', 'particle']:
            trj_copy[col] = pd.to_numeric(trj_copy[col], errors='coerce').astype('Int64')
        props_df_all_frames['frame'] = props_df_all_frames['frame'].astype('Int64')
        props_df_all_frames['particle'] = props_df_all_frames['particle'].astype('Int64')

        enriched_trj = pd.merge(trj_copy, props_df_all_frames, on=['particle', 'frame'], how='left')

        if 'y_new' in enriched_trj.columns:
            enriched_trj['y'] = enriched_trj['y_new'].fillna(enriched_trj['y'])
            enriched_trj.drop(columns=['y_new'], inplace=True)
        if 'x_new' in enriched_trj.columns:
            enriched_trj['x'] = enriched_trj['x_new'].fillna(enriched_trj['x'])
            enriched_trj.drop(columns=['x_new'], inplace=True)

        if main_app_state and 'col_tuple' in main_app_state:
            current_original = main_app_state['col_tuple'].get('original', [])
            for col in required_cols:
                if col not in current_original:
                    current_original.append(col)
            main_app_state['col_tuple']['original'] = current_original
            logging.info(f"Updated col_tuple with enriched properties: {current_original}")

    logging.info("Trajectory enrichment complete.")
    return enriched_trj


def save_lineage_info_to_csv(path_out, trj, cell_lineage):
    """
    Saves lineage relationship information (parent, daughter, mitosis frames) to a CSV file.
    """
    logging.info('================= Saving Lineage Relationship Table =================')
    if not cell_lineage or trj.empty:
        logging.warning("No lineage data or trajectory data to save for lineage table.")
        return
    lineage_records = []
    if 'particle' in trj.columns:
        trj['particle'] = pd.to_numeric(trj['particle'], errors='coerce')
    if 'frame' in trj.columns:
        trj['frame'] = pd.to_numeric(trj['frame'], errors='coerce')

    for parent_id_str, daughter_ids in cell_lineage.items():
        try:
            parent_id = int(parent_id_str)
        except ValueError:
            logging.warning(f"Skipping non-integer parent_id '{parent_id_str}' in lineage table.")
            continue
        if not daughter_ids: continue

        parent_frames = trj[trj['particle'] == parent_id]['frame']
        mitosis_frame_parent_end = parent_frames.max() if not parent_frames.empty else pd.NA

        for daughter_id_str in daughter_ids:
            try:
                daughter_id = int(daughter_id_str)
            except ValueError:
                logging.warning(
                    f"Skipping non-integer daughter_id '{daughter_id_str}' for parent {parent_id} in lineage table.")
                continue

            daughter_frames = trj[trj['particle'] == daughter_id]['frame']
            mitosis_frame_daughter_start = daughter_frames.min() if not daughter_frames.empty else pd.NA
            lineage_records.append({
                'parent_id': parent_id, 'daughter_id': daughter_id,
                'mitosis_frame_parent_end': mitosis_frame_parent_end,
                'mitosis_frame_daughter_start': mitosis_frame_daughter_start
            })
    if not lineage_records:
        logging.info("No valid lineage records to save.")
        return
    lineage_df = pd.DataFrame(lineage_records)
    try:
        lineage_df['parent_id'] = pd.to_numeric(lineage_df['parent_id'], errors='coerce')
        lineage_df['daughter_id'] = pd.to_numeric(lineage_df['daughter_id'], errors='coerce')

        lineage_df.sort_values(by=['parent_id', 'daughter_id']).to_csv(
            join(path_out, 'lineage_relationships.csv'), index=False)
        logging.info(f"Lineage relationship table saved to {join(path_out, 'lineage_relationships.csv')}")
    except Exception as e:
        logging.error(f"Error saving lineage relationship table: {e}")


def save_ctc_res_formatted_results(res_sequence_path, trj, id_masks, main_app_state=None, ui_elements=None):
    """
    Saves tracking results in the Cell Tracking Challenge RES format.
    Filters tracks and masks based on main_app_state['cell_visibility'].
    res_track.txt format: L B E P (track_id, frame_start, frame_end, parent_id)
    Parent ID is 0 if no parent or if parent is not visible.
    """
    logging.info(f"=== ENTERING save_ctc_res_formatted_results ===")
    logging.info(f"res_sequence_path: {res_sequence_path}")
    logging.info(f"trj is None: {trj is None}")
    logging.info(f"id_masks is None: {id_masks is None}")
    if id_masks is not None:
        logging.info(f"id_masks shape: {id_masks.shape}")
    logging.info(f"main_app_state is None: {main_app_state is None}")
    
    if not exists(res_sequence_path):
        os.makedirs(res_sequence_path, exist_ok=True)

    logging.info(f"Saving CTC RES formatted results to: {res_sequence_path}")

    res_track_file = join(res_sequence_path, "res_track.txt") # Save res_track.txt in root
    cell_visibility = main_app_state.get('cell_visibility', {}) if main_app_state else {}
    # Ensure keys in visible_track_ids are integers
    visible_track_ids = {
        int(k) for k, v in cell_visibility.items()
        if v and pd.notna(pd.to_numeric(k, errors='coerce'))
    }
    logging.info(f"Number of tracks marked as visible: {len(visible_track_ids)}")

    if trj is not None and not trj.empty and 'particle' in trj.columns and 'frame' in trj.columns:
        logging.info("Generating res_track.txt for CTC evaluation (filtered by visibility).")

        trj_res = trj.copy()
        trj_res['particle'] = pd.to_numeric(trj_res['particle'], errors='coerce')
        trj_res.dropna(subset=['particle'], inplace=True)

        if trj_res.empty:
            logging.warning("Trajectory became empty after coercing particle IDs to numeric for res_track.txt.")
            try:
                with open(res_track_file, 'w') as f:
                    pass
                logging.info(f"  Created empty res_track.txt as input trajectory was empty after processing.")
            except Exception as e_empty_save:
                logging.error(f"  Error saving empty res_track.txt: {e_empty_save}")
            return

        trj_res['particle'] = trj_res['particle'].astype(int)

        if visible_track_ids:
            trj_res_visible_filtered = trj_res[trj_res['particle'].isin(visible_track_ids)]
            trj_res_visible = trj_res_visible_filtered.copy()
            logging.info(
                f"  Number of trajectory entries after filtering by visibility: {len(trj_res_visible)} (Original: {len(trj_res)})")
        else:
            logging.warning("  No tracks are marked as visible. res_track.txt will be empty or reflect no tracks.")
            trj_res_visible = pd.DataFrame(columns=trj_res.columns)

        logging.info("Performing pixel presence check for each track across its lifespan...")
        missing_pixel_tracks = {}
        for track_id in visible_track_ids:
            track_frames = trj_res_visible[trj_res_visible['particle'] == track_id]['frame'].values
            if len(track_frames) == 0:
                continue
            frame_start, frame_end = int(track_frames.min()), int(track_frames.max())
            for f in range(frame_start, frame_end + 1):
                if f < 0 or f >= id_masks.shape[2]:
                    logging.warning(f"Frame {f} for track {track_id} is out of bounds.")
                    continue
                mask_slice = id_masks[:, :, f]
                unique_ids = np.unique(mask_slice)
                if track_id not in unique_ids:
                    missing_pixel_tracks.setdefault(track_id, []).append(f)
        # Print summary
        if missing_pixel_tracks:
            for tid, frames in missing_pixel_tracks.items():
                logging.warning(f"[DEBUG] Track {tid} is missing in frames: {frames}")
        else:
            logging.info("All tracks have pixel presence in their full lifespan.")

        if 'parent_particle' not in trj_res_visible.columns:
            trj_res_visible.loc[:, 'parent_particle'] = 0
        else:
            trj_res_visible.loc[:, 'parent_particle'] = pd.to_numeric(trj_res_visible['parent_particle'],
                                                                      errors='coerce')
            trj_res_visible.loc[:, 'parent_particle'] = trj_res_visible['parent_particle'].fillna(0).astype(int)
            trj_res_visible.loc[trj_res_visible['parent_particle'] < 0, 'parent_particle'] = 0

        res_track_entries = []
        if not trj_res_visible.empty:
            # Get a set of all particle IDs that will actually be in res_track.txt
            final_track_ids_in_res_track = set(trj_res_visible['particle'].unique())

            for particle_id, group in trj_res_visible.groupby('particle'):
                if group.empty:
                    continue
                frame_start = int(group['frame'].min())
                frame_end = int(group['frame'].max())

                parent_id_candidate = int(group['parent_particle'].iloc[0])
                actual_parent_for_res_track = 0  # Default to no parent

                if parent_id_candidate != 0:
                    if parent_id_candidate in final_track_ids_in_res_track:
                        actual_parent_for_res_track = parent_id_candidate
                    else:
                        # This parent is not visible/not in the final list, so child becomes a root in res_track.txt
                        logging.warning(
                            f"  res_track.txt: Visible track {particle_id} has parent {parent_id_candidate} which is NOT in final res_track list. Setting parent to 0 for this entry.")
                        actual_parent_for_res_track = 0

                res_track_entries.append({
                    'L': int(particle_id),
                    'B': frame_start,
                    'E': frame_end,
                    'P': actual_parent_for_res_track
                })
            res_track_entries.sort(key=lambda x: x['L'])

        try:
            with open(res_track_file, 'w') as f:
                for entry in res_track_entries:
                    f.write(f"{entry['L']} {entry['B']} {entry['E']} {entry['P']}\n")
            logging.info(f"  res_track.txt saved with {len(res_track_entries)} visible tracks.")
            
            # Also save res_track.txt in the root directory for the evaluation library
            res_track_file_root = join(res_sequence_path, "res_track.txt")
            with open(res_track_file_root, 'w') as f:
                for entry in res_track_entries:
                    f.write(f"{entry['L']} {entry['B']} {entry['E']} {entry['P']}\n")
            logging.info(f"  res_track.txt also saved in root directory for evaluation library.")
            
            # Debug: Read and log the first few lines of the saved file
            logging.info("=== DEBUG: Content of res_track.txt ===")
            try:
                with open(res_track_file, 'r') as f:
                    lines = f.readlines()
                    logging.info(f"Total lines in res_track.txt: {len(lines)}")
                    if lines:
                        logging.info("First 5 lines:")
                        for i, line in enumerate(lines[:5]):
                            logging.info(f"  Line {i+1}: {line.strip()}")
                        if len(lines) > 5:
                            logging.info(f"  ... and {len(lines)-5} more lines")
                    else:
                        logging.warning("res_track.txt is empty!")
            except Exception as e:
                logging.error(f"Error reading res_track.txt for debugging: {e}")
            logging.info("=== END DEBUG ===")
            
        except Exception as e:
            logging.error(f"  Error saving res_track.txt: {e}")
            if ui_elements and ui_elements.get('win'):
                QMessageBox.critical(ui_elements.get('win'), "Save Error",
                                     f"Could not save res_track.txt: {e}")

        # After creating the filtered masks, save them to the RES folder in root
        if id_masks is not None:
            logging.info(f"  Processing {id_masks.shape[2]} frames for RES mask TIFF files (filtered by visibility).")
            if not visible_track_ids:
                logging.warning("No visible track IDs provided; this may result in empty masks if not intended.")
            
            saved_count = 0
            for frame_num in range(id_masks.shape[2]):
                original_mask = id_masks[:, :, frame_num]
                # Create a new mask that only contains visible tracks
                final_mask = np.zeros_like(original_mask, dtype=np.uint16)
                
                # Get the track IDs present in this frame from the filtered trajectory
                track_ids_in_frame = set(trj_res[trj_res['frame'] == frame_num]['particle'].unique())

                # Get unique labels from the original mask
                unique_labels = np.unique(original_mask)
                unique_labels = unique_labels[unique_labels != 0]  # Exclude background

                # Copy only visible tracks to the final mask
                for label in unique_labels:
                    if label in visible_track_ids:
                        final_mask[original_mask == label] = label

                # Always save a mask file, even if it's empty, to match GT count
                mask_filename = f'mask{frame_num:03d}.tif'
                tifffile.imwrite(join(res_sequence_path, mask_filename), final_mask.astype(np.uint16))
                saved_count += 1

            logging.info(f"  Saved {saved_count}/{id_masks.shape[2]} filtered RES mask TIFF files to root directory.")
        else:
            logging.warning("No trajectory or mask data available to save in CTC RES format.")

    else:
        logging.warning("No trajectory data available to generate res_track.txt for CTC evaluation.")


def _calculate_comprehensive_stats(main_app_state):
    """Calculates a comprehensive set of statistics from the tracking data."""
    stats = {}
    trj = main_app_state.get('trj')
    cell_visibility = main_app_state.get('cell_visibility', {})
    cell_lineage = main_app_state.get('cell_lineage', {})
    ancestry = main_app_state.get('ancestry', {})
    raw_imgs = main_app_state.get('raw_imgs')
    params = main_app_state.get('params', {})
    frames_per_hour = params.get('Frames per hour', (12,))[0]

    if not isinstance(frames_per_hour, (int, float)) or frames_per_hour <= 0:
        frames_per_hour = 12

    if trj is None or trj.empty:
        stats["error"] = "No tracking data available."
        return stats

    trj_stats = trj.copy()
    trj_stats['particle'] = pd.to_numeric(trj_stats['particle'], errors='coerce').dropna().astype(int)
    trj_stats['frame'] = pd.to_numeric(trj_stats['frame'], errors='coerce').dropna().astype(int)

    stats["total_unique_tracks"] = trj_stats['particle'].nunique()

    track_lengths_all = trj_stats.groupby('particle')['frame'].nunique()
    stats["average_track_length_frames"] = track_lengths_all.mean() if not track_lengths_all.empty else 0
    stats["median_track_length_frames"] = track_lengths_all.median() if not track_lengths_all.empty else 0

    total_frames_sequence = raw_imgs.shape[2] if raw_imgs is not None else (
        trj_stats['frame'].max() + 1 if not trj_stats.empty else 0)
    stats["total_frames_in_sequence"] = total_frames_sequence

    visible_track_ids = {tid for tid, is_vis in cell_visibility.items() if is_vis and isinstance(tid, int)}
    if visible_track_ids:
        trj_visible = trj_stats[trj_stats['particle'].isin(visible_track_ids)]
        if not trj_visible.empty:
            stats["visible_unique_tracks"] = trj_visible['particle'].nunique()
            track_lengths_visible = trj_visible.groupby('particle')['frame'].nunique()
            stats[
                "average_visible_track_length_frames"] = track_lengths_visible.mean() if not track_lengths_visible.empty else 0
        else:
            stats["visible_unique_tracks"] = 0
            stats["average_visible_track_length_frames"] = 0
    else:
        stats["visible_unique_tracks"] = 0
        stats["average_visible_track_length_frames"] = 0

    parents_with_1_daughter = sum(1 for d in cell_lineage.values() if len(d) == 1)
    parents_with_2_daughters = sum(1 for d in cell_lineage.values() if len(d) == 2)
    parents_with_more_than_2_daughters = sum(1 for d in cell_lineage.values() if len(d) > 2)

    stats["lineage_division_events_1_daughter"] = parents_with_1_daughter
    stats["mitosis_events (2 daughters)"] = parents_with_2_daughters
    stats["lineage_division_events_more_than_2_daughters"] = parents_with_more_than_2_daughters

    interdivision_times_frames = []
    if cell_lineage:
        for parent_id_str, daughter_ids_list in cell_lineage.items():
            try:
                parent_id = int(parent_id_str)
                if len(daughter_ids_list) >= 2:
                    parent_track_data = trj_stats[trj_stats['particle'] == parent_id]
                    if parent_track_data.empty: continue
                    division_frame = parent_track_data['frame'].max()
                    parent_birth_frame = parent_track_data['frame'].min()
                    interdivision_time = division_frame - parent_birth_frame
                    if interdivision_time > 0:
                        interdivision_times_frames.append(interdivision_time)
            except ValueError:
                continue

    if interdivision_times_frames:
        stats["interdivision_time_avg_frames"] = np.mean(interdivision_times_frames)
        stats["interdivision_time_avg_hours"] = stats["interdivision_time_avg_frames"] / frames_per_hour
        if stats["interdivision_time_avg_hours"] > 0:
            stats["population_doubling_rate_per_hour"] = 1.0 / stats["interdivision_time_avg_hours"]
    else:
        stats["population_doubling_rate_per_hour"] = "N/A"

    stats["fusion_events (2 parents)"] = sum(1 for p in ancestry.values() if len(p) == 2)
    stats["fusion_events (>2 parents)"] = sum(1 for p in ancestry.values() if len(p) > 2)

    if total_frames_sequence > 0:
        stats["cell_count_at_first_frame"] = trj_stats[trj_stats['frame'] == 0]['particle'].nunique()
        stats["cell_count_at_last_frame"] = trj_stats[trj_stats['frame'] == (total_frames_sequence - 1)][
            'particle'].nunique()

    # --- Lifecycle time calculation ---
    lifecycle_times_frames = []
    for track_id, group in trj_stats.groupby('particle'):
        birth_frame = group['frame'].min()
        end_frame = group['frame'].max()
        lifecycle_time = end_frame - birth_frame + 1  # inclusive
        if lifecycle_time > 0:
            lifecycle_times_frames.append(lifecycle_time)
    if lifecycle_times_frames:
        stats['lifecycle_time_avg_frames'] = np.mean(lifecycle_times_frames)
        stats['lifecycle_time_median_frames'] = np.median(lifecycle_times_frames)
        stats['lifecycle_time_min_frames'] = np.min(lifecycle_times_frames)
        stats['lifecycle_time_max_frames'] = np.max(lifecycle_times_frames)
        stats['lifecycle_time_avg_hours'] = stats['lifecycle_time_avg_frames'] / frames_per_hour
        stats['lifecycle_time_median_hours'] = stats['lifecycle_time_median_frames'] / frames_per_hour
        stats['lifecycle_time_min_hours'] = stats['lifecycle_time_min_frames'] / frames_per_hour
        stats['lifecycle_time_max_hours'] = stats['lifecycle_time_max_frames'] / frames_per_hour
    else:
        stats['lifecycle_time_avg_frames'] = 'N/A'
        stats['lifecycle_time_median_frames'] = 'N/A'
        stats['lifecycle_time_min_frames'] = 'N/A'
        stats['lifecycle_time_max_frames'] = 'N/A'
        stats['lifecycle_time_avg_hours'] = 'N/A'
        stats['lifecycle_time_median_hours'] = 'N/A'
        stats['lifecycle_time_min_hours'] = 'N/A'
        stats['lifecycle_time_max_hours'] = 'N/A'

    main_app_state['calculated_stats'] = stats
    return stats


def save_tracking_statistics(path_out, stats_dict):
    """Saves the comprehensive tracking statistics dictionary to a TXT file."""
    if not stats_dict or "error" in stats_dict:
        logging.warning("No valid statistics to save.")
        return

    stats_file_path = join(path_out, 'tracking_statistics.txt')
    try:
        # Helper to convert numpy types for printing
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(stats_file_path, 'w', encoding='utf-8') as f_stats:
            f_stats.write("Comprehensive Tracking Statistics\n")
            f_stats.write("=" * 35 + "\n")
            for key, value in stats_dict.items():
                formatted_key = key.replace('_', ' ').title()
                # Apply numpy conversion here before formatting
                value = convert_numpy_types(value)
                if isinstance(value, float):
                    f_stats.write(f"{formatted_key}: {value:.2f}\n")
                else:
                    f_stats.write(f"{formatted_key}: {value}\n")
        logging.info(f"Comprehensive tracking statistics saved to {stats_file_path}")
    except Exception as e_stats_save:
        logging.error(f"Error saving tracking statistics TXT file: {e_stats_save}\n{traceback.format_exc()}")


# --- MODIFICATION START: New helper functions for evaluation ---
def find_gt_directory(path_in, sequence_name, advanced_file_structure, params):
    """
    Attempts to find the corresponding Ground Truth (GT) directory for a given sequence.
    """
    logging.info(f"Attempting to find GT directory for sequence: {sequence_name}")

    gt_sequence_dir = None
    if advanced_file_structure:
        parent_dir = os.path.dirname(path_in)
        # Check for a CTC_GT_Data folder as a sibling to the sequence folder
        potential_gt_parent = os.path.join(os.path.dirname(parent_dir), "CTC_GT_Data")
        if not os.path.isdir(potential_gt_parent):
            potential_gt_parent = parent_dir  # Fallback to sequence's parent

        gt1 = os.path.join(potential_gt_parent, sequence_name + "_GT")
        gt2 = os.path.join(potential_gt_parent, sequence_name)

        logging.info(f"Advanced Mode: Checking for GT at '{gt1}' and '{gt2}'")
        for gt_path_option in [gt1, gt2]:
            tra_path = os.path.join(gt_path_option, "TRA")
            if os.path.isdir(tra_path) and os.path.exists(os.path.join(tra_path, "man_track.txt")):
                gt_sequence_dir = gt_path_option
                break
    else:  # Standard mode
        raw_folder_basename = basename(normpath(path_in))
        parent = os.path.dirname(path_in)
        dataset_name = basename(normpath(parent))
        grandparent = os.path.dirname(parent)

        options = [
            join(parent, raw_folder_basename + "_GT"),
            join(grandparent, dataset_name + "_GT", raw_folder_basename),
            join(grandparent, "CTC_GT_Data", dataset_name, raw_folder_basename),
            join(grandparent, "CTC_GT_Data", raw_folder_basename)
        ]
        logging.info("Standard Mode: Checking for GT in common locations.")
        for gt_path_option in options:
            tra_path = join(gt_path_option, "TRA")
            if os.path.isdir(tra_path) and os.path.exists(join(tra_path, "man_track.txt")):
                gt_sequence_dir = gt_path_option
                break

    if gt_sequence_dir and os.path.isdir(gt_sequence_dir):
        logging.info(f"Confirmed GT directory found: {gt_sequence_dir}")
        return gt_sequence_dir
    else:
        logging.warning("Could not find a valid GT directory.")
        return None


def save_ctc_metrics_to_txt(path_out_experiment, metrics, gt_sequence_dir, res_sequence_dir):
    """Saves the CTC evaluation metrics to a text file."""
    if not metrics:
        logging.warning("No CTC metrics to save.")
        return

    stats_file_path = join(path_out_experiment, 'ctc_evaluation_results.txt')
    logging.info(f"Saving CTC evaluation results to: {stats_file_path}")

    try:
        if "LNK" in metrics and "BIO(0)" in metrics:
            metrics["OP_CLB(0)"] = op_clb(metrics["LNK"], metrics["BIO(0)"])

        selected_keys = [
            "TRA", "SEG", "DET", "IDF1", "MOTA", "HOTA", "CHOTA", "LNK",
            "CT", "TF", "BC", "CCA", "MT", "ML",
            "AOGM", "AOGM_FN", "AOGM_EA", "AOGM_EC", "IDSW",
            "TP", "FN", "FP", "Precision", "Recall", "BIO(0)", "OP_CLB(0)"
        ]

        with open(stats_file_path, 'w', encoding='utf-8') as f:
            f.write("Cell Tracking Challenge (CTC) Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Result Directory: {basename(res_sequence_dir)}\n")
            f.write(f"Ground Truth Directory: {basename(gt_sequence_dir)}\n")
            f.write("-" * 50 + "\n\n")

            for key in selected_keys:
                if key in metrics:
                    val = metrics[key]
                    if isinstance(val, (float, np.floating)):
                        f.write(f"{key:<12}: {val:.4f}\n")
                    else:
                        f.write(f"{key:<12}: {val}\n")
        logging.info(f"Successfully saved CTC metrics to {stats_file_path}")
    except Exception as e:
        logging.error(f"Failed to save CTC metrics file: {e}\n{traceback.format_exc()}")


# --- MODIFICATION END ---


def save_settings_to_yaml(path_out, params_dict):
    """Save all settings (params) to a YAML file, including value, type, and description."""
    # Convert tuples to lists for YAML compatibility
    params_for_yaml = {k: list(v) if isinstance(v, tuple) else v for k, v in params_dict.items()}
    yaml_path = os.path.join(path_out, 'experiment_settings.yaml')
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f_yaml:
            yaml.dump(params_for_yaml, f_yaml, default_flow_style=False, allow_unicode=True)
        logging.info(f"Experiment settings saved to {yaml_path}")
    except Exception as e:
        logging.error(f"Error saving experiment settings YAML: {e}")


def save_results(path_in, trj, col_tuple, col_weights, id_masks, cell_ids, background_id,
                 color_list, cell_color_idx, cell_visibility, pixel_scale, pixel_unit,
                 show_ids, show_contours, show_tracks, mask_extension,
                 cell_lineage=None, ancestry_map=None,
                 use_thick_line=True, show_mitosis=False,
                 lineage_plot_widget=None, command_output_log=None,
                 main_app_state=None, ui_elements=None):
    
    # Create progress dialog for save operations
    progress_dialog = None
    if ui_elements and 'win' in ui_elements:
        from PyQt5.QtWidgets import QProgressDialog, QApplication
        from PyQt5.QtCore import Qt
        progress_dialog = QProgressDialog("Saving results...", "Cancel", 0, 100, ui_elements['win'])
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        progress_dialog.show()
        QApplication.processEvents()
    
    def update_progress(value, text):
        """Update progress dialog and keep UI responsive"""
        if progress_dialog:
            progress_dialog.setValue(value)
            progress_dialog.setLabelText(text)
            QApplication.processEvents()
            return not progress_dialog.wasCanceled()
        return True
    
    def check_timeout(start_time, timeout_seconds=300):
        """Check if operation has exceeded timeout"""
        if time.time() - start_time > timeout_seconds:
            logging.warning(f"Save operation exceeded {timeout_seconds} second timeout")
            return True
        return False
    
    start_time = time.time()
    timeout_seconds = 300  # 5 minutes timeout
    
    try:
        update_progress(5, "Preparing trajectory data...")
        if check_timeout(start_time, timeout_seconds):
            raise TimeoutError("Save operation timed out during trajectory preparation")
            
        trj_to_use = _enrich_trajectory_if_needed(trj, id_masks, main_app_state)
        if main_app_state:
            main_app_state['trj'] = trj_to_use

        col_tuple_to_use = main_app_state.get('col_tuple') if main_app_state else col_tuple

        save_time = datetime.now()
        root_path, folder_in_basename = split(normpath(path_in))

        tracking_mode = "Unknown"
        if main_app_state:
            params = main_app_state.get('params', {})
            tracking_mode = params.get('Tracking Mode', ['Unknown'])[0]

        experiment_folder_name = '{}_Exp_{}-{:02d}-{:02d}T{:02d}{:02d}{:02d}_{}'.format(
            folder_in_basename, save_time.year, save_time.month, save_time.day,
            save_time.hour, save_time.minute, save_time.second, tracking_mode)

        path_out_experiment = join(root_path, experiment_folder_name)
        if not exists(path_out_experiment): os.makedirs(path_out_experiment)

        update_progress(10, "Saving CSV data...")
        if check_timeout(start_time, timeout_seconds):
            raise TimeoutError("Save operation timed out during CSV saving")
            
        pixel_scale_microns = pixel_scale * 10 ** 6
        save_results_to_csv(path_out_experiment, trj_to_use, col_tuple_to_use, cell_visibility, pixel_scale_microns, main_app_state)

        update_progress(15, "Calculating statistics...")
        if check_timeout(start_time, timeout_seconds):
            raise TimeoutError("Save operation timed out during statistics calculation")
            
        _calculate_comprehensive_stats(main_app_state)
        calculated_stats_to_save = None
        legacy_counts_for_params_csv = None
        if main_app_state and 'calculated_stats' in main_app_state:
            calculated_stats_to_save = main_app_state['calculated_stats']
            if calculated_stats_to_save and "error" not in calculated_stats_to_save:
                legacy_counts_for_params_csv = {
                    'start': calculated_stats_to_save.get('cell_count_at_first_frame', 'N/A'),
                    'end': calculated_stats_to_save.get('cell_count_at_last_frame', 'N/A')
                }
                save_tracking_statistics(path_out_experiment, calculated_stats_to_save)

        update_progress(20, "Saving experiment parameters...")
        if check_timeout(start_time, timeout_seconds):
            raise TimeoutError("Save operation timed out during parameter saving")
            
        if (main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Experiment Parameters', (True,))[0]):
            save_experiment_parameters(path_out_experiment, pixel_scale_microns, "microns", col_weights,
                                       counts=legacy_counts_for_params_csv)

        if main_app_state and 'params' in main_app_state:
            save_settings_to_yaml(path_out_experiment, main_app_state['params'])

        if (cell_lineage is not None and trj_to_use is not None and not trj_to_use.empty and
            main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Lineage Relationships', (True,))[0]):
            save_lineage_info_to_csv(path_out_experiment, trj_to_use, cell_lineage)
    
        update_progress(25, "Saving cell editor table...")
        if check_timeout(start_time, timeout_seconds):
            raise TimeoutError("Save operation timed out during cell editor table saving")
            
        # Save complete cell editor table data
        logging.info(f"Checking cell editor table save conditions:")
        logging.info(f"  main_app_state exists: {main_app_state is not None}")
        logging.info(f"  'trj' in main_app_state: {'trj' in main_app_state if main_app_state else False}")
        logging.info(f"  trj is empty: {main_app_state['trj'].empty if main_app_state and 'trj' in main_app_state else True}")
        logging.info(f"  'params' in main_app_state: {'params' in main_app_state if main_app_state else False}")
        
        save_cell_editor_param = main_app_state['params'].get('Save Cell Editor Table', (True,))[0] if main_app_state and 'params' in main_app_state else True
        logging.info(f"  Save Cell Editor Table parameter: {save_cell_editor_param}")
        
        if (main_app_state and 'trj' in main_app_state and not main_app_state['trj'].empty and
            main_app_state and 'params' in main_app_state and 
            save_cell_editor_param):
            try:
                # Get all the data needed for the cell editor table
                trj = main_app_state['trj']
                cell_visibility = main_app_state.get('cell_visibility', {})
                ancestry = main_app_state.get('ancestry', {})
                cell_lineage = main_app_state.get('cell_lineage', {})
                track_states = main_app_state.get('track_states', {})
                
                # Create comprehensive cell editor table
                cell_editor_data = []
                unique_cell_ids = sorted(trj['particle'].unique())
                logging.info(f"Creating cell editor table for {len(unique_cell_ids)} unique cell IDs")
                logging.info(f"  Cell visibility data available: {len(cell_visibility)} entries")
                logging.info(f"  Ancestry data available: {len(ancestry)} entries")
                logging.info(f"  Cell lineage data available: {len(cell_lineage)} entries")
                logging.info(f"  Track states data available: {len(track_states)} entries")
                
                for cell_id in unique_cell_ids:
                    cell_id_int = int(cell_id)
                    
                    # Get visibility
                    visible = cell_visibility.get(cell_id_int, True)
                    
                    # Get parent IDs
                    parents = ancestry.get(cell_id_int, [])
                    parent_ids_str = ", ".join(map(str, sorted(parents))) if parents else ""
                    
                    # Get daughters
                    daughters = cell_lineage.get(cell_id_int, [])
                    daughters_str = ", ".join(map(str, sorted(daughters))) if daughters else ""
                    
                    # Get state
                    state = track_states.get(cell_id_int, "unknown")
                    
                    # Get original segmentation labels
                    cell_data = trj[trj['particle'] == cell_id]
                    if 'original_mask_label' in cell_data.columns:
                        orig_labels = cell_data['original_mask_label'].dropna().unique()
                        orig_labels_str = ", ".join(map(str, sorted(orig_labels))) if len(orig_labels) > 0 else "N/A"
                    else:
                        orig_labels_str = "N/A"
                    
                    # Get track statistics
                    track_frames = cell_data['frame'].unique()
                    start_frame = int(track_frames.min()) if len(track_frames) > 0 else 0
                    end_frame = int(track_frames.max()) if len(track_frames) > 0 else 0
                    track_length = len(track_frames)
                    
                    # Get displacement components from first to last frame
                    x_displacement = 0.0
                    y_displacement = 0.0
                    total_displacement = 0.0
                    if len(cell_data) > 1:
                        first_frame_data = cell_data[cell_data['frame'] == cell_data['frame'].min()]
                        last_frame_data = cell_data[cell_data['frame'] == cell_data['frame'].max()]
                        if not first_frame_data.empty and not last_frame_data.empty:
                            x1, y1 = first_frame_data.iloc[0]['x'], first_frame_data.iloc[0]['y']
                            x2, y2 = last_frame_data.iloc[0]['x'], last_frame_data.iloc[0]['y']
                            x_displacement = x2 - x1
                            y_displacement = y2 - y1
                            total_displacement = np.sqrt(x_displacement**2 + y_displacement**2)
                    
                    # Calculate frame-to-frame displacement statistics
                    x_displacement_std = 0.0
                    y_displacement_std = 0.0
                    if len(cell_data) > 1:
                        # Sort by frame to ensure chronological order
                        sorted_data = cell_data.sort_values('frame')
                        
                        # Calculate frame-to-frame displacements
                        x_diffs = sorted_data['x'].diff().dropna()
                        y_diffs = sorted_data['y'].diff().dropna()
                        
                        if len(x_diffs) > 0:
                            x_displacement_std = float(x_diffs.std())
                        if len(y_diffs) > 0:
                            y_displacement_std = float(y_diffs.std())
                    
                    # Get orientation angle statistics (in radians)
                    avg_angle_rad = 0.0
                    angle_std_rad = 0.0
                    # Check for both 'angle' and 'orientation' columns (from different tracking methods)
                    angle_column = None
                    if 'angle' in cell_data.columns:
                        angle_column = 'angle'
                    elif 'orientation' in cell_data.columns:
                        angle_column = 'orientation'
                        
                    if angle_column and len(cell_data) > 0:
                        # Filter out NaN values
                        valid_angles = cell_data[angle_column].dropna()
                        if len(valid_angles) > 0:
                            avg_angle_rad = float(valid_angles.mean())
                            angle_std_rad = float(valid_angles.std())
                    
                    cell_editor_data.append({
                        'cell_id': cell_id_int,
                        'visible': visible,
                        'parent_ids': parent_ids_str,
                        'daughters': daughters_str,
                        'state': state,
                        'original_segmentation_labels': orig_labels_str,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'track_length_frames': track_length,
                        'x_displacement': x_displacement,
                        'y_displacement': y_displacement,
                        'total_displacement': total_displacement,
                        'x_displacement_std': x_displacement_std,
                        'y_displacement_std': y_displacement_std,
                        'total_detections': len(cell_data),
                        'avg_orientation_rad': avg_angle_rad,
                        'orientation_std_rad': angle_std_rad
                    })
                
                # Save complete cell editor table (moved outside the loop)
                cell_editor_df = pd.DataFrame(cell_editor_data)
                
                # Format displacement values to 2 decimal places
                if 'x_displacement' in cell_editor_df.columns:
                    cell_editor_df['x_displacement'] = cell_editor_df['x_displacement'].round(2)
                if 'y_displacement' in cell_editor_df.columns:
                    cell_editor_df['y_displacement'] = cell_editor_df['y_displacement'].round(2)
                if 'total_displacement' in cell_editor_df.columns:
                    cell_editor_df['total_displacement'] = cell_editor_df['total_displacement'].round(2)
                
                # Format displacement standard deviation values to 2 decimal places
                if 'x_displacement_std' in cell_editor_df.columns:
                    cell_editor_df['x_displacement_std'] = cell_editor_df['x_displacement_std'].round(2)
                if 'y_displacement_std' in cell_editor_df.columns:
                    cell_editor_df['y_displacement_std'] = cell_editor_df['y_displacement_std'].round(2)
                
                # Format angle values to 3 decimal places for better precision
                if 'avg_orientation_rad' in cell_editor_df.columns:
                    cell_editor_df['avg_orientation_rad'] = cell_editor_df['avg_orientation_rad'].round(3)
                if 'orientation_std_rad' in cell_editor_df.columns:
                    cell_editor_df['orientation_std_rad'] = cell_editor_df['orientation_std_rad'].round(3)
                
                cell_editor_path = join(path_out_experiment, 'cell_editor_table.csv')
                cell_editor_df.to_csv(cell_editor_path, index=False, float_format='%.2f')
                logging.info(f"Complete cell editor table saved to {cell_editor_path}")
                logging.info(f"  Table contains {len(cell_editor_df)} rows and {len(cell_editor_df.columns)} columns")
                logging.info(f"  Columns: {list(cell_editor_df.columns)}")
                    
            except Exception as e:
                logging.error(f"Error saving cell editor table: {e}")
                logging.error(f"Exception details: {traceback.format_exc()}")
        else:
            logging.warning("Cell editor table save conditions not met - skipping save")
        
        update_progress(30, "Saving track states...")
        if check_timeout(start_time, timeout_seconds):
            raise TimeoutError("Save operation timed out during track states saving")
            
        # Save track states mapping for complete cell editor functionality
        if (main_app_state and 'track_states' in main_app_state and main_app_state['track_states'] and
            main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Track States', (True,))[0]):
            try:
                track_states_df = pd.DataFrame([
                    {'cell_id': cell_id, 'state': state} 
                    for cell_id, state in main_app_state['track_states'].items()
                ])
                track_states_path = join(path_out_experiment, 'track_states.csv')
                track_states_df.to_csv(track_states_path, index=False)
                logging.info(f"Track states mapping saved to {track_states_path}")
            except Exception as e:
                logging.error(f"Error saving track states mapping: {e}")

        update_progress(35, "Saving additional state data...")
        if check_timeout(start_time, timeout_seconds):
            raise TimeoutError("Save operation timed out during additional state data saving")
            
        # Save additional state data for complete functionality restoration
        if main_app_state:
            try:
                # Save cell visibility
                if ('cell_visibility' in main_app_state and main_app_state['cell_visibility'] and
                    main_app_state and 'params' in main_app_state and 
                    main_app_state['params'].get('Save Cell Visibility', (True,))[0]):
                    visibility_df = pd.DataFrame([
                        {'cell_id': cell_id, 'visible': visible} 
                        for cell_id, visible in main_app_state['cell_visibility'].items()
                    ])
                    visibility_path = join(path_out_experiment, 'cell_visibility.csv')
                    visibility_df.to_csv(visibility_path, index=False)
                    logging.info(f"Cell visibility saved to {visibility_path}")

                # Save cell frame presence
                if ('cell_frame_presence' in main_app_state and main_app_state['cell_frame_presence'] and
                    main_app_state and 'params' in main_app_state and 
                    main_app_state['params'].get('Save Cell Frame Presence', (True,))[0]):
                    frame_presence_data = []
                    for cell_id, frames in main_app_state['cell_frame_presence'].items():
                        for frame in frames:
                            frame_presence_data.append({'cell_id': cell_id, 'frame': frame})
                    if frame_presence_data:
                        frame_presence_df = pd.DataFrame(frame_presence_data)
                        frame_presence_path = join(path_out_experiment, 'cell_frame_presence.csv')
                        frame_presence_df.to_csv(frame_presence_path, index=False)
                        logging.info(f"Cell frame presence saved to {frame_presence_path}")

                # Save cell coordinates
                if ('cell_x' in main_app_state and 'cell_y' in main_app_state and
                    main_app_state and 'params' in main_app_state and 
                    main_app_state['params'].get('Save Cell Coordinates', (True,))[0]):
                    coord_data = []
                    for (cell_id, frame), x_val in main_app_state['cell_x'].items():
                        y_val = main_app_state['cell_y'].get((cell_id, frame))
                        if y_val is not None:
                            coord_data.append({'cell_id': cell_id, 'frame': frame, 'x': x_val, 'y': y_val})
                    if coord_data:
                        coord_df = pd.DataFrame(coord_data)
                        coord_path = join(path_out_experiment, 'cell_coordinates.csv')
                        coord_df.to_csv(coord_path, index=False)
                        logging.info(f"Cell coordinates saved to {coord_path}")

                    # Save raw images as PNG files for later loading
                    if ('raw_imgs' in main_app_state and main_app_state['raw_imgs'] is not None and
                        main_app_state and 'params' in main_app_state and 
                        main_app_state['params'].get('Save Raw Images as PNG', (True,))[0]):
                        raw_images_dir = join(path_out_experiment, 'raw_images')
                        os.makedirs(raw_images_dir, exist_ok=True)
                        raw_imgs = main_app_state['raw_imgs']
                        for i_frame in range(raw_imgs.shape[2]):
                            img_filename = join(raw_images_dir, f'raw_image_{i_frame:03d}.png')
                            # Normalize image to 0-255 range for PNG saving
                            img_frame = raw_imgs[:, :, i_frame]
                            if img_frame.max() > img_frame.min():
                                img_normalized = ((img_frame - img_frame.min()) / (img_frame.max() - img_frame.min()) * 255).astype(np.uint8)
                            else:
                                img_normalized = img_frame.astype(np.uint8)
                            iio.imwrite(img_filename, img_normalized)
                        logging.info(f"Raw images saved as PNG to {raw_images_dir}")

                    # Save merged masks if available
                    if ('merged_masks' in main_app_state and main_app_state['merged_masks'] is not None and
                        main_app_state and 'params' in main_app_state and 
                        main_app_state['params'].get('Save Merged Masks', (True,))[0]):
                        merged_masks_dir = join(path_out_experiment, 'merged_masks')
                        os.makedirs(merged_masks_dir, exist_ok=True)
                        merged_masks = main_app_state['merged_masks']
                        for i_frame in range(merged_masks.shape[2]):
                            mask_filename = join(merged_masks_dir, f'merged_mask_{i_frame:03d}.tif')
                            tifffile.imwrite(mask_filename, merged_masks[:, :, i_frame].astype(np.uint16))
                        logging.info(f"Merged masks saved to {merged_masks_dir}")

            except Exception as e:
                logging.error(f"Error saving additional state data: {e}")

        update_progress(40, "Saving tracking log...")
        if check_timeout(start_time, timeout_seconds):
            raise TimeoutError("Save operation timed out during tracking log saving")
            
        if (command_output_log and
            main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Tracking Log', (True,))[0]):
            log_file_path = join(path_out_experiment, 'tracking_operations_log.txt')
            try:
                with open(log_file_path, 'w', encoding='utf-8') as f_log:
                    f_log.write(command_output_log)
                logging.info(f"Tracking operations log saved to {log_file_path}")
            except Exception as e_log_save:
                logging.error(f"Error saving tracking operations log: {e_log_save}")
        else:
            logging.info("No command output log provided to save_results.")

        update_progress(45, "Saving ID masks...")
        if check_timeout(start_time, timeout_seconds):
            raise TimeoutError("Save operation timed out during ID mask saving")
            
        if id_masks is not None:
            logging.info(
                '================= Saving id masks frame by frame (raw values - original format) =================')
            
            # Filter mask data based on cell visibility
            filtered_id_masks = id_masks.copy()
            background_id_val = main_app_state.get('background_id', 0) if main_app_state else 0
            
            # Get visible cell IDs
            visible_cell_ids = set()
            if main_app_state and 'cell_visibility' in main_app_state:
                visible_cell_ids = {cell_id for cell_id, is_visible in main_app_state['cell_visibility'].items() 
                                  if is_visible and cell_id != background_id_val}
            
            # If there are invisible cells, filter them out from the mask
            if visible_cell_ids:
                # Get all unique cell IDs in the mask (excluding background)
                all_mask_ids = set(np.unique(id_masks))
                all_mask_ids.discard(background_id_val)
                
                # Find invisible cell IDs
                invisible_cell_ids = all_mask_ids - visible_cell_ids
                
                if invisible_cell_ids:
                    logging.info(f"Filtering out {len(invisible_cell_ids)} invisible cells from mask data: {sorted(invisible_cell_ids)}")
                    # Set invisible cells to background ID
                    for invisible_id in invisible_cell_ids:
                        filtered_id_masks[filtered_id_masks == invisible_id] = background_id_val
                else:
                    logging.info("All cells are visible, no filtering needed")
            else:
                logging.warning("No visible cells found, saving empty mask data")
                # If no cells are visible, set everything to background
                filtered_id_masks[filtered_id_masks != background_id_val] = background_id_val
            
            id_mask_frames_raw = [filtered_id_masks[:, :, i_frame] for i_frame in range(filtered_id_masks.shape[2])]
            save_sequence_frame_by_frame(id_mask_frames_raw, path_out_experiment, 'Id_masks_raw_per_frame_original',
                                         mask_extension, 'id_masks_raw_original')

        update_progress(50, "Creating ID mask animations...")
        if check_timeout(start_time, timeout_seconds):
            raise TimeoutError("Save operation timed out during ID mask animation creation")
            
        if (id_masks is not None and trj_to_use is not None and not trj_to_use.empty and color_list and cell_color_idx and
            main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save ID Mask Animations', (True,))[0]):
            logging.info(
                '================= Saving Raw ID Mask Animation (Colored by ID - original format) =================')
            # Use filtered mask data for animations
            mask_data_for_anim = filtered_id_masks if 'filtered_id_masks' in locals() else id_masks
            raw_id_mask_anim_frames = create_raw_id_mask_animation_frames(mask_data_for_anim, trj_to_use, color_list,
                                                                          cell_color_idx,
                                                                          background_id, cell_visibility, show_ids,
                                                                          cell_lineage=cell_lineage,
                                                                          show_mitosis=show_mitosis)
            if raw_id_mask_anim_frames:
                try:
                    gif_ready_raw_id_masks = [frame.astype(np.uint8) for frame in raw_id_mask_anim_frames]
                    iio.imwrite(join(path_out_experiment, 'id_mask_colored_animation_original.gif'),
                                gif_ready_raw_id_masks,
                                duration=0.2, loop=0, plugin='pillow')
                    logging.info("Successfully saved raw ID mask (colored) animation as GIF (original format).")
                except Exception as e_raw_id_gif:
                    logging.error(f"Error saving raw ID mask (colored) GIF animation (original format): {e_raw_id_gif}")

            # Additional save: Create ID mask animation WITHOUT text overlay (filled cells only)
            if id_masks is not None and trj_to_use is not None and not trj_to_use.empty and color_list and cell_color_idx:
                logging.info(
                    '================= Saving Additional Raw ID Mask Animation (No Text - Filled Cells Only) =================')
                # Use filtered mask data for animations
                mask_data_for_anim_no_text = filtered_id_masks if 'filtered_id_masks' in locals() else id_masks
                raw_id_mask_anim_frames_no_text = create_raw_id_mask_animation_frames(mask_data_for_anim_no_text, trj_to_use, color_list,
                                                                                      cell_color_idx,
                                                                                      background_id, cell_visibility, False,  # Always False for no text
                                                                                      cell_lineage=cell_lineage,
                                                                                      show_mitosis=show_mitosis)
                if raw_id_mask_anim_frames_no_text:
                    try:
                        gif_ready_raw_id_masks_no_text = [frame.astype(np.uint8) for frame in raw_id_mask_anim_frames_no_text]
                        iio.imwrite(join(path_out_experiment, 'id_mask_colored_animation_no_text.gif'),
                                    gif_ready_raw_id_masks_no_text,
                                    duration=0.2, loop=0, plugin='pillow')
                        logging.info("Successfully saved additional raw ID mask (colored, no text) animation as GIF.")
                    except Exception as e_raw_id_gif_no_text:
                        logging.error(f"Error saving additional raw ID mask (colored, no text) GIF animation: {e_raw_id_gif_no_text}")

        update_progress(55, "Creating colorized masks...")
        if check_timeout(start_time, timeout_seconds):
            raise TimeoutError("Save operation timed out during colorized mask creation")
            
        if (id_masks is not None and trj_to_use is not None and not trj_to_use.empty and
            main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Colorized Mask Animations', (True,))[0]):
            # Use filtered mask data for colorized masks
            mask_data_for_colorized = filtered_id_masks if 'filtered_id_masks' in locals() else id_masks
            colorized_masks = create_colorized_masks(mask_data_for_colorized, trj_to_use, cell_ids, background_id, color_list,
                                                     cell_color_idx,
                                                     cell_visibility, show_ids, cell_lineage=cell_lineage,
                                                     show_mitosis=show_mitosis)
            logging.info(
                '================= Saving colorized masks (segmentation) frame by frame - original format =================')
            save_sequence_frame_by_frame(colorized_masks, path_out_experiment, 'Masks_segmented_per_frame_original',
                                         mask_extension, 'mask')
            logging.info(
                '================= Saving colorized masks (segmentation) as animation - original format =================')
            try:
                gif_ready_masks = [frame.astype(np.uint8) for frame in colorized_masks]
                iio.imwrite(join(path_out_experiment, 'masks_segmented_animation_original.gif'), gif_ready_masks,
                            duration=0.2, loop=0, plugin='pillow')
                logging.info("Successfully saved segmented masks animation as GIF (original format)")
            except Exception as e2:
                logging.error(f"Error saving segmented masks GIF animation (original format): {str(e2)}")

        update_progress(60, "Creating colorized tracks...")
        if check_timeout(start_time, timeout_seconds):
            raise TimeoutError("Save operation timed out during colorized track creation")
            
        if (id_masks is not None and trj_to_use is not None and not trj_to_use.empty and
            main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Track Animations', (True,))[0]):
            # Use filtered mask data for colorized tracks
            mask_data_for_tracks = filtered_id_masks if 'filtered_id_masks' in locals() else id_masks
            colorized_tracks = create_colorized_tracks(mask_data_for_tracks, trj_to_use, cell_ids, background_id, color_list,
                                                       cell_color_idx,
                                                       cell_visibility, show_ids, show_contours, show_tracks,
                                                       use_thick_line, cell_lineage=cell_lineage,
                                                       show_mitosis=show_mitosis)
            logging.info('================= Saving colorized tracks frame by frame - original format =================')
            save_sequence_frame_by_frame(colorized_tracks, path_out_experiment, 'Tracks_per_frame_original',
                                         mask_extension,
                                         'tracks_original')
            logging.info('================= Saving colorized tracks as animation - original format =================')
            try:
                gif_ready_tracks = [frame.astype(np.uint8) for frame in colorized_tracks]
                iio.imwrite(join(path_out_experiment, 'tracks_animation_original.gif'), gif_ready_tracks, duration=0.2,
                            loop=0, plugin='pillow')
                logging.info("Successfully saved tracks animation as GIF (original format)")
            except Exception as e2:
                logging.error(f"Error saving tracks GIF animation (original format): {str(e2)}")

        update_progress(60, "Saving lineage plots...")
        if lineage_plot_widget is not None:
            if (lineage_plot_widget is not None and 
                main_app_state and 'params' in main_app_state and 
                main_app_state['params'].get('Save Lineage Plots', (True,))[0]):
                logging.info('================= Saving Lineage Plots =================')
                
                # Save current view type plot
                current_view_type = lineage_plot_widget.current_view_type
                plot_save_path = join(path_out_experiment,
                                      f'lineage_plot_{current_view_type.replace(" ", "_")}.png')
                try:
                    if hasattr(lineage_plot_widget, 'export_current_plot_as_image'):
                        # Use high-resolution export for better large screen display
                        lineage_plot_widget.export_current_plot_as_image(plot_save_path, high_resolution=True)
                        logging.info(f"Saved high-resolution lineage plot ({current_view_type}) to: {plot_save_path}")
                    else:
                        logging.warning("Lineage plot widget lacks 'export_current_plot_as_image' method.")
                except Exception as e_plot:
                    logging.error(f"Error saving current lineage plot: {e_plot}\n{traceback.format_exc()}")
                
                # Automatically save both line view and class type plots
                view_types_to_save = ['Track Segments', 'Class Type']
                for view_type in view_types_to_save:
                    if view_type != current_view_type:  # Don't save the same view twice
                        try:
                            # Temporarily switch to the other view type
                            original_view_type = lineage_plot_widget.current_view_type
                            lineage_plot_widget.current_view_type = view_type
                            
                            # Redraw the plot for the new view type
                            lineage_plot_widget.draw_all_lineage_trees()
                            
                            # Save the plot with high resolution
                            plot_save_path = join(path_out_experiment,
                                                  f'lineage_plot_{view_type.replace(" ", "_")}.png')
                            if hasattr(lineage_plot_widget, 'export_current_plot_as_image'):
                                lineage_plot_widget.export_current_plot_as_image(plot_save_path, high_resolution=True)
                                logging.info(f"Saved high-resolution lineage plot ({view_type}) to: {plot_save_path}")
                            
                            # Restore original view type
                            lineage_plot_widget.current_view_type = original_view_type
                            lineage_plot_widget.draw_all_lineage_trees()
                            
                        except Exception as e_plot:
                            logging.error(f"Error saving lineage plot ({view_type}): {e_plot}\n{traceback.format_exc()}")
                            # Try to restore original view type even if saving failed
                            try:
                                lineage_plot_widget.current_view_type = original_view_type
                                lineage_plot_widget.draw_all_lineage_trees()
                            except:
                                pass

        update_progress(70, "Creating track overview...")
        if (id_masks is not None and trj_to_use is not None and not trj_to_use.empty and
            main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Track Overview', (True,))[0]):
            logging.info('================= Saving track overview - original format =================')
            # Use filtered mask data for track overview
            mask_data_for_overview = filtered_id_masks if 'filtered_id_masks' in locals() else id_masks
            all_tracks_img = create_track_overview(mask_data_for_overview, trj_to_use, cell_ids, background_id, color_list,
                                                   cell_color_idx,
                                                   cell_visibility, show_ids, use_thick_line, cell_lineage=cell_lineage,
                                                   show_mitosis=show_mitosis)
            iio.imwrite(join(path_out_experiment, f'all_tracks_overview_original.{mask_extension}'),
                        all_tracks_img.astype(np.uint8))

        update_progress(75, "Saving CTC formatted results...")
        # --- MODIFICATION START: Integrated CTC evaluation ---
        # Use original path_in from main_app_state if available (for loaded data)
        original_path_in = main_app_state.get('path_in') if main_app_state else path_in
        
        # Extract sequence name from the original data path, not the experiment path
        if original_path_in and original_path_in != path_in:
            # We're saving from loaded data, use the original sequence name
            original_root_path, original_folder_name = split(normpath(original_path_in))
            sequence_name_for_res = original_folder_name
            logging.info(f"Using original sequence name from loaded data: {sequence_name_for_res}")
            logging.info(f"Original path: {original_path_in}")
            logging.info(f"Original folder name: {original_folder_name}")
        else:
            # We're saving from fresh data, use current folder name
            sequence_name_for_res = folder_in_basename
            logging.info(f"Using current folder name for sequence: {sequence_name_for_res}")
            logging.info(f"Current path: {path_in}")
            logging.info(f"Current folder name: {folder_in_basename}")
            
        res_sequence_dir = join(path_out_experiment, f"{sequence_name_for_res}_RES")
        
        # Use filtered mask data for CTC formatted results to respect cell visibility
        filtered_id_masks_for_ctc = filtered_id_masks if 'filtered_id_masks' in locals() else id_masks
        
        # Handle missing cells to ensure CTC format consistency
        handle_missing_cells = True
        if main_app_state and 'params' in main_app_state:
            handle_missing_cells = main_app_state['params'].get('Handle Missing Cells in CTC', (True,))[0]
        
        if handle_missing_cells:
            cleaned_trj, cleaned_masks, removed_tracks = handle_missing_cells_in_ctc_output(
                trj_to_use, filtered_id_masks_for_ctc, background_id
            )
            
            if removed_tracks:
                logging.warning(f"Removed {len(removed_tracks)} tracks with missing mask pixels for CTC evaluation")
                logging.info(f"Removed track IDs: {removed_tracks}")
        else:
            cleaned_trj, cleaned_masks, removed_tracks = trj_to_use, filtered_id_masks_for_ctc, []
        
        save_ctc_res_formatted_results(res_sequence_dir, cleaned_trj, cleaned_masks, main_app_state, ui_elements)

        # Check if user wants to run CTC metrics evaluation
        run_ctc_metrics = True
        if main_app_state and 'params' in main_app_state:
            run_ctc_metrics = main_app_state['params'].get('Run CTC Metrics Evaluation', (True,))[0]

        if run_ctc_metrics:
            update_progress(80, "Running CTC evaluation...")
            if check_timeout(start_time, timeout_seconds):
                raise TimeoutError("Save operation timed out during CTC evaluation")
                
            logging.info("================= Automatic CTC Evaluation on Save ==================")
            logging.info(f"Sequence name for GT search: '{sequence_name_for_res}'")
            logging.info(f"Current path_in: '{path_in}'")
            logging.info(f"Original path_in from main_app_state: '{main_app_state.get('path_in') if main_app_state else 'None'}'")
            advanced_mode = main_app_state.get('params', {}).get('Enable Advanced File Structure', (False,))[0]
            # Use original_path_in for GT directory search, not the experiment path
            search_path = original_path_in if original_path_in else path_in
            logging.info(f"Searching for GT directory using path: '{search_path}'")
            gt_dir = find_gt_directory(search_path, sequence_name_for_res, advanced_mode, main_app_state.get('params'))

            if gt_dir:
                logging.info(f"Found GT directory: {gt_dir}. Proceeding with evaluation.")
                metrics_to_run = ['TRA', 'SEG', 'DET', 'IDF1', 'MOTA', 'HOTA', 'LNK', 'CHOTA', 'CT', 'TF', 'BC', 'CCA', 'MT',
                                  'ML']
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    eval_results = run_ctc_evaluation_api(
                        gt_sequence_dir=gt_dir,
                        res_sequence_dir=res_sequence_dir,
                        metrics_list=metrics_to_run,
                        num_threads=0
                    )
                    if eval_results and eval_results.get("status") == "success":
                        metrics = eval_results.get("metrics", {})
                        main_app_state['ctc_eval_results'] = metrics
                        save_ctc_metrics_to_txt(path_out_experiment, metrics, gt_dir, res_sequence_dir)
                    else:
                        msg = eval_results.get("message", "Unknown error") if eval_results else "No results returned"
                        with open(join(path_out_experiment, 'ctc_evaluation_failed.log'), 'w') as f_err:
                            f_err.write(
                                f"CTC evaluation failed.\nGT Path: {gt_dir}\nRES Path: {res_sequence_dir}\n\nError: {msg}")
                except Exception as e:
                    logging.error(f"An exception occurred during automatic CTC evaluation: {e}\n{traceback.format_exc()}")
                    with open(join(path_out_experiment, 'ctc_evaluation_exception.log'), 'w') as f_exc:
                        f_exc.write(f"An exception occurred during automatic CTC evaluation:\n{traceback.format_exc()}")
                finally:
                    QApplication.restoreOverrideCursor()
            else:
                logging.info("No corresponding GT directory found. Skipping CTC evaluation.")
        else:
            logging.info("User disabled CTC metrics evaluation. Skipping CTC evaluation.")

        update_progress(85, "Saving analysis plots...")
        # Save temporal outline stack image
        try:
            if main_app_state and 'params' in main_app_state and main_app_state['params'].get('Save Temporal Outline Stack', (True,))[0]:
                from cell_drawing import save_temporal_outline_stack
                colormap_name = 'plasma'
                if main_app_state and 'params' in main_app_state:
                    colormap_name = main_app_state['params'].get('Temporal Outline Colormap', ('plasma',))[0]
                # Create analysis_plots directory
                analysis_dir = os.path.join(path_out_experiment, 'analysis_plots')
                os.makedirs(analysis_dir, exist_ok=True)
                outline_stack_path = join(analysis_dir, 'temporal_outline_stack.png')
                # Get resolution parameters
                dpi = 300
                scale_factor = 2
                fast_mode = False
                if main_app_state and 'params' in main_app_state:
                    dpi = main_app_state['params'].get('Figure DPI', (300,))[0]
                    scale_factor = main_app_state['params'].get('Temporal Outline Scale Factor', (2,))[0]
                    fast_mode = main_app_state['params'].get('Temporal Outline Fast Mode', (False,))[0]
                
                # Determine figsize_scale based on fast_mode (consistent with other plotting functions)
                figsize_scale = 0.7 if fast_mode else 1.0
                
                # Use filtered mask data for temporal outline stack
                mask_data_for_outline = filtered_id_masks if 'filtered_id_masks' in locals() else id_masks
                save_temporal_outline_stack(mask_data_for_outline, outline_stack_path, colormap_name=colormap_name, alpha=0.15, dpi=dpi, scale_factor=scale_factor, fast_mode=fast_mode, figsize_scale=figsize_scale)
                logging.info(f"Saved temporal outline stack image to {outline_stack_path}")
        except Exception as e:
            logging.error(f"Error saving temporal outline stack image: {e}")

        # Save state masks and color masks if available
        try:
            if (main_app_state and 'params' in main_app_state and 
                main_app_state['params'].get('Save State Masks and Colors', (True,))[0] and
                id_masks is not None and trj_to_use is not None and 'state' in trj_to_use.columns):
                from cell_drawing import save_state_masks_and_colors
                # Use filtered mask data for state masks
                mask_data_for_state = filtered_id_masks if 'filtered_id_masks' in locals() else id_masks
                save_state_masks_and_colors(mask_data_for_state, trj_to_use, path_out_experiment)
                logging.info(f"Saved state masks and color masks to {path_out_experiment}")
        except Exception as e:
            logging.error(f"Error saving state masks and color masks: {e}")

        # Save offset track plot if enabled
        try:
            if main_app_state and 'params' in main_app_state and main_app_state['params'].get('Offset Track Plot', (True,))[0]:
                from cell_drawing import save_offset_track_plot
                left_color = main_app_state['params'].get('Offset Track Plot Left Color', ('black',))[0]
                right_color = main_app_state['params'].get('Offset Track Plot Right Color', ('red',))[0]
                output_filename = main_app_state['params'].get('Offset Track Plot Output Filename', ('offset_track_plot.png',))[0]
                # Create analysis_plots directory
                analysis_dir = os.path.join(path_out_experiment, 'analysis_plots')
                os.makedirs(analysis_dir, exist_ok=True)
                output_path = join(analysis_dir, output_filename)
                save_offset_track_plot(trj_to_use, output_path, left_color=left_color, right_color=right_color)
                logging.info(f"Saved offset track plot to {output_path}")
        except Exception as e:
            logging.error(f"Error saving offset track plot: {e}")

        # Save time-based offset track plot if enabled
        try:
            if main_app_state and 'params' in main_app_state and main_app_state['params'].get('Time-based Offset Track Plot', (True,))[0]:
                from cell_drawing import save_time_based_offset_track_plot
                colormap_name = main_app_state['params'].get('Time-based Offset Track Plot Colormap', ('plasma',))[0]
                output_filename = main_app_state['params'].get('Time-based Offset Track Plot Output Filename', ('time_based_offset_track_plot.png',))[0]
                # Create analysis_plots directory
                analysis_dir = os.path.join(path_out_experiment, 'analysis_plots')
                os.makedirs(analysis_dir, exist_ok=True)
                output_path = join(analysis_dir, output_filename)
                dpi = main_app_state['params'].get('Figure DPI', (300,))[0]
                save_time_based_offset_track_plot(trj_to_use, output_path, colormap_name=colormap_name, dpi=dpi)
                logging.info(f"Saved time-based offset track plot to {output_path}")
        except Exception as e:
            logging.error(f"Error saving time-based offset track plot: {e}")

        update_progress(90, "Saving GIF overlays...")
        # Save GIF overlays if enabled
        try:
            if main_app_state and 'params' in main_app_state:
                # Check if raw images are available
                raw_images = main_app_state.get('raw_imgs')
                if raw_images is not None and id_masks is not None:
                    from cell_drawing import create_gif_overlay_id_based, create_gif_overlay_state_based
                    
                    # Create gif_overlays directory
                    gif_dir = os.path.join(path_out_experiment, 'gif_overlays')
                    os.makedirs(gif_dir, exist_ok=True)
                    
                    # Use filtered mask data for GIF overlays
                    mask_data_for_gif = filtered_id_masks if 'filtered_id_masks' in locals() else id_masks
                    
                    # ID-based GIF overlay
                    if main_app_state['params'].get('Save ID-based GIF Overlay', (True,))[0]:
                        alpha = main_app_state['params'].get('GIF Overlay Alpha', (0.6,))[0]
                        fps = main_app_state['params'].get('GIF Overlay FPS', (10,))[0]
                        id_gif_path = join(gif_dir, 'id_based_overlay.gif')
                        create_gif_overlay_id_based(raw_images, mask_data_for_gif, trj_to_use, id_gif_path, alpha=alpha, fps=fps, cell_visibility=cell_visibility)
                        logging.info(f"Saved ID-based GIF overlay to {id_gif_path}")
                    
                    # State-based GIF overlay (always create, using "unknown" for missing states)
                    if main_app_state['params'].get('Save State-based GIF Overlay', (True,))[0]:
                        alpha = main_app_state['params'].get('GIF Overlay Alpha', (0.6,))[0]
                        fps = main_app_state['params'].get('GIF Overlay FPS', (10,))[0]
                        state_gif_path = join(gif_dir, 'state_based_overlay.gif')
                        create_gif_overlay_state_based(raw_images, mask_data_for_gif, trj_to_use, state_gif_path, alpha=alpha, fps=fps, cell_visibility=cell_visibility)
                        logging.info(f"Saved state-based GIF overlay to {state_gif_path}")
                else:
                    logging.warning("Raw images not available for GIF overlay creation")
        except Exception as e:
            logging.error(f"Error saving GIF overlays: {e}")

        update_progress(92, "Saving 16-bit class masks...")
        # Save 16-bit class masks if enabled
        try:
            if (main_app_state and 'params' in main_app_state and 
                main_app_state['params'].get('Save 16-bit Class Masks', (True,))[0] and
                trj_to_use is not None):
                from cell_drawing import save_16bit_class_masks
                class_masks_dir = os.path.join(path_out_experiment, 'class_masks_16bit')
                mask_prefix = main_app_state['params'].get('Class Mask Prefix', ('class_mask',))[0]
                # Use filtered mask data for class masks
                mask_data_for_class = filtered_id_masks if 'filtered_id_masks' in locals() else id_masks
                save_16bit_class_masks(mask_data_for_class, trj_to_use, class_masks_dir, mask_prefix=mask_prefix)
                logging.info(f"Saved 16-bit class masks to {class_masks_dir}")
        except Exception as e:
            logging.error(f"Error saving 16-bit class masks: {e}")

        update_progress(95, "Saving track type data...")
        # After all main results are saved, save track type CSVs and tree plots
        try:
            if (main_app_state and 'params' in main_app_state and 
                main_app_state['params'].get('Save Track Type CSVs and Plots', (True,))[0] and
                trj is not None and not trj.empty and cell_lineage and ancestry_map):
                track_types_dir = os.path.join(path_out_experiment, 'track_types')
                save_singular_tracks(trj, cell_lineage, ancestry_map, track_types_dir)
                save_mitosis_tracks(trj, cell_lineage, track_types_dir)
                save_fusion_tracks(trj, ancestry_map, track_types_dir)
                logging.info(f"Saved singular, mitosis, and fusion track CSVs and tree plots to {track_types_dir}")
        except Exception as e:
            logging.error(f"Error saving track type CSVs and tree plots: {e}")

        update_progress(98, "Creating comprehensive analysis plots...")
        # Create comprehensive analysis plots
        if (trj_to_use is not None and not trj_to_use.empty and id_masks is not None and
            main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Comprehensive Analysis Plots', (True,))[0]):
            from cell_drawing import create_comprehensive_analysis_plots
            # Use filtered mask data for comprehensive analysis plots
            mask_data_for_analysis = filtered_id_masks if 'filtered_id_masks' in locals() else id_masks
            
            # Get fast mode setting for analysis plots
            fast_mode = False
            if main_app_state and 'params' in main_app_state:
                fast_mode = main_app_state['params'].get('Analysis Plots Fast Mode', (True,))[0]
            
            create_comprehensive_analysis_plots(trj_to_use, cell_lineage, ancestry_map, mask_data_for_analysis, path_out_experiment, main_app_state, fast_mode=fast_mode)

        update_progress(100, "Save completed!")
        logging.info(f'Saving finished. Results in: {path_out_experiment}')
        
    except TimeoutError as e:
        logging.error(f"Save operation timed out: {e}")
        if progress_dialog:
            progress_dialog.close()
        # Show error message to user
        if ui_elements and 'win' in ui_elements:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(ui_elements['win'], "Save Timeout", 
                               f"Save operation timed out after {timeout_seconds} seconds. Some files may have been saved.\n\nPath: {path_out_experiment}")
        raise
    except Exception as e:
        logging.error(f"Error during save operation: {e}")
        if progress_dialog:
            progress_dialog.close()
        # Show error message to user
        if ui_elements and 'win' in ui_elements:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(ui_elements['win'], "Save Error", 
                               f"An error occurred during saving:\n{str(e)}\n\nSome files may have been saved.\n\nPath: {path_out_experiment}")
        raise
    finally:
        if progress_dialog:
            progress_dialog.close()


def save_results_to_csv(path_out, trj, col_tuple, cell_visibility, pixel_scale, main_app_state=None):
    """Saves the enriched trajectory data to a CSV file with a standardized column order."""
    if trj is None or trj.empty:
        logging.warning("CSV Save: Trajectory is empty. Nothing to save.");
        return
    if 'particle' not in trj.columns or 'frame' not in trj.columns:
        logging.error("CSV Save: 'particle' or 'frame' column missing in trajectory.");
        return

    # Define the standard, ordered list of columns for the output CSV.
    # Include state and original_mask_label for complete cell editor functionality
    standard_cols = [
        'frame', 'particle', 'x', 'y', 'equivalent_diameter', 'area',
        'perimeter', 'eccentricity', 'major_axis_length', 'minor_axis_length',
        'orientation', 'solidity', 'state', 'original_mask_label'
    ]

    trj_copy_for_csv = trj.copy()

    # Ensure all standard columns exist, add as NaN if they are still missing after enrichment
    for col in standard_cols:
        if col not in trj_copy_for_csv.columns:
            logging.warning(f"Standard column '{col}' not found in trajectory data. Adding it as empty column.")
            if col == 'state':
                # Try to get state information from main_app_state if available
                if main_app_state and 'track_states' in main_app_state:
                    track_states = main_app_state['track_states']
                    trj_copy_for_csv[col] = trj_copy_for_csv['particle'].map(track_states).fillna('unknown')
                    logging.info(f"Added state column from track_states mapping")
                else:
                    trj_copy_for_csv[col] = 'unknown'
            elif col == 'original_mask_label':
                # Try to get original mask labels if available
                if 'original_mask_label' in trj_copy_for_csv.columns:
                    # Column already exists, keep it
                    pass
                else:
                    trj_copy_for_csv[col] = 0  # Default to 0 for original mask labels
                    logging.info(f"Added original_mask_label column with default value 0")
            else:
                trj_copy_for_csv[col] = np.nan

    # Use the standardized list for saving
    cols_to_save = standard_cols

    # --- Geometric Scaling ---
    scaled_trj = trj_copy_for_csv.copy()
    geometric_cols_to_scale = {
        'y': pixel_scale, 'x': pixel_scale, 'equivalent_diameter': pixel_scale,
        'perimeter': pixel_scale, 'area': pixel_scale * pixel_scale,
        'major_axis_length': pixel_scale, 'minor_axis_length': pixel_scale
    }
    for col_name, scale_val in geometric_cols_to_scale.items():
        if col_name in scaled_trj.columns:
            # Ensure the column is numeric before scaling
            scaled_trj[col_name] = pd.to_numeric(scaled_trj[col_name], errors='coerce') * scale_val
        else:
            logging.warning(f"Cannot scale column '{col_name}' for CSV output: column not found.")

    # --- Visibility Filtering ---
    visible_ids = {int(pd.to_numeric(cid, errors='coerce')) for cid, vis in cell_visibility.items() if
                   vis and pd.notna(pd.to_numeric(cid, errors='coerce'))}
    if not visible_ids:
        logging.warning("No cells are marked as visible. The tracks.csv file will be empty.")
        scaled_trj_filtered = pd.DataFrame(columns=cols_to_save)
    else:
        scaled_trj['particle'] = pd.to_numeric(scaled_trj['particle'], errors='coerce')
        scaled_trj_filtered = scaled_trj[scaled_trj['particle'].dropna().astype(int).isin(visible_ids)]

    if scaled_trj_filtered.empty:
        logging.warning("No visible/valid tracks remain after filtering for CSV saving.");

    # --- Save to CSV ---
    try:
        # Sort by track ID, then frame for readability
        scaled_trj_filtered = scaled_trj_filtered.sort_values(by=['particle', 'frame'])
        # Save using the standardized column list
        scaled_trj_filtered.to_csv(join(path_out, 'tracks.csv'),
                                   columns=cols_to_save,
                                   float_format='%.3f',
                                   index=False)
        logging.info(f"Tracks CSV saved to {join(path_out, 'tracks.csv')} with standard columns.")
    except Exception as e_csv:
        logging.error(f"Error writing tracks CSV: {e_csv}")
        logging.error(traceback.format_exc())


def save_experiment_parameters(path_out, pixel_scale, pixel_unit, col_weights, counts=None):
    out_lines = [f'Parameter,Value\n', f'pixel_scale_microns,{pixel_scale}\n', f'pixel_unit,{pixel_unit}\n']
    for param_name, param_value in col_weights.items(): out_lines.append(f'{param_name},{param_value}\n')
    if counts:
        out_lines.append(f'cell_count_start_frame_legacy,{counts.get("start", "N/A")}\n')
        out_lines.append(f'cell_count_end_frame_legacy,{counts.get("end", "N/A")}\n')
    try:
        with open(join(path_out, 'experiment_parameters.csv'), 'w', encoding='utf-8') as f_out:
            f_out.writelines(out_lines)
        logging.info(f"Experiment parameters saved to {join(path_out, 'experiment_parameters.csv')}")
    except Exception as e:
        logging.error(f"Error saving experiment parameters: {e}")


def save_sequence_frame_by_frame(sequence, path_out, sequence_folder, file_extension, file_prefix):
    path_out_sequence = join(path_out, sequence_folder)
    if not exists(path_out_sequence): os.makedirs(path_out_sequence)
    if not sequence: logging.warning(f"Empty sequence for {file_prefix}. Nothing to save."); return
    if not isinstance(sequence, list) or not all(isinstance(f, np.ndarray) for f in sequence):
        logging.error(f"Sequence for {file_prefix} not list of NumPy arrays. Skipping.");
        return
    n_frames = len(sequence)
    if n_frames == 0: logging.warning(f"Empty sequence for {file_prefix} after validation."); return
    max_n_digits = int(np.floor(np.log10(n_frames - 1))) + 1 if n_frames > 1 else 1
    for i_frame, frame_data in enumerate(sequence):
        frame_name = f'{file_prefix}_{str(i_frame).zfill(max_n_digits)}.{file_extension}'
        current_frame_to_save = frame_data
        if not isinstance(frame_data, np.ndarray):
            logging.error(f"Frame {i_frame} of {file_prefix} not np.ndarray. Type: {type(frame_data)}. Skipping.");
            continue
        if file_prefix == 'id_masks_raw_original':
            if frame_data.ndim != 2: logging.error(
                f"Frame {i_frame} of id_masks_raw_original not 2D. Shape: {frame_data.shape}. Skip."); continue
            clipped_data = np.clip(frame_data, 0, None)
            max_val = np.max(clipped_data) if clipped_data.size > 0 else 0
            if file_extension.lower() in ['tif', 'tiff']:
                if max_val < 65536:
                    current_frame_to_save = clipped_data.astype(np.uint16)
                else:
                    current_frame_to_save = clipped_data.astype(np.uint32)
                    logging.warning(
                        f"Raw ID mask frame {i_frame} max value {max_val} exceeds uint16. Saved as uint32 for {file_extension}.")
            elif file_extension.lower() in ['png', 'gif', 'jpg', 'jpeg', 'bmp']:
                if max_val == 0 and clipped_data.size > 0:
                    current_frame_to_save = clipped_data.astype(np.uint8)
                elif max_val < 256:
                    current_frame_to_save = clipped_data.astype(np.uint8)
                elif max_val < 65536 and file_extension.lower() == 'png':
                    current_frame_to_save = clipped_data.astype(np.uint16)
                else:
                    logging.warning(
                        f"Raw ID mask frame {i_frame} max val {max_val} for {file_extension}. Saving as rescaled uint8 for compatibility.");
                    if max_val > 0:
                        current_frame_to_save = (clipped_data / max_val * 255).astype(np.uint8)
                    else:
                        current_frame_to_save = clipped_data.astype(np.uint8)
        elif current_frame_to_save.dtype != np.uint8 and file_extension.lower() in ['png', 'gif', 'jpg', 'jpeg', 'bmp']:
            if np.issubdtype(current_frame_to_save.dtype, np.floating):
                min_v, max_v = np.min(current_frame_to_save), np.max(current_frame_to_save)
                current_frame_to_save = np.zeros_like(current_frame_to_save, dtype=np.uint8) if max_v == min_v else (
                            (current_frame_to_save - min_v) / (max_v - min_v) * 255).astype(np.uint8)
            elif np.issubdtype(current_frame_to_save.dtype, np.integer):
                current_frame_to_save = np.clip(current_frame_to_save, 0, 255).astype(np.uint8)
        if not current_frame_to_save.flags.c_contiguous: current_frame_to_save = np.ascontiguousarray(
            current_frame_to_save)
        try:
            iio.imwrite(join(path_out_sequence, frame_name), current_frame_to_save)
        except Exception as e:
            logging.error(f"Error saving frame {i_frame} ({file_prefix}, {file_extension}): {e}")
            logging.error(
                f"  Data type: {current_frame_to_save.dtype}, Shape: {current_frame_to_save.shape}, Min: {np.min(current_frame_to_save) if current_frame_to_save.size > 0 else 'N/A'}, Max: {np.max(current_frame_to_save) if current_frame_to_save.size > 0 else 'N/A'}")
    logging.debug(f"Finished saving sequence for {file_prefix} in {sequence_folder}")


def handle_missing_cells_in_ctc_output(trj, id_masks, background_id=0):
    """
    Handle cases where tracked cells don't appear in mask files.
    This ensures CTC format consistency by either:
    1. Removing tracks that don't have corresponding mask pixels
    2. Or adding placeholder pixels for missing cells
    
    Returns:
        tuple: (cleaned_trj, cleaned_id_masks, removed_tracks)
    """
    if trj is None or trj.empty or id_masks is None:
        return trj, id_masks, []
    
    logging.info("Checking for missing cells in CTC output...")
    removed_tracks = []
    cleaned_trj = trj.copy()
    
    # Get all unique track IDs
    track_ids = set(pd.to_numeric(trj['particle'], errors='coerce').dropna().astype(int))
    track_ids.discard(background_id)
    
    for track_id in track_ids:
        track_frames = trj[trj['particle'] == track_id]['frame'].values
        missing_frames = []
        
        for frame in track_frames:
            if 0 <= frame < id_masks.shape[2]:
                mask_slice = id_masks[:, :, frame]
                if track_id not in np.unique(mask_slice):
                    missing_frames.append(frame)
        
        if missing_frames:
            logging.warning(f"Track {track_id} missing in frames: {missing_frames}")
            # Option 1: Remove the entire track (recommended for CTC evaluation)
            cleaned_trj = cleaned_trj[cleaned_trj['particle'] != track_id]
            removed_tracks.append(track_id)
            
            # Option 2: Add placeholder pixels (alternative approach)
            # for frame in missing_frames:
            #     # Find a background pixel and assign it to the track
            #     background_pixels = np.where(mask_slice == background_id)
            #     if len(background_pixels[0]) > 0:
            #         idx = 0  # Use first background pixel
            #         id_masks[background_pixels[0][idx], background_pixels[1][idx], frame] = track_id
    
    if removed_tracks:
        logging.info(f"Removed {len(removed_tracks)} tracks with missing mask pixels: {removed_tracks}")
    else:
        logging.info("No missing cells found - all tracked cells have corresponding mask pixels")
    
    return cleaned_trj, id_masks, removed_tracks