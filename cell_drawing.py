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
# V6 (Gemini - This Update): Robust create_id_masks with window search and marker painting fallback.
# V7 (User Request): Added debug logging for text overlay in create_raw_id_mask_animation_frames.
# V8 (Gemini - This Update): Prioritize 'original_mask_label' in create_id_masks.

import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.draw import line, line_aa
from skimage.morphology import binary_erosion
from skimage.measure import label as skimage_label, regionprops, find_contours
import logging
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, ListedColormap
from skimage.transform import resize
import colorsys
from random import shuffle
import imageio
from skimage import img_as_ubyte
import re
from collections import defaultdict
import matplotlib.colors as mcolors  # Add this import near other imports


def _distance(p1, p2):
    """Helper function to calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def prepare_mask_colors(merged_masks, trj):
    """Prepares ID masks and generates distinct colors for cell IDs."""
    if 'particle' not in trj.columns or trj['particle'].isna().all():
        logging.warning("No valid 'particle' IDs in trajectory for color preparation.")
        background_id = 0
        if merged_masks is None:
            logging.error("merged_masks is None in prepare_mask_colors. Cannot proceed.")
            dummy_shape = (100, 100, 1)
            id_masks_out = np.zeros(dummy_shape, dtype=np.int32) + background_id
        else:
            id_masks_out = np.zeros_like(merged_masks, dtype=np.int32) + background_id

        cell_ids_list = [background_id]
        color_list = [(20, 20, 20)]
        return id_masks_out, cell_ids_list, color_list, background_id

    valid_particle_ids = pd.to_numeric(trj['particle'], errors='coerce').dropna().astype(int)
    cell_ids_unique = sorted(list(set(valid_particle_ids)))
    background_id = 0

    if merged_masks is None:
        logging.error("merged_masks is None before calling create_id_masks. This will likely fail.")

    id_masks_out = create_id_masks(merged_masks, trj, background_id)

    cell_ids_list = [background_id] + cell_ids_unique

    actual_cell_ids_for_coloring = cell_ids_unique
    num_actual_cells = len(actual_cell_ids_for_coloring)

    generated_colors = generate_distinct_colors(n_colors=num_actual_cells,
                                                n_intensity_levels=3) if num_actual_cells > 0 else []

    color_list = [(20, 20, 20)] + generated_colors

    return id_masks_out, cell_ids_list, color_list, background_id


def create_id_masks(merged_masks, trj, background_id, fallback_search_radius=3, last_resort_marker_size=3):
    """
    Creates masks where pixel values correspond to cell IDs from the trajectory.
    Prioritizes using 'original_mask_label' from trj for pixel assignment.
    Includes fallback search and marker painting for robustness.
    """
    if merged_masks is None:
        logging.error("merged_masks is None in create_id_masks. Cannot create ID masks.")
        if not trj.empty and 'frame' in trj.columns and trj['frame'].max() > 0:  # Check if trj has frames
            max_frame = int(trj['frame'].max())
            logging.warning("Returning a small dummy id_masks array due to None merged_masks.")
            return np.full((100, 100, max_frame + 1), background_id, dtype=np.int32)
        else:  # Fallback if trj is also empty or has no frames
            return np.full((100, 100, 1), background_id, dtype=np.int32)

    id_masks_out = np.full_like(merged_masks, background_id, dtype=np.int32)

    if trj.empty or not all(col in trj.columns for col in ['particle', 'frame', 'x', 'y', 'original_mask_label']):
        logging.warning(
            "Trajectory data is missing required columns (incl. 'original_mask_label') for create_id_masks. Resulting ID masks might be empty or use fallbacks heavily.")
        # Continue to allow fallbacks to attempt painting

    trj_copy = trj.copy()
    trj_copy['particle'] = pd.to_numeric(trj_copy['particle'], errors='coerce')
    trj_copy.dropna(subset=['particle'], inplace=True)
    trj_copy['particle'] = trj_copy['particle'].astype(int)
    if 'original_mask_label' in trj_copy.columns:
        trj_copy['original_mask_label'] = pd.to_numeric(trj_copy['original_mask_label'], errors='coerce')  # Allow NaNs

    for i_frame in range(merged_masks.shape[2]):
        current_frame_particle_id_map = np.full(merged_masks.shape[:2], background_id, dtype=np.int32)

        # merged_masks should contain the original integer segmentation labels.
        current_original_segmentation_labels = merged_masks[:, :, i_frame].astype(np.int32)

        # For centroid/window fallbacks, get regionprops from the current frame's original segmentation
        props_for_frame_labels_fallback = regionprops(current_original_segmentation_labels,
                                                      intensity_image=current_original_segmentation_labels)
        label_centroids_fallback = {prop.label: prop.centroid for prop in props_for_frame_labels_fallback}

        trj_frame_data = trj_copy[trj_copy['frame'] == i_frame]

        if trj_frame_data.empty:
            id_masks_out[:, :, i_frame] = current_frame_particle_id_map
            logging.debug(f"Frame {i_frame}: No trajectory data. ID mask is all background.")
            continue

        assigned_original_labels_in_frame = set()

        for _, particle_row in trj_frame_data.iterrows():
            try:
                particle_id = int(particle_row['particle'])
                y_coord_orig, x_coord_orig = particle_row['y'], particle_row['x']
                y_coord_int, x_coord_int = int(round(y_coord_orig)), int(round(x_coord_orig))

                used_direct_original_label = False

                if 'original_mask_label' in particle_row and pd.notna(particle_row['original_mask_label']):
                    current_original_label_from_trj = int(particle_row['original_mask_label'])

                    if current_original_label_from_trj != 0 and np.any(
                            current_original_segmentation_labels == current_original_label_from_trj):
                        if current_original_label_from_trj not in assigned_original_labels_in_frame:
                            current_frame_particle_id_map[
                                current_original_segmentation_labels == current_original_label_from_trj] = particle_id
                            assigned_original_labels_in_frame.add(current_original_label_from_trj)
                            logging.debug(
                                f"Frame {i_frame}: Particle {particle_id} assigned via original_mask_label {current_original_label_from_trj}.")
                            used_direct_original_label = True
                        else:
                            logging.warning(
                                f"Frame {i_frame}: Particle {particle_id} wants original_mask_label {current_original_label_from_trj}, "
                                f"but it was already assigned in this frame. Trying fallbacks for Particle {particle_id}.")
                    else:
                        logging.warning(
                            f"Frame {i_frame}: Particle {particle_id} has original_mask_label {current_original_label_from_trj}, "
                            f"but this label is not found (or is 0) in 'current_original_segmentation_labels'. Proceeding to centroid fallback.")
                else:
                    logging.warning(
                        f"Frame {i_frame}: Particle {particle_id} missing 'original_mask_label' in trj. Proceeding to centroid-based assignment.")

                if used_direct_original_label:
                    continue

                    # --- Fallback Logic ---
                if not (0 <= y_coord_int < current_original_segmentation_labels.shape[0] and \
                        0 <= x_coord_int < current_original_segmentation_labels.shape[1]):
                    logging.warning(
                        f"Frame {i_frame}: Fallback: Particle {particle_id} centroid ({y_coord_int}, {x_coord_int}) is out of bounds. Skipping direct assignment fallback.")
                    if last_resort_marker_size > 0:
                        logging.info(
                            f"  Attempting last-resort marker paint for out-of-bounds Particle {particle_id} at ({y_coord_int}, {x_coord_int}).")
                        m_half = last_resort_marker_size // 2
                        min_r, max_r = max(0, y_coord_int - m_half), min(current_frame_particle_id_map.shape[0],
                                                                         y_coord_int + m_half + 1)
                        min_c, max_c = max(0, x_coord_int - m_half), min(current_frame_particle_id_map.shape[1],
                                                                         x_coord_int + m_half + 1)
                        if max_r > min_r and max_c > min_c:
                            current_frame_particle_id_map[min_r:max_r, min_c:max_c] = particle_id
                        else:
                            logging.warning(
                                f"  Marker for Particle {particle_id} also entirely out of bounds. Cannot paint.")
                    continue

                object_label_at_centroid = current_original_segmentation_labels[y_coord_int, x_coord_int]
                if object_label_at_centroid != 0:
                    if object_label_at_centroid not in assigned_original_labels_in_frame:
                        current_frame_particle_id_map[
                            current_original_segmentation_labels == object_label_at_centroid] = particle_id
                        assigned_original_labels_in_frame.add(object_label_at_centroid)
                        logging.debug(
                            f"Frame {i_frame}: Particle {particle_id} assigned via CENTROID fallback to original_label {object_label_at_centroid}.")
                        continue

                logging.debug(
                    f"Frame {i_frame}: Fallback: Particle {particle_id} centroid ({y_coord_int}, {x_coord_int}) "
                    f"hit background or taken label. Attempting window search (radius {fallback_search_radius}).")
                min_r_win = max(0, y_coord_int - fallback_search_radius)
                max_r_win = min(current_original_segmentation_labels.shape[0], y_coord_int + fallback_search_radius + 1)
                min_c_win = max(0, x_coord_int - fallback_search_radius)
                max_c_win = min(current_original_segmentation_labels.shape[1], x_coord_int + fallback_search_radius + 1)

                window_of_original_labels = current_original_segmentation_labels[min_r_win:max_r_win,
                                            min_c_win:max_c_win]
                unique_labels_in_window = np.unique(window_of_original_labels)
                candidate_nonzero_original_labels = [lbl for lbl in unique_labels_in_window if
                                                     lbl != 0 and lbl not in assigned_original_labels_in_frame]

                best_fallback_original_label = None
                min_dist_to_candidate = float('inf')

                if candidate_nonzero_original_labels:
                    for cand_orig_lbl in candidate_nonzero_original_labels:
                        if cand_orig_lbl in label_centroids_fallback:
                            cand_centroid_y, cand_centroid_x = label_centroids_fallback[cand_orig_lbl]
                            dist = _distance((y_coord_orig, x_coord_orig), (cand_centroid_y, cand_centroid_x))
                            if dist < min_dist_to_candidate:
                                min_dist_to_candidate = dist
                                best_fallback_original_label = cand_orig_lbl

                    if best_fallback_original_label is not None:
                        current_frame_particle_id_map[
                            current_original_segmentation_labels == best_fallback_original_label] = particle_id
                        assigned_original_labels_in_frame.add(best_fallback_original_label)
                        logging.info(
                            f"Frame {i_frame}: Particle {particle_id} assigned via WINDOW SEARCH fallback to original_label {best_fallback_original_label} (dist: {min_dist_to_candidate:.2f}).")
                        continue

                if last_resort_marker_size > 0 and not np.any(current_frame_particle_id_map == particle_id):
                    logging.warning(
                        f"Frame {i_frame}: Fallback: Particle {particle_id} not assigned by any method. Painting {last_resort_marker_size}x{last_resort_marker_size} marker at ({y_coord_int}, {x_coord_int}).")
                    m_half = last_resort_marker_size // 2
                    marker_r_start, marker_r_end = max(0, y_coord_int - m_half), min(
                        current_frame_particle_id_map.shape[0], y_coord_int + m_half + 1)
                    marker_c_start, marker_c_end = max(0, x_coord_int - m_half), min(
                        current_frame_particle_id_map.shape[1], x_coord_int + m_half + 1)
                    if marker_r_end > marker_r_start and marker_c_end > marker_c_start:
                        current_frame_particle_id_map[marker_r_start:marker_r_end,
                        marker_c_start:marker_c_end] = particle_id
                        logging.info(f"Successfully painted last-resort marker for Particle {particle_id} at ({y_coord_int}, {x_coord_int})")
                    else:
                        logging.warning(
                            f"  Last-resort marker for Particle {particle_id} at ({y_coord_int}, {x_coord_int}) resulted in zero size after bounds correction. Cannot paint.")
                        
                        # Final fallback: paint a single pixel at the centroid if it's within bounds
                        if 0 <= y_coord_int < current_frame_particle_id_map.shape[0] and 0 <= x_coord_int < current_frame_particle_id_map.shape[1]:
                            current_frame_particle_id_map[y_coord_int, x_coord_int] = particle_id
                            logging.info(f"Painted single pixel for Particle {particle_id} at ({y_coord_int}, {x_coord_int}) as final fallback")
                        else:
                            logging.error(f"Particle {particle_id} centroid ({y_coord_int}, {x_coord_int}) is completely out of bounds for frame {i_frame}")
                # --- End of Fallback Logic ---

            except (ValueError, KeyError, IndexError) as e_create_id:
                logging.error(
                    f"Error processing particle in create_id_masks for frame {i_frame}: {e_create_id}. Particle row: {particle_row.to_dict() if isinstance(particle_row, pd.Series) else particle_row}")
                continue

        id_masks_out[:, :, i_frame] = current_frame_particle_id_map

        # Verification loop
        for _, particle_row_verify in trj_frame_data.iterrows():
            try:
                pid_verify = int(particle_row_verify['particle'])
                if not np.any(id_masks_out[:, :, i_frame] == pid_verify):
                    logging.error(f"RES FILE POTENTIAL MISMATCH: Frame {i_frame}, Particle ID {pid_verify} from trj "
                                  f"was NOT painted onto id_masks_out for this frame. Centroid was "
                                  f"({int(round(particle_row_verify['y']))}, {int(round(particle_row_verify['x']))}). "
                                  f"Original_mask_label (if any): {particle_row_verify.get('original_mask_label', 'N/A')}. "
                                  f"This WILL cause 'Label in res_track but not in mask' warning from CTC eval if not fixed.")
            except (ValueError, KeyError):
                continue

    if not trj.empty:
        # Verify all tracked particles are represented in masks
        missing_particles = []
        for frame in range(id_masks_out.shape[2]):
            frame_trj = trj[trj['frame'] == frame]
            if frame_trj.empty:
                continue

            for _, row in frame_trj.iterrows():
                particle_id = int(row['particle'])
                if particle_id == background_id:
                    continue

                if not np.any(id_masks_out[:, :, frame] == particle_id):
                    missing_particles.append((particle_id, frame, row['y'], row['x']))
                    # Force paint a small marker at the centroid
                    y, x = int(round(row['y'])), int(round(row['x']))
                    if 0 <= y < id_masks_out.shape[0] and 0 <= x < id_masks_out.shape[1]:
                        # Paint a 3x3 marker
                        y_min = max(0, y - 1)
                        y_max = min(id_masks_out.shape[0], y + 2)
                        x_min = max(0, x - 1)
                        x_max = min(id_masks_out.shape[1], x + 2)
                        id_masks_out[y_min:y_max, x_min:x_max, frame] = particle_id
                        logging.info(f"Force-painted marker for particle {particle_id} at ({y},{x}) in frame {frame}")
                    else:
                        logging.error(
                            f"Cannot paint particle {particle_id} at ({y},{x}) - out of bounds for frame {frame}")
                else:
                    # Debug: Log successful ID mask creation
                    pixel_count = np.sum(id_masks_out[:, :, frame] == particle_id)
                    logging.debug(f"Particle {particle_id} has {pixel_count} pixels in frame {frame}")
        
        # Summary of missing particles
        if missing_particles:
            logging.warning(f"Found {len(missing_particles)} particle-frame combinations that needed force-painting:")
            for particle_id, frame, y, x in missing_particles[:5]:  # Show first 5
                logging.warning(f"  Particle {particle_id} in frame {frame} at ({y:.1f}, {x:.1f})")
            if len(missing_particles) > 5:
                logging.warning(f"  ... and {len(missing_particles) - 5} more")
        else:
            logging.info("All tracked particles were successfully represented in ID masks")

    return id_masks_out


def generate_distinct_colors(n_colors, n_intensity_levels=2, max_channel_val=255):
    if n_colors <= 0: return []
    n_colors_per_intensity = int(np.ceil(n_colors / n_intensity_levels))
    if n_colors_per_intensity == 0 and n_colors > 0: n_colors_per_intensity = 1

    RGB_tuples = []
    for intensity_idx in range(n_intensity_levels):
        current_value = 1.0 - (intensity_idx / n_intensity_levels) * 0.5
        current_saturation = 1.0 - (intensity_idx / n_intensity_levels) * 0.2

        hues = np.linspace(0, 1, n_colors_per_intensity, endpoint=False)
        for h_idx in range(n_colors_per_intensity):
            if len(RGB_tuples) < n_colors:
                r, g, b = colorsys.hsv_to_rgb(hues[h_idx], current_saturation, current_value)
                RGB_tuples.append((int(r * max_channel_val), int(g * max_channel_val), int(b * max_channel_val)))

    shuffle(RGB_tuples)
    return RGB_tuples[:n_colors]


def generate_state_specific_colors(unique_states):
    """
    Generate colors specifically optimized for cell states with better visual distinction.
    
    Args:
        unique_states: List of unique state names
    
    Returns:
        Dictionary mapping state names to RGB color tuples
    """
    # Predefined colors for common cell states (high contrast, distinguishable)
    state_color_map = {
        'cell': (0, 255, 0),        # Bright green for normal cells
        'mitosis': (255, 0, 0),     # Bright red for mitosis
        'fusion': (255, 0, 255),    # Magenta for fusion
        'unknown': (255, 255, 255), # Bright white for unknown (more visible)
        'hidden': (128, 128, 128),  # Gray for hidden
        'apoptosis': (255, 165, 0), # Orange for apoptosis
        'necrosis': (139, 69, 19),  # Brown for necrosis
        'quiescent': (0, 0, 255),   # Blue for quiescent
        'migrating': (255, 255, 0), # Yellow for migrating
        'differentiating': (0, 255, 255), # Cyan for differentiating
        'n/a': (255, 255, 255),     # Bright white for N/A (more visible)
        'unclassified': (255, 255, 255), # Bright white for unclassified (more visible)
    }
    
    # Generate colors for any additional states not in the predefined map
    additional_states = [state for state in unique_states if state.lower() not in state_color_map]
    if additional_states:
        additional_colors = generate_distinct_colors(len(additional_states), n_intensity_levels=3)
        for i, state in enumerate(additional_states):
            state_color_map[state] = additional_colors[i]
    
    # Create the final mapping, using predefined colors when available
    state_to_color = {}
    for state in unique_states:
        state_lower = state.lower()
        if state_lower in state_color_map:
            state_to_color[state] = state_color_map[state_lower]
        else:
            # Fallback to predefined map with original case
            state_to_color[state] = state_color_map.get(state, (128, 128, 128))
    
    logging.info(f"Generated state-specific colors: {state_to_color}")
    return state_to_color


def create_colorized_masks(id_masks, trj, cell_ids, background_id,
                           color_list, cell_color_idx,
                           cell_visibility, show_ids, cell_lineage=None, ancestry_map=None,
                           show_mitosis=False):
    logging.info('================= Colorizing masks (Simplified IDs for saved images) =================')
    if id_masks is None:
        logging.warning("id_masks is None in create_colorized_masks. Cannot proceed.")
        return []

    font = ImageFont.load_default()
    bg_color_tuple = (20, 20, 20)
    if background_id in cell_color_idx and 0 <= cell_color_idx[background_id] < len(color_list):
        bg_color_tuple = color_list[cell_color_idx[background_id]]

    bg_frame = np.full((id_masks.shape[0], id_masks.shape[1], 3), bg_color_tuple, dtype=np.uint8)
    colorized_masks_output = []

    for i_frame in range(id_masks.shape[2]):
        mask_frame_pixels = id_masks[:, :, i_frame]
        col_frame_current = bg_frame.copy()
        pil_id_layer = Image.new("RGBA", (mask_frame_pixels.shape[1], mask_frame_pixels.shape[0]),
                                 (0, 0, 0, 0)) if show_ids else None
        draw_text_pil = ImageDraw.Draw(pil_id_layer) if show_ids and pil_id_layer else None

        trj_frame = trj[trj['frame'] == i_frame] if not trj.empty else pd.DataFrame()

        unique_ids_in_mask_pixels = np.unique(mask_frame_pixels)

        for cell_id_val in unique_ids_in_mask_pixels:
            if cell_id_val == background_id: continue
            # Default to True if cell_visibility doesn't have this cell_id
            if not cell_visibility.get(cell_id_val, True): continue

            cell_color_rgb = (128, 128, 128)
            if cell_id_val in cell_color_idx and 0 <= cell_color_idx[cell_id_val] < len(color_list):
                cell_color_rgb = color_list[cell_color_idx[cell_id_val]]

            cell_coords = (mask_frame_pixels == cell_id_val)
            for ch in range(3): col_frame_current[cell_coords, ch] = cell_color_rgb[ch]

            if show_ids and draw_text_pil:
                cell_data_in_trj = trj_frame[pd.to_numeric(trj_frame['particle'], errors='coerce') == cell_id_val]
                if not cell_data_in_trj.empty:
                    cell_y_pos, cell_x_pos = int(cell_data_in_trj['y'].iloc[0]), int(cell_data_in_trj['x'].iloc[0])
                    id_txt_to_display = str(cell_id_val)
                    if show_mitosis:
                        # Check if parent_particle column exists, otherwise use lineage data
                        parent_val = pd.NA
                        if 'parent_particle' in cell_data_in_trj.columns:
                            parent_val_series = cell_data_in_trj['parent_particle']
                            parent_val = parent_val_series.iloc[0] if not parent_val_series.empty else pd.NA
                        
                        if pd.notna(parent_val) and parent_val != -1:
                            id_txt_to_display = f"D{cell_id_val}[P{int(parent_val)}]"
                        elif cell_lineage and cell_id_val in cell_lineage and cell_lineage[cell_id_val]:
                            id_txt_to_display = f"P{cell_id_val}({','.join(map(str, sorted(cell_lineage[cell_id_val])))})"
                        elif ancestry_map and cell_id_val in ancestry_map and ancestry_map[cell_id_val]:
                            # Use ancestry data as fallback
                            parent_ids = ancestry_map[cell_id_val]
                            id_txt_to_display = f"D{cell_id_val}[P{','.join(map(str, sorted(parent_ids)))}]"
                    logging.debug(
                        f"  ColorizedMask: Attempting to draw label for {cell_id_val}: '{id_txt_to_display}' at ({cell_x_pos},{cell_y_pos})")
                    try:
                        text_bbox = font.getbbox(id_txt_to_display);
                        id_width, id_height = text_bbox[2] - text_bbox[
                            0], text_bbox[3] - text_bbox[1]
                    except AttributeError:
                        id_width, id_height = draw_text_pil.textsize(id_txt_to_display, font=font)

                    text_x = min(max(0, cell_x_pos - id_width // 2), mask_frame_pixels.shape[1] - id_width)
                    text_y = min(max(0, cell_y_pos - id_height // 2), mask_frame_pixels.shape[0] - id_height)
                    draw_text_pil.text((text_x, text_y), id_txt_to_display, font=font,
                                       fill=(255, 255, 255, 255))  # Opaque white
                else:
                    logging.debug(
                        f"  ColorizedMask: No trj data for cell ID {cell_id_val} in frame {i_frame}, label not drawn.")

        if show_ids and pil_id_layer:
            id_layer_np = np.array(pil_id_layer);
            alpha_s = id_layer_np[:, :, 3] / 255.0;
            alpha_l = 1.0 - alpha_s
            for c in range(3): col_frame_current[:, :, c] = (
                    id_layer_np[:, :, c] * alpha_s + col_frame_current[:, :, c] * alpha_l)
        colorized_masks_output.append(col_frame_current.astype(np.uint8))
    return colorized_masks_output


def create_raw_id_mask_animation_frames(id_masks, trj, color_list, cell_color_idx, background_id,
                                        cell_visibility, show_ids, cell_lineage=None, show_mitosis=False):
    logging.info('================= Creating Raw ID Mask Animation Frames =================')
    if id_masks is None:
        logging.warning("id_masks is None. Cannot create raw ID mask animation.")
        return []

    font = ImageFont.load_default()
    animation_frames = []

    bg_color_tuple = (0, 0, 0)
    if background_id in cell_color_idx and 0 <= cell_color_idx[background_id] < len(color_list):
        bg_color_tuple = color_list[cell_color_idx[background_id]]
    else:
        logging.warning(
            f"Background ID {background_id} not in cell_color_idx or color_list. Defaulting background to black for ID GIF.")

    for i_frame in range(id_masks.shape[2]):
        logging.debug(f'Raw ID Anim Frame {i_frame}: Starting. show_ids={show_ids}')
        current_id_mask_frame = id_masks[:, :, i_frame]
        color_frame = np.full((current_id_mask_frame.shape[0], current_id_mask_frame.shape[1], 3),
                              bg_color_tuple, dtype=np.uint8)
        unique_particle_ids_in_frame = np.unique(current_id_mask_frame)

        for particle_id in unique_particle_ids_in_frame:
            if particle_id == background_id: continue
            # Default to True if cell_visibility doesn't have this particle_id
            if not cell_visibility.get(particle_id, True): continue
            color_for_id = (128, 128, 128)
            if particle_id in cell_color_idx and 0 <= cell_color_idx[particle_id] < len(color_list):
                color_for_id = color_list[cell_color_idx[particle_id]]
            mask_coords = (current_id_mask_frame == particle_id)
            for ch in range(3):
                color_frame[mask_coords, ch] = color_for_id[ch]

        if show_ids:
            pil_id_layer = Image.new("RGBA", (current_id_mask_frame.shape[1], current_id_mask_frame.shape[0]),
                                     (0, 0, 0, 0))
            draw_text_pil = ImageDraw.Draw(pil_id_layer)
            trj_frame_data = trj[trj['frame'] == i_frame] if not trj.empty else pd.DataFrame()

            if not trj_frame_data.empty:
                logging.debug(
                    f"  Raw ID Anim Frame {i_frame}: Found {len(trj_frame_data)} particles in trj for labeling.")
                for _, particle_row in trj_frame_data.iterrows():
                    try:
                        cell_id_val = int(particle_row['particle'])
                        if cell_id_val == background_id: continue
                        # Default to True if cell_visibility doesn't have this cell_id
                        if not cell_visibility.get(cell_id_val, True): continue
                        cell_y_pos = int(round(particle_row['y']))
                        cell_x_pos = int(round(particle_row['x']))

                        if not (0 <= cell_y_pos < current_id_mask_frame.shape[0] and \
                                0 <= cell_x_pos < current_id_mask_frame.shape[1]):
                            logging.debug(
                                f"  Raw ID Anim Frame {i_frame}: Label for {cell_id_val} skipped: centroid ({cell_x_pos},{cell_y_pos}) out of bounds {current_id_mask_frame.shape[:2]}.")
                            continue

                        id_txt_to_display = str(cell_id_val)
                        if show_mitosis:
                            parent_val_series = particle_row.get('parent_particle', pd.NA)
                            parent_val = parent_val_series if not isinstance(parent_val_series, pd.Series) else (
                                parent_val_series.iloc[0] if not parent_val_series.empty else pd.NA)

                            if pd.notna(parent_val) and parent_val != -1:
                                id_txt_to_display = f"D{cell_id_val}[P{int(parent_val)}]"
                            elif cell_lineage and cell_id_val in cell_lineage and cell_lineage[cell_id_val]:
                                id_txt_to_display = f"P{cell_id_val}({','.join(map(str, sorted(cell_lineage[cell_id_val])))})"

                        region_color_for_text = color_list[
                            cell_color_idx[cell_id_val]] if cell_id_val in cell_color_idx and \
                                                            cell_color_idx[cell_id_val] < len(
                            color_list) else (0, 0, 0)
                        luminance = 0.299 * region_color_for_text[0] + 0.587 * region_color_for_text[1] + 0.114 * \
                                    region_color_for_text[2]
                        text_fill_color = (0, 0, 0, 255) if luminance > 128 else (255, 255, 255, 255)

                        try:
                            text_bbox = font.getbbox(id_txt_to_display)
                            id_width, id_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                        except AttributeError:
                            id_width, id_height = draw_text_pil.textsize(id_txt_to_display, font=font)

                        text_x = min(max(0, cell_x_pos - id_width // 2), current_id_mask_frame.shape[1] - id_width)
                        text_y = min(max(0, cell_y_pos - id_height // 2), current_id_mask_frame.shape[0] - id_height)

                        logging.debug(
                            f"  Raw ID Anim Frame {i_frame}: Drawing label for {cell_id_val}: '{id_txt_to_display}' at ({text_x},{text_y}) with color {text_fill_color}")
                        draw_text_pil.text((text_x, text_y), id_txt_to_display, font=font, fill=text_fill_color)
                    except (ValueError, KeyError, IndexError) as e_label:
                        logging.debug(
                            f"  Raw ID Anim Frame {i_frame}: Skipping label for a particle due to {e_label}. Row: {particle_row.to_dict() if isinstance(particle_row, pd.Series) else particle_row}")
                        continue
            else:
                logging.debug(f"  Raw ID Anim Frame {i_frame}: No trj data, no labels drawn.")

            id_layer_np = np.array(pil_id_layer)
            alpha_s = id_layer_np[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                color_frame[:, :, c] = (id_layer_np[:, :, c] * alpha_s + color_frame[:, :, c] * alpha_l)
        else:
            logging.debug(f"  Raw ID Anim Frame {i_frame}: show_ids is False, no labels drawn.")

        animation_frames.append(color_frame.astype(np.uint8))
    logging.info(f"Generated {len(animation_frames)} frames for Raw ID Mask Animation.")
    return animation_frames


def create_colorized_tracks(id_masks, trj, cell_ids, background_id,
                            color_list, cell_color_idx,
                            cell_visibility, show_ids, show_contours, show_tracks_param,
                            use_thick_line, cell_lineage=None, ancestry_map=None,
                            show_mitosis=False):
    logging.info('================= Colorizing tracks (Simplified IDs for saved images) =================')
    if id_masks is None:
        logging.warning("id_masks is None in create_colorized_tracks. Cannot proceed.")
        return []

    font = ImageFont.load_default()
    structure_element = np.ones((5, 5)) if use_thick_line else np.ones((3, 3))
    colorized_tracks_output = []

    for i_frame in range(id_masks.shape[2]):
        col_frame_current = np.zeros((id_masks.shape[0], id_masks.shape[1], 3), dtype=np.uint8)
        pil_id_layer = Image.new("RGBA", (id_masks.shape[1], id_masks.shape[0]), (0, 0, 0, 0)) if show_ids else None
        draw_text_pil = ImageDraw.Draw(pil_id_layer) if show_ids and pil_id_layer else None

        trj_frame = trj[trj['frame'] == i_frame] if not trj.empty else pd.DataFrame()
        if not trj_frame.empty:
            trj_frame_copy = trj_frame.copy()
            trj_frame_copy['particle'] = pd.to_numeric(trj_frame_copy['particle'], errors='coerce')
            trj_frame_copy.dropna(subset=['particle'], inplace=True)
            trj_frame_copy['particle'] = trj_frame_copy['particle'].astype(int)

            unique_ids_in_frame_trj = trj_frame_copy['particle'].unique()

            for cell_id_val in unique_ids_in_frame_trj:
                if cell_id_val == background_id: continue
                # Default to True if cell_visibility doesn't have this cell_id
                if not cell_visibility.get(cell_id_val, True): continue

                cell_data_curr = trj_frame_copy[trj_frame_copy['particle'] == cell_id_val]
                if cell_data_curr.empty: continue

                cell_y_pos, cell_x_pos = int(cell_data_curr['y'].iloc[0]), int(cell_data_curr['x'].iloc[0])
                cell_color_rgb = color_list[cell_color_idx[cell_id_val]] if cell_id_val in cell_color_idx and \
                                                                            cell_color_idx[cell_id_val] < len(
                    color_list) else (128, 128, 128)

                if show_tracks_param:
                    trj_lookup_copy = trj.copy()
                    trj_lookup_copy['particle'] = pd.to_numeric(trj_lookup_copy['particle'],
                                                                errors='coerce').dropna().astype(int)

                    track_hist_df = trj_lookup_copy[(trj_lookup_copy['particle'] == cell_id_val) & (
                            trj_lookup_copy['frame'] <= i_frame)].sort_values(by='frame')
                    if len(track_hist_df) > 1:
                        for i in range(len(track_hist_df) - 1):
                            y1, x1 = int(track_hist_df['y'].iloc[i]), int(track_hist_df['x'].iloc[i])
                            y2, x2 = int(track_hist_df['y'].iloc[i + 1]), int(track_hist_df['x'].iloc[i + 1])
                            rr_l, cc_l, val = line_aa(y1, x1, y2, x2)
                            if use_thick_line:
                                for dr_thick in [-1, 0, 1]:
                                    for dc_thick in [-1, 0, 1]:
                                        rr_t, cc_t, val_t = line_aa(y1 + dr_thick, x1 + dc_thick, y2 + dr_thick,
                                                                    x2 + dc_thick)
                                        v_idx_t = (rr_t >= 0) & (rr_t < col_frame_current.shape[0]) & \
                                                  (cc_t >= 0) & (cc_t < col_frame_current.shape[1])
                                        for ch_idx in range(3):
                                            existing_color = col_frame_current[rr_t[v_idx_t], cc_t[v_idx_t], ch_idx]
                                            new_val = (cell_color_rgb[ch_idx] * val_t[v_idx_t] + existing_color * (
                                                    1 - val_t[v_idx_t])).astype(np.uint8)
                                            col_frame_current[rr_t[v_idx_t], cc_t[v_idx_t], ch_idx] = new_val
                            else:
                                v_idx_l = (rr_l >= 0) & (rr_l < col_frame_current.shape[0]) & \
                                          (cc_l >= 0) & (cc_l < col_frame_current.shape[1])
                                for ch_idx in range(3):
                                    existing_color = col_frame_current[rr_l[v_idx_l], cc_l[v_idx_l], ch_idx]
                                    new_val = (cell_color_rgb[ch_idx] * val[v_idx_l] + existing_color * (
                                            1 - val[v_idx_l])).astype(np.uint8)
                                    col_frame_current[rr_l[v_idx_l], cc_l[v_idx_l], ch_idx] = new_val
                    elif len(track_hist_df) == 1:
                        y_pt, x_pt = int(track_hist_df['y'].iloc[0]), int(track_hist_df['x'].iloc[0])
                        if 0 <= y_pt < col_frame_current.shape[0] and 0 <= x_pt < col_frame_current.shape[1]:
                            for ch_idx in range(3): col_frame_current[y_pt, x_pt, ch_idx] = cell_color_rgb[ch_idx]

                if show_contours:
                    mask_frame_curr = id_masks[:, :, i_frame]
                    cell_mask_bin = (mask_frame_curr == cell_id_val).astype(np.uint8)
                    if np.any(cell_mask_bin):
                        eroded = binary_erosion(cell_mask_bin, footprint=structure_element)
                        border = np.where(cell_mask_bin - eroded > 0)
                        for ch_idx in range(3): col_frame_current[border[0], border[1], ch_idx] = cell_color_rgb[ch_idx]

                if show_ids and draw_text_pil:
                    id_txt_to_display = str(cell_id_val)
                    if show_mitosis:
                        # Check if parent_particle column exists, otherwise use lineage data
                        parent_val = pd.NA
                        if 'parent_particle' in cell_data_curr.columns:
                            parent_val_series = cell_data_curr['parent_particle']
                            parent_val = parent_val_series.iloc[0] if not parent_val_series.empty else pd.NA
                        
                        if pd.notna(parent_val) and parent_val != -1:
                            id_txt_to_display = f"D{cell_id_val}[P{int(parent_val)}]"
                        elif cell_lineage and cell_id_val in cell_lineage and cell_lineage[cell_id_val]:
                            id_txt_to_display = f"P{cell_id_val}({','.join(map(str, sorted(cell_lineage[cell_id_val])))})"
                        elif ancestry_map and cell_id_val in ancestry_map and ancestry_map[cell_id_val]:
                            # Use ancestry data as fallback
                            parent_ids = ancestry_map[cell_id_val]
                            id_txt_to_display = f"D{cell_id_val}[P{','.join(map(str, sorted(parent_ids)))}]"
                    try:
                        text_bbox = font.getbbox(id_txt_to_display);
                        id_width, id_height = text_bbox[2] - text_bbox[0], \
                                              text_bbox[3] - text_bbox[1]
                    except AttributeError:
                        id_width, id_height = draw_text_pil.textsize(id_txt_to_display, font=font)
                    text_x = min(max(0, cell_x_pos - id_width // 2), id_masks.shape[1] - id_width)
                    text_y = min(max(0, cell_y_pos - id_height // 2), id_masks.shape[0] - id_height)
                    draw_text_pil.text((text_x, text_y), id_txt_to_display, font=font, fill=(*cell_color_rgb, 255))

        if show_ids and pil_id_layer:
            id_layer_np = np.array(pil_id_layer);
            alpha_s = id_layer_np[:, :, 3] / 255.0;
            alpha_l = 1.0 - alpha_s
            for c_rgb in range(3): col_frame_current[:, :, c_rgb] = (
                    id_layer_np[:, :, c_rgb] * alpha_s + col_frame_current[:, :, c_rgb] * alpha_l)
        colorized_tracks_output.append(col_frame_current.astype(np.uint8))
    return colorized_tracks_output


def create_track_overview(id_masks, trj, cell_ids, background_id,
                          color_list, cell_color_idx,
                          cell_visibility, show_ids,
                          use_thick_line, cell_lineage=None, ancestry_map=None,
                          show_mitosis=False):
    logging.info('================= Creating track overview (Simplified IDs for saved image) =================')
    if id_masks is None:
        logging.warning("id_masks is None in create_track_overview. Cannot proceed.")
        return np.zeros((100, 100, 3), dtype=np.uint8)

    track_overview_img = np.zeros((id_masks.shape[0], id_masks.shape[1], 3), dtype=np.uint8)
    pil_id_layer_overview = Image.new("RGBA", (id_masks.shape[1], id_masks.shape[0]),
                                      (0, 0, 0, 0)) if show_ids else None
    draw_text_pil_overview = ImageDraw.Draw(pil_id_layer_overview) if show_ids and pil_id_layer_overview else None
    font = ImageFont.load_default()

    if 'particle' not in trj.columns or trj['particle'].isna().all():
        logging.warning("No valid particles in trajectory for track overview.")
        if show_ids and pil_id_layer_overview:
            id_layer_np_overview = np.array(pil_id_layer_overview);
            alpha_s_ov = id_layer_np_overview[:, :, 3] / 255.0;
            alpha_l_ov = 1.0 - alpha_s_ov
            for c_rgb_idx in range(3): track_overview_img[:, :, c_rgb_idx] = (
                    id_layer_np_overview[:, :, c_rgb_idx] * alpha_s_ov + track_overview_img[:, :,
                                                                         c_rgb_idx] * alpha_l_ov)
        return track_overview_img.astype(np.uint8)

    trj_copy_overview = trj.copy()
    trj_copy_overview['particle'] = pd.to_numeric(trj_copy_overview['particle'], errors='coerce')
    trj_copy_overview.dropna(subset=['particle'], inplace=True)
    trj_copy_overview['particle'] = trj_copy_overview['particle'].astype(int)

    unique_trj_particle_ids = trj_copy_overview['particle'].unique()

    for cell_id_val in unique_trj_particle_ids:
        if cell_id_val == background_id or not cell_visibility.get(cell_id_val, True): continue

        cell_color_rgb = color_list[cell_color_idx[cell_id_val]] if cell_id_val in cell_color_idx and cell_color_idx[
            cell_id_val] < len(color_list) else (128, 128, 128)

        track_hist_df = trj_copy_overview[(trj_copy_overview['particle'] == cell_id_val)].sort_values(by='frame')

        if len(track_hist_df) > 1:
            for i in range(len(track_hist_df) - 1):
                y1, x1 = int(track_hist_df['y'].iloc[i]), int(track_hist_df['x'].iloc[i])
                y2, x2 = int(track_hist_df['y'].iloc[i + 1]), int(track_hist_df['x'].iloc[i + 1])
                rr_l, cc_l, val = line_aa(y1, x1, y2, x2)
                if use_thick_line:
                    for dr_thick_ov in [-1, 0, 1]:
                        for dc_thick_ov in [-1, 0, 1]:
                            rr_t_ov, cc_t_ov, val_t_ov = line_aa(y1 + dr_thick_ov, x1 + dc_thick_ov, y2 + dr_thick_ov,
                                                                 x2 + dc_thick_ov)
                            v_idx_t_ov = (rr_t_ov >= 0) & (rr_t_ov < track_overview_img.shape[0]) & \
                                         (cc_t_ov >= 0) & (cc_t_ov < track_overview_img.shape[1])
                            for ch_idx_ov in range(3):
                                existing_color_ov = track_overview_img[
                                    rr_t_ov[v_idx_t_ov], cc_t_ov[v_idx_t_ov], ch_idx_ov]
                                new_val_ov = (cell_color_rgb[ch_idx_ov] * val_t_ov[v_idx_t_ov] + existing_color_ov * (
                                        1 - val_t_ov[v_idx_t_ov])).astype(np.uint8)
                                track_overview_img[rr_t_ov[v_idx_t_ov], cc_t_ov[v_idx_t_ov], ch_idx_ov] = new_val_ov
                else:
                    v_idx_l_ov = (rr_l >= 0) & (rr_l < track_overview_img.shape[0]) & \
                                 (cc_l >= 0) & (cc_l < track_overview_img.shape[1])
                    for ch_idx_ov in range(3):
                        existing_color_ov = track_overview_img[rr_l[v_idx_l_ov], cc_l[v_idx_l_ov], ch_idx_ov]
                        new_val_ov = (cell_color_rgb[ch_idx_ov] * val[v_idx_l_ov] + existing_color_ov * (
                                1 - val[v_idx_l_ov])).astype(np.uint8)
                        track_overview_img[rr_l[v_idx_l_ov], cc_l[v_idx_l_ov], ch_idx_ov] = new_val_ov
        elif len(track_hist_df) == 1:
            y_pt, x_pt = int(track_hist_df['y'].iloc[0]), int(track_hist_df['x'].iloc[0])
            if 0 <= y_pt < track_overview_img.shape[0] and 0 <= x_pt < track_overview_img.shape[1]:
                for ch_idx_ov in range(3): track_overview_img[y_pt, x_pt, ch_idx_ov] = cell_color_rgb[ch_idx_ov]

        if show_ids and draw_text_pil_overview and not track_hist_df.empty:
            first_pos_y, first_pos_x = int(track_hist_df['y'].iloc[0]), int(track_hist_df['x'].iloc[0])
            id_txt_to_display = str(cell_id_val)
            if show_mitosis:
                # Check if parent_particle column exists, otherwise use lineage data
                parent_val = pd.NA
                if 'parent_particle' in track_hist_df.columns:
                    parent_val_series = track_hist_df['parent_particle']
                    parent_val = parent_val_series.iloc[0] if not parent_val_series.empty else pd.NA
                
                if pd.notna(parent_val) and parent_val != -1: 
                    id_txt_to_display = f"D{cell_id_val}[P{int(parent_val)}]"
                elif cell_lineage and cell_id_val in cell_lineage and cell_lineage[cell_id_val]:
                    id_txt_to_display = f"P{cell_id_val}({','.join(map(str, sorted(cell_lineage[cell_id_val])))})"
                elif ancestry_map and cell_id_val in ancestry_map and ancestry_map[cell_id_val]:
                    # Use ancestry data as fallback
                    parent_ids = ancestry_map[cell_id_val]
                    id_txt_to_display = f"D{cell_id_val}[P{','.join(map(str, sorted(parent_ids)))}]"
            try:
                text_bbox = font.getbbox(id_txt_to_display);
                id_width, id_height = text_bbox[2] - text_bbox[0], \
                                      text_bbox[3] - text_bbox[1]
            except AttributeError:
                id_width, id_height = draw_text_pil_overview.textsize(id_txt_to_display, font=font)
            text_x = min(max(0, first_pos_x - id_width // 2), id_masks.shape[1] - id_width)
            text_y = min(max(0, first_pos_y - id_height // 2), id_masks.shape[0] - id_height)
            draw_text_pil_overview.text((text_x, text_y), id_txt_to_display, font=font, fill=(*cell_color_rgb, 255))

    if show_ids and pil_id_layer_overview:
        id_layer_np_overview = np.array(pil_id_layer_overview);
        alpha_s_ov = id_layer_np_overview[:, :, 3] / 255.0;
        alpha_l_ov = 1.0 - alpha_s_ov
        for c_rgb_idx_ov in range(3): track_overview_img[:, :, c_rgb_idx_ov] = (
                id_layer_np_overview[:, :, c_rgb_idx_ov] * alpha_s_ov + track_overview_img[:, :,
                                                                        c_rgb_idx_ov] * alpha_l_ov)
    return track_overview_img.astype(np.uint8)


def save_temporal_outline_stack(id_masks, output_path, colormap_name="plasma", alpha=0.4, dpi=300, scale_factor=3, fast_mode=False, figsize_scale=1.0):
    """
    Generate a temporal outline stack image: extract cell outlines for each frame, overlay them with transparency,
    and color them by time using a colormap. Save the result as a high-resolution PNG with a white background.
    Adds a vertical colorbar labeled with frame numbers.
    
    OPTIMIZED VERSION: Significantly faster than the original implementation by:
    - Using vectorized operations instead of nested loops
    - Reducing redundant contour processing
    - Optimizing glow effects with pre-computed offsets
    - Using more efficient pixel manipulation
    - Implementing smart scaling strategies
    
    Args:
        id_masks: 3D numpy array (H, W, T) of cell ID masks
        output_path: Path to save the output PNG
        colormap_name: Name of the matplotlib colormap to use
        alpha: Transparency of the outline overlays (0-1)
        dpi: DPI for the colorbar (higher = sharper)
        scale_factor: Factor to scale up the image resolution (higher = larger, sharper image)
    """
    if id_masks is None or id_masks.ndim != 3:
        raise ValueError("id_masks must be a 3D numpy array (H, W, T)")
    
    H, W, T = id_masks.shape
    
    # Fast mode implementation for very large datasets
    if fast_mode:
        logging.info("Using fast mode for temporal outline stack generation")
        return _save_temporal_outline_stack_fast(id_masks, output_path, colormap_name, alpha, dpi, scale_factor, figsize_scale)
    
    # Performance optimization: Limit scale factor for very large datasets
    if H * W * T > 10000000:  # 10M pixels threshold
        scale_factor = min(scale_factor, 2)
        logging.info(f"Large dataset detected ({H*W*T} pixels), limiting scale_factor to {scale_factor} for performance")
    
    # Scale up the image dimensions for higher resolution
    H_scaled = H * scale_factor
    W_scaled = W * scale_factor
    
    # Start with white background (RGBA) at higher resolution
    outline_img = np.ones((H_scaled, W_scaled, 4), dtype=np.float32)  # White background
    outline_img[:, :, 3] = 1.0  # Full alpha for background
    
    cmap = plt.get_cmap(colormap_name)
    
    # Pre-compute glow offsets for efficiency
    glow_offsets = []
    for dy in [-2, -1, 0, 1, 2]:
        for dx in [-2, -1, 0, 1, 2]:
            if abs(dy) + abs(dx) <= 2:  # Glow radius
                glow_offsets.append((dy, dx))
    
    main_offsets = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if abs(dy) + abs(dx) <= 1:  # Main outline radius
                main_offsets.append((dy, dx))
    
    # Process frames in batches for better memory management
    batch_size = min(10, T)  # Process 10 frames at a time or all frames if less than 10
    
    for batch_start in range(0, T, batch_size):
        batch_end = min(batch_start + batch_size, T)
        batch_frames = range(batch_start, batch_end)
        
        for t in batch_frames:
            mask = id_masks[:, :, t]
            unique_cells = np.unique(mask)
            unique_cells = unique_cells[unique_cells != 0]  # Exclude background
            
            if len(unique_cells) == 0:
                continue
                
            color = cmap(t / max(T - 1, 1))  # RGBA, values 0-1
            
            # Process all cells in this frame at once
            for cell_id in unique_cells:
                binary = (mask == cell_id).astype(np.uint8)
                
                # Skip if no pixels for this cell
                if not np.any(binary):
                    continue
                
                # Find contours more efficiently
                contours = find_contours(binary, 0.5)
                
                for contour in contours:
                    if len(contour) < 2:
                        continue
                        
                    # Scale up the contour coordinates
                    contour_scaled = contour * scale_factor
                    contour_int = np.round(contour_scaled).astype(int)
                    
                    # Draw line segments more efficiently
                    for i in range(len(contour_int) - 1):
                        y0, x0 = contour_int[i]
                        y1, x1 = contour_int[i + 1]
                        
                        # Get line pixels
                        rr, cc = line(y0, x0, y1, x1)
                        
                        # Apply glow effect
                        for dy, dx in glow_offsets:
                            rr_glow = rr + dy
                            cc_glow = cc + dx
                            
                            # Vectorized boundary checking
                            valid = (rr_glow >= 0) & (rr_glow < H_scaled) & (cc_glow >= 0) & (cc_glow < W_scaled)
                            rr_valid = rr_glow[valid]
                            cc_valid = cc_glow[valid]
                            
                            if len(rr_valid) > 0:
                                # Vectorized color assignment
                                outline_img[rr_valid, cc_valid, :3] = color[:3]
                                outline_img[rr_valid, cc_valid, 3] = 0.3
                        
                        # Apply main outline
                        for dy, dx in main_offsets:
                            rr_main = rr + dy
                            cc_main = cc + dx
                            
                            # Vectorized boundary checking
                            valid = (rr_main >= 0) & (rr_main < H_scaled) & (cc_main >= 0) & (cc_main < W_scaled)
                            rr_valid = rr_main[valid]
                            cc_valid = cc_main[valid]
                            
                            if len(rr_valid) > 0:
                                # Vectorized color assignment
                                outline_img[rr_valid, cc_valid, :3] = color[:3]
                                outline_img[rr_valid, cc_valid, 3] = 0.9
                    
                    # Draw closing segment if contour is closed
                    if len(contour_int) > 2:
                        y0, x0 = contour_int[-1]
                        y1, x1 = contour_int[0]
                        
                        rr, cc = line(y0, x0, y1, x1)
                        
                        # Apply glow effect to closing segment
                        for dy, dx in glow_offsets:
                            rr_glow = rr + dy
                            cc_glow = cc + dx
                            
                            valid = (rr_glow >= 0) & (rr_glow < H_scaled) & (cc_glow >= 0) & (cc_glow < W_scaled)
                            rr_valid = rr_glow[valid]
                            cc_valid = cc_glow[valid]
                            
                            if len(rr_valid) > 0:
                                outline_img[rr_valid, cc_valid, :3] = color[:3]
                                outline_img[rr_valid, cc_valid, 3] = 0.3
                        
                        # Apply main outline to closing segment
                        for dy, dx in main_offsets:
                            rr_main = rr + dy
                            cc_main = cc + dx
                            
                            valid = (rr_main >= 0) & (rr_main < H_scaled) & (cc_main >= 0) & (cc_main < W_scaled)
                            rr_valid = rr_main[valid]
                            cc_valid = cc_main[valid]
                            
                            if len(rr_valid) > 0:
                                outline_img[rr_valid, cc_valid, :3] = color[:3]
                                outline_img[rr_valid, cc_valid, 3] = 0.9
    
    outline_img_uint8 = (np.clip(outline_img, 0, 1) * 255).astype(np.uint8)

    # Create high-resolution colorbar with ticks and labels
    # Calculate colorbar width to be proportional to the image
    colorbar_width = max(150, W_scaled // 10)  # Wider colorbar for better readability
    
    # Calculate tick positions - show more ticks for longer sequences
    if T <= 10:
        tick_positions = list(range(T))
        tick_labels = [f'{t}' for t in range(T)]
    elif T <= 50:
        # Show every 5th frame
        tick_positions = list(range(0, T, 5))
        tick_labels = [f'{t}' for t in tick_positions]
    else:
        # Show every 10th frame for very long sequences
        tick_positions = list(range(0, T, 10))
        tick_labels = [f'{t}' for t in tick_positions]
    
    # Always include the last frame
    if T-1 not in tick_positions:
        tick_positions.append(T-1)
        tick_labels.append(f'{T-1}')
    
    # Use PIL colorbar approach for guaranteed label visibility
    logging.info("Using PIL colorbar approach for guaranteed label visibility")
    
    # Create manual colorbar using PIL
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a white background
    colorbar_pil = Image.new('RGBA', (colorbar_width, H_scaled), (255, 255, 255, 255))
    draw = ImageDraw.Draw(colorbar_pil)
    
    # Calculate dimensions
    gradient_width = colorbar_width // 3  # Color gradient takes 1/3 of width
    
    # Get the colormap
    norm = Normalize(vmin=0, vmax=T-1)
    
    # Draw the color gradient
    gradient_x = 10  # Start position
    for y in range(H_scaled):
        # Map y position to frame number (inverted: 0 at top, T-1 at bottom)
        frame_ratio = 1.0 - (y / H_scaled)
        frame_num = frame_ratio * (T - 1)
        
        # Get color from colormap
        color = cmap(norm(frame_num))
        color_rgb = tuple(int(c * 255) for c in color[:3])
        
        # Draw a horizontal line for this color
        draw.line([(gradient_x, y), (gradient_x + gradient_width, y)], fill=color_rgb, width=1)
    
    # Draw tick marks and labels with better visibility
    font_size = max(14, min(20, H_scaled // (len(tick_positions) + 2)))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    for i, (pos, label) in enumerate(zip(tick_positions, tick_labels)):
        # Calculate y position for this tick
        frame_ratio = 1.0 - (pos / (T - 1))
        y_pos = int(frame_ratio * H_scaled)
        
        # Draw tick mark (thicker and more prominent)
        tick_x = gradient_x + gradient_width + 5
        draw.line([(tick_x, y_pos), (tick_x + 10, y_pos)], fill=(0, 0, 0, 255), width=3)
        
        # Draw label with clean positioning and no background
        label_x = tick_x + 25  # Adjusted offset for better positioning
        label_y = y_pos - font_size // 2
        
        # Draw the text directly without any background
        draw.text((label_x, label_y), label, fill=(0, 0, 0, 255), font=font)
    
    # Draw title (centered, above colorbar, not overlapping)
    title_font_size = max(12, font_size + 2)
    try:
        title_font = ImageFont.truetype("arial.ttf", title_font_size)
    except:
        title_font = ImageFont.load_default()
    
    title = "Temporal Progression"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = gradient_x + (gradient_width // 2) - (title_width // 2)
    title_y = 10
    draw.text((title_x, title_y), title, fill=(0, 0, 0, 255), font=title_font)
    
    # Draw axis label (centered, below colorbar, not overlapping)
    label_text = "Time (Frame Number)"
    label_bbox = draw.textbbox((0, 0), label_text, font=font)
    label_width_text = label_bbox[2] - label_bbox[0]
    label_x = gradient_x + (gradient_width // 2) - (label_width_text // 2)
    label_y = H_scaled - font_size - 10
    draw.text((label_x, label_y), label_text, fill=(0, 0, 0, 255), font=font)
    
    # Convert PIL image to numpy array
    colorbar_img = np.array(colorbar_pil)
    
    logging.info("✓ Successfully created manual PIL colorbar with ticks")
    
    # Ensure colorbar has the same height as the main image
    if colorbar_img.shape[0] != H_scaled:
        # Use high-quality resizing
        from skimage.transform import resize
        colorbar_img = resize(colorbar_img, (H_scaled, colorbar_img.shape[1]), 
                            preserve_range=True, order=1, anti_aliasing=True).astype(np.uint8)
    
    # Debug: Print colorbar info
    logging.info(f"Colorbar shape: {colorbar_img.shape}, Main image shape: {outline_img_uint8.shape}")
    logging.info(f"T={T}, Colorbar tick positions: {tick_positions}, Colorbar labels: {tick_labels}")
    
    # Add some padding between the main image and colorbar
    padding_width = 10
    padding = np.ones((H_scaled, padding_width, 4), dtype=np.uint8) * 255
    padding[:, :, 3] = 255  # Full alpha for white padding
    
    # Concatenate outline image, padding, and colorbar horizontally
    combined_img = np.concatenate([outline_img_uint8, padding, colorbar_img], axis=1)
    
    # Save with high quality settings
    from imageio import imwrite
    imwrite(output_path, combined_img, quality=95)
    
    logging.info(f"Saved high-resolution temporal outline stack to {output_path} (scale_factor={scale_factor}, dpi={dpi})")

def save_state_masks_and_colors(id_masks, trj, output_dir, mask_prefix='state_mask', color_prefix='state_mask_color', colormap_name='tab20'):
    """
    For each frame, save a 16-bit PNG mask (pixel value = cell state) and a color mask (categorical colormap) if 'state' is available in trj.
    """
    import numpy as np
    from imageio import imwrite
    if id_masks is None or trj is None or 'state' not in trj.columns:
        return
    H, W, T = id_masks.shape
    os.makedirs(os.path.join(output_dir, 'State_masks_per_frame'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'State_masks_color_per_frame'), exist_ok=True)
    # Get all unique states and assign color
    all_states = sorted([s for s in trj['state'].unique() if pd.notna(s)])
    state_to_idx = {s: i+1 for i, s in enumerate(all_states)}  # 0 is background
    n_states = len(all_states)
    cmap = plt.get_cmap(colormap_name, n_states+1)
    for t in range(T):
        mask = id_masks[:, :, t]
        state_mask = np.zeros((H, W), dtype=np.uint16)
        color_mask = np.zeros((H, W, 3), dtype=np.uint8)
        trj_frame = trj[trj['frame'] == t]
        id_to_state = {int(row['particle']): row['state'] for _, row in trj_frame.iterrows() if pd.notna(row['state'])}
        for cell_id in np.unique(mask):
            if cell_id == 0:
                continue
            state = id_to_state.get(int(cell_id), None)
            if state is None:
                continue
            state_idx = state_to_idx[state]
            state_mask[mask == cell_id] = state_idx
            color = (np.array(cmap(state_idx)[:3]) * 255).astype(np.uint8)
            color_mask[mask == cell_id] = color
        # Save 16-bit mask
        imwrite(os.path.join(output_dir, 'State_masks_per_frame', f'{mask_prefix}_{t:03d}.png'), state_mask)
        # Save color mask
        imwrite(os.path.join(output_dir, 'State_masks_color_per_frame', f'{color_prefix}_{t:03d}.png'), color_mask)
    
    # Save per-state binary masks in State_masks_by_type/StateName/
    base_by_type = os.path.join(output_dir, 'State_masks_by_type')
    for state in all_states:
        # Create folder using actual state name, sanitized for filesystem
        state_folder_name = _sanitize_filename(state)
        state_folder = os.path.join(base_by_type, state_folder_name)
        os.makedirs(state_folder, exist_ok=True)
    
    for t in range(T):
        mask = id_masks[:, :, t]
        trj_frame = trj[trj['frame'] == t]
        id_to_state = {int(row['particle']): row['state'] for _, row in trj_frame.iterrows() if pd.notna(row['state'])}
        
        # Save per-state binary and color masks
        for state in all_states:
            state_idx = state_to_idx[state]
            state_folder_name = _sanitize_filename(state)
            state_folder = os.path.join(base_by_type, state_folder_name)
            
            # Create binary mask for this state
            binary_mask = np.zeros((H, W), dtype=np.uint16)
            for cell_id in np.unique(mask):
                if cell_id == 0:
                    continue
                cell_state = id_to_state.get(int(cell_id), None)
                if cell_state == state:
                    binary_mask[mask == cell_id] = cell_id  # Use cell ID as value
            
            # Save binary mask
            imwrite(os.path.join(state_folder, f'{state_folder_name}_mask_{t:03d}.png'), binary_mask)
            
            # Create color mask for this state
            state_color_mask = np.zeros((H, W, 3), dtype=np.uint8)
            state_color = (np.array(cmap(state_idx)[:3]) * 255).astype(np.uint8)
            state_color_mask[binary_mask > 0] = state_color
            
            # Save color mask
            imwrite(os.path.join(state_folder, f'{state_folder_name}_color_mask_{t:03d}.png'), state_color_mask)


def _sanitize_filename(filename):
    """
    Sanitize a string to be used as a filename by removing/replacing invalid characters.
    """
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    # Replace multiple underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    # Ensure it's not empty
    if not sanitized:
        sanitized = 'unknown'
    return sanitized


def save_offset_track_plot(trj, output_path, left_color='black', right_color='red'):
    """
    Save a plot of all tracks offset so each cell's position at t=0 (first frame) is normalized to (0,0).
    Tracks are colored by their final x position: left_color for x<0, right_color for x>=0.
    """
    if trj is None or trj.empty or 'x' not in trj.columns or 'y' not in trj.columns or 'frame' not in trj.columns or 'particle' not in trj.columns:
        return
    
    plt.figure(figsize=(6, 6))
    count_left = 0
    count_right = 0
    
    for pid, group in trj.groupby('particle'):
        if group.empty:
            continue
            
        # Sort by frame to ensure proper ordering
        group = group.sort_values('frame')
        
        # Get the first frame position (t=0) for this cell
        first_frame_data = group.iloc[0]
        first_x = first_frame_data['x']
        first_y = first_frame_data['y']
        
        # Offset all positions relative to this cell's first position
        xs = group['x'] - first_x
        ys = group['y'] - first_y
        
        if len(xs) == 0:
            continue
            
        # Color based on final position relative to origin
        final_x = xs.iloc[-1]
        color = right_color if final_x >= 0 else left_color
        if final_x >= 0:
            count_right += 1
        else:
            count_left += 1
            
        plt.plot(xs, ys, color=color, marker='o', markersize=3, linewidth=1)
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('x axis [μm]')
    plt.ylabel('y axis [μm]')
    plt.title(f'Number of tracks: {len(trj["particle"].unique())}, Count up: {count_right}, Count down: {count_left}')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def draw_line_aa(y0, x0, y1, x1, H, W):
    """Draw anti-aliased line and return pixel indices (no blending, just pixel locations)."""
    from skimage.draw import line
    rr, cc = line(y0, x0, y1, x1)
    # Clip to image bounds
    valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    return rr[valid], cc[valid]


def draw_thick_line_aa(y0, x0, y1, x1, H, W, thickness=1.5):
    """Draw anti-aliased thick line and return pixel indices."""
    from skimage.draw import line
    import numpy as np
    
    # Draw the main line
    rr, cc = line(y0, x0, y1, x1)
    
    # For thickness > 1, add additional pixels around the line
    if thickness > 1.0:
        # Calculate perpendicular direction
        dy = y1 - y0
        dx = x1 - x0
        length = np.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            # Normalize perpendicular vector
            perp_x = -dy / length
            perp_y = dx / length
            
            # Calculate offset distance
            offset = (thickness - 1.0) / 2.0
            
            # Create additional lines parallel to the main line
            additional_rr = []
            additional_cc = []
            
            # Add lines above and below the main line
            for offset_mult in [-offset, offset]:
                y0_offset = y0 + perp_y * offset_mult
                y1_offset = y1 + perp_y * offset_mult
                x0_offset = x0 + perp_x * offset_mult
                x1_offset = x1 + perp_x * offset_mult
                
                rr_offset, cc_offset = line(int(round(y0_offset)), int(round(x0_offset)), 
                                          int(round(y1_offset)), int(round(x1_offset)))
                additional_rr.extend(rr_offset)
                additional_cc.extend(cc_offset)
            
            # Combine all pixels
            rr = np.concatenate([rr, additional_rr])
            cc = np.concatenate([cc, additional_cc])
    
    # Clip to image bounds
    valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    return rr[valid], cc[valid]


def create_cell_count_over_time_plot(trj, output_path, dpi=300, figsize_scale=1.0):
    """Create a plot showing cell count over time."""
    if trj is None or trj.empty:
        return
    
    import matplotlib.pyplot as plt
    
    # Count cells per frame
    cell_counts = trj.groupby('frame')['particle'].nunique()
    
    # Calculate figure size based on scale
    figsize = (12 * figsize_scale, 6 * figsize_scale)
    plt.figure(figsize=figsize)
    plt.plot(cell_counts.index, cell_counts.values, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Frame')
    plt.ylabel('Number of Cells')
    plt.title('Cell Count Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logging.info(f"Cell count over time plot saved to {output_path}")


def create_event_timeline_plot(trj, cell_lineage, ancestry, output_path, dpi=300, figsize_scale=1.0):
    """Create a timeline plot showing mitosis and fusion events."""
    if trj is None or trj.empty:
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Collect mitosis events (parent with >= 2 children)
    mitosis_events = []
    for parent_id, children in cell_lineage.items():
        if len(children) >= 2:
            # Find the frame where parent ends (last frame of parent)
            parent_frames = trj[trj['particle'] == parent_id]['frame']
            if not parent_frames.empty:
                mitosis_frame = parent_frames.max()
                mitosis_events.append(mitosis_frame)
    
    # Collect fusion events (child with >= 2 parents)
    fusion_events = []
    for child_id, parents in ancestry.items():
        if len(parents) >= 2:
            # Find the frame where child starts (first frame of child)
            child_frames = trj[trj['particle'] == child_id]['frame']
            if not child_frames.empty:
                fusion_frame = child_frames.min()
                fusion_events.append(fusion_frame)
    
    # Calculate figure size based on scale
    figsize = (14 * figsize_scale, 8 * figsize_scale)
    plt.figure(figsize=figsize)
    
    # Plot mitosis events
    if mitosis_events:
        plt.scatter(mitosis_events, [1] * len(mitosis_events), 
                   c='red', s=100, alpha=0.7, label=f'Mitosis ({len(mitosis_events)} events)', marker='^')
    
    # Plot fusion events
    if fusion_events:
        plt.scatter(fusion_events, [0.5] * len(fusion_events), 
                   c='blue', s=100, alpha=0.7, label=f'Fusion ({len(fusion_events)} events)', marker='v')
    
    plt.xlabel('Frame')
    plt.ylabel('Event Type')
    plt.title('Mitosis and Fusion Events Timeline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logging.info(f"Event timeline plot saved to {output_path}")


def create_cell_cycle_duration_plot(trj, cell_lineage, output_path, dpi=300, figsize_scale=1.0):
    """Create a histogram of cell cycle durations (time between divisions)."""
    if trj is None or trj.empty or not cell_lineage:
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    cycle_durations = []
    
    for parent_id, children in cell_lineage.items():
        if len(children) >= 2:  # Mitosis event
            # Find parent's start and end frames
            parent_frames = trj[trj['particle'] == parent_id]['frame']
            if not parent_frames.empty:
                parent_start = parent_frames.min()
                parent_end = parent_frames.max()
                cycle_duration = parent_end - parent_start + 1  # +1 for inclusive
                cycle_durations.append(cycle_duration)
    
    if not cycle_durations:
        return
    
    # Calculate figure size based on scale
    figsize = (10 * figsize_scale, 6 * figsize_scale)
    plt.figure(figsize=figsize)
    plt.hist(cycle_durations, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Cell Cycle Duration (frames)')
    plt.ylabel('Frequency')
    plt.title(f'Cell Cycle Duration Distribution (n={len(cycle_durations)})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logging.info(f"Cell cycle duration plot saved to {output_path}")


def create_track_length_distribution_plot(trj, output_path, dpi=300, figsize_scale=1.0):
    """Create a histogram of track lengths."""
    if trj is None or trj.empty:
        return
    
    import matplotlib.pyplot as plt
    
    # Calculate track lengths
    track_lengths = trj.groupby('particle')['frame'].nunique()
    
    # Calculate figure size based on scale
    figsize = (10 * figsize_scale, 6 * figsize_scale)
    plt.figure(figsize=figsize)
    plt.hist(track_lengths.values, bins=20, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Track Length (frames)')
    plt.ylabel('Frequency')
    plt.title(f'Track Length Distribution (n={len(track_lengths)})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logging.info(f"Track length distribution plot saved to {output_path}")


def create_cell_density_heatmap(trj, id_masks, output_path, dpi=300, figsize_scale=1.0):
    """Create a heatmap showing cell density over time and space."""
    if trj is None or trj.empty or id_masks is None:
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get dimensions
    height, width, num_frames = id_masks.shape
    
    # Create density matrix
    density_matrix = np.zeros((height, width))
    
    # Sum all frames to get overall density
    for frame in range(num_frames):
        frame_mask = id_masks[:, :, frame]
        # Count non-zero pixels (cells) in this frame
        density_matrix += (frame_mask > 0).astype(float)
    
    # Normalize by number of frames
    density_matrix /= num_frames
    
    # Calculate figure size based on scale
    figsize = (12 * figsize_scale, 8 * figsize_scale)
    plt.figure(figsize=figsize)
    plt.imshow(density_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Average Cell Density (cells/pixel)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Cell Density Heatmap (Average over all frames)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logging.info(f"Cell density heatmap saved to {output_path}")


def create_movement_vector_field(trj, output_path, num_frames_sample=10, dpi=300, figsize_scale=1.0):
    """Create a vector field showing average cell movement patterns."""
    if trj is None or trj.empty:
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Sample frames for vector field
    all_frames = sorted(trj['frame'].unique())
    if len(all_frames) <= num_frames_sample:
        sample_frames = all_frames
    else:
        step = len(all_frames) // num_frames_sample
        sample_frames = all_frames[::step][:num_frames_sample]
    
    # Calculate figure size based on scale
    figsize = (12 * figsize_scale, 8 * figsize_scale)
    plt.figure(figsize=figsize)
    
    for frame in sample_frames:
        # Get positions for this frame
        frame_data = trj[trj['frame'] == frame]
        if frame_data.empty:
            continue
        
        # Calculate movement vectors to next frame
        next_frame = frame + 1
        next_frame_data = trj[trj['frame'] == next_frame]
        
        for _, row in frame_data.iterrows():
            particle_id = row['particle']
            # Fix coordinate inversion: swap x and y
            y1, x1 = row['y'], row['x']  # y is first dimension, x is second dimension
            
            # Find same particle in next frame
            next_row = next_frame_data[next_frame_data['particle'] == particle_id]
            if not next_row.empty:
                y2, x2 = next_row.iloc[0]['y'], next_row.iloc[0]['x']  # Fix coordinate inversion
                dx, dy = x2 - x1, y2 - y1
                
                # Plot vector
                plt.arrow(x1, y1, dx, dy, head_width=2, head_length=2, 
                         fc='blue', ec='blue', alpha=0.6, length_includes_head=True)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Cell Movement Vector Field')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logging.info(f"Movement vector field saved to {output_path}")


def create_spatial_clustering_analysis(trj, output_path, frame_sample=None, dpi=300, figsize_scale=1.0):
    """Create a plot showing spatial clustering of cells."""
    if trj is None or trj.empty:
        return
    
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    # Use a specific frame or the middle frame
    if frame_sample is None:
        frame_sample = trj['frame'].median()
    
    frame_data = trj[trj['frame'] == frame_sample]
    if frame_data.empty:
        return
    
    # Get positions
    positions = frame_data[['x', 'y']].values
    
    if len(positions) < 2:
        return
    
    # Calculate pairwise distances
    distances = pdist(positions)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distances, method='ward')
    
    # Calculate figure size based on scale
    figsize = (12 * figsize_scale, 8 * figsize_scale)
    plt.figure(figsize=figsize)
    
    # Plot dendrogram
    plt.subplot(1, 2, 1)
    dendrogram(linkage_matrix, labels=frame_data['particle'].values, leaf_rotation=90)
    plt.title(f'Spatial Clustering Dendrogram (Frame {frame_sample})')
    plt.xlabel('Cell ID')
    plt.ylabel('Distance')
    
    # Plot spatial positions with cluster colors
    plt.subplot(1, 2, 2)
    from scipy.cluster.hierarchy import fcluster
    clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')  # 3 clusters
    
    scatter = plt.scatter(frame_data['x'], frame_data['y'], c=clusters, cmap='tab10', s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Spatial Clustering (Frame {frame_sample})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logging.info(f"Spatial clustering analysis saved to {output_path}")


def create_state_transition_diagram(trj, output_path, dpi=300, figsize_scale=1.0):
    """Create a state transition diagram showing how cells change states over time."""
    if trj is None or trj.empty or 'state' not in trj.columns:
        return
    
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # Create transition matrix
    states = trj['state'].unique()
    state_to_idx = {state: i for i, state in enumerate(states)}
    
    # Count transitions
    transitions = {}
    for particle_id in trj['particle'].unique():
        particle_data = trj[trj['particle'] == particle_id].sort_values('frame')
        for i in range(len(particle_data) - 1):
            from_state = particle_data.iloc[i]['state']
            to_state = particle_data.iloc[i + 1]['state']
            if from_state != to_state:
                key = (from_state, to_state)
                transitions[key] = transitions.get(key, 0) + 1
    
    if not transitions:
        return
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    for state in states:
        G.add_node(state)
    
    # Add edges with weights
    for (from_state, to_state), weight in transitions.items():
        G.add_edge(from_state, to_state, weight=weight)
    
    # Calculate figure size based on scale
    figsize = (12 * figsize_scale, 8 * figsize_scale)
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000)
    nx.draw_networkx_labels(G, pos)
    
    # Draw edges with weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='red', 
                          arrows=True, arrowsize=20)
    
    # Add edge labels
    edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title('State Transition Diagram')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logging.info(f"State transition diagram saved to {output_path}")


def create_state_distribution_over_time(trj, output_path, dpi=300, figsize_scale=1.0):
    """Create a plot showing state distribution over time."""
    if trj is None or trj.empty or 'state' not in trj.columns:
        return
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Count states per frame
    state_counts = trj.groupby(['frame', 'state']).size().unstack(fill_value=0)
    
    # Calculate figure size based on scale
    figsize = (12 * figsize_scale, 6 * figsize_scale)
    plt.figure(figsize=figsize)
    state_counts.plot(kind='area', stacked=True, alpha=0.7)
    plt.xlabel('Frame')
    plt.ylabel('Number of Cells')
    plt.title('State Distribution Over Time')
    plt.legend(title='State')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logging.info(f"State distribution over time plot saved to {output_path}")


def create_ensemble_outline_polar_plot(trj, id_masks, output_path, num_angle_bins=36, time_bins=None, dpi=300, figsize_scale=1.0):
    """
    Create a polar plot showing ensemble cell outline changes over time.
    
    Coordinate system:
    - 0° points to the right (positive x-axis)
    - 90° points up (positive y-axis) 
    - 180° points left (negative x-axis)
    - 270° points down (negative y-axis)
    
    Args:
        trj: Trajectory dataframe
        id_masks: 3D array of cell masks
        output_path: Output file path
        num_angle_bins: Number of angle bins for polar plot (default 36 = 10° each)
        time_bins: List of frame numbers to sample (if None, uses all frames)
    """
    if trj is None or trj.empty or id_masks is None:
        logging.warning("Cannot create ensemble outline polar plot: missing or empty trajectory/mask data")
        return
    
    if id_masks.ndim != 3:
        logging.error(f"Invalid id_masks shape: {id_masks.shape}, expected 3D array")
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.measure import find_contours
    from scipy.spatial.distance import cdist
    
    # Get dimensions
    height, width, num_frames = id_masks.shape
    
    # Determine time bins
    if time_bins is None:
        # Sample frames evenly
        if num_frames <= 20:
            time_bins = list(range(num_frames))
        else:
            step = num_frames // 20
            time_bins = list(range(0, num_frames, step))
    
    # Create angle bins (0 to 2π, centered on each bin)
    angles = np.linspace(0, 2*np.pi, num_angle_bins + 1, endpoint=True)  # 0 to 2π, including 2π
    angle_bin_centers = angles[:-1] + (2*np.pi / num_angle_bins) / 2  # Center of each bin
    
    # Initialize ensemble outline data
    ensemble_outlines = np.zeros((len(time_bins), num_angle_bins))
    cell_counts = np.zeros(len(time_bins))
    
    for t_idx, frame in enumerate(time_bins):
        if frame >= num_frames:
            continue
            
        frame_mask = id_masks[:, :, frame]
        frame_cells = trj[trj['frame'] == frame]
        
        if frame_cells.empty:
            continue
        
        cell_outlines = []
        
        # Process each cell in this frame
        for _, cell_row in frame_cells.iterrows():
            cell_id = cell_row['particle']
            cell_mask = (frame_mask == cell_id)
            
            if not np.any(cell_mask):
                continue
            
            # Find contours
            contours = find_contours(cell_mask, 0.5)
            if not contours:
                continue
            
            # Use the largest contour
            largest_contour = max(contours, key=len)
            
            # Center the contour
            center_y, center_x = np.mean(largest_contour, axis=0)
            centered_contour = largest_contour - [center_y, center_x]
            
            # Convert to polar coordinates
            # Note: contour[:, 0] is y, contour[:, 1] is x in image coordinates
            x_coords = centered_contour[:, 1]  # x coordinates (column)
            y_coords = centered_contour[:, 0]  # y coordinates (row)
            
            # Calculate polar coordinates (r, theta)
            r_coords = np.sqrt(x_coords**2 + y_coords**2)
            theta_coords = np.arctan2(y_coords, x_coords)
            
            # Normalize angles to [0, 2π)
            theta_coords = np.mod(theta_coords, 2*np.pi)
            
            # Bin the radius values by angle
            angle_binned_radii = np.zeros(num_angle_bins)
            angle_counts = np.zeros(num_angle_bins)
            
            for i, (r, theta) in enumerate(zip(r_coords, theta_coords)):
                # Find which angle bin this point belongs to
                # Normalize theta to [0, 2π) and find corresponding bin
                angle_bin = int(theta / (2*np.pi) * num_angle_bins)
                angle_bin = min(angle_bin, num_angle_bins - 1)  # Ensure within bounds
                
                angle_binned_radii[angle_bin] += r
                angle_counts[angle_bin] += 1
            
            # Average radius for each angle bin
            valid_bins = angle_counts > 0
            if np.any(valid_bins):
                angle_binned_radii[valid_bins] /= angle_counts[valid_bins]
                cell_outlines.append(angle_binned_radii)
        
        if cell_outlines:
            # Average across all cells in this frame
            ensemble_outlines[t_idx] = np.mean(cell_outlines, axis=0)
            cell_counts[t_idx] = len(cell_outlines)
    
    # Create the polar plot
    figsize = (12 * figsize_scale, 10 * figsize_scale)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)
    
    # Create meshgrid for plotting
    theta_mesh, time_mesh = np.meshgrid(angles, time_bins)
    
    # Wrap the data for periodicity
    ensemble_outlines_wrapped = np.hstack([ensemble_outlines, ensemble_outlines[:, [0]]])
    
    # Plot as a filled contour
    contour = ax.contourf(theta_mesh, time_mesh, ensemble_outlines_wrapped, 
                         levels=20, cmap='viridis', alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label('Average Cell Radius (pixels)', rotation=270, labelpad=20)
    
    # Customize the plot
    ax.set_title('Ensemble Cell Outline Changes Over Time\n(Polar View)', 
                pad=20, fontsize=14, fontweight='bold')
    
    # Set angle labels (0° at right, 90° at top, 180° at left, 270° at bottom)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax.set_xticklabels(['0°', '90°', '180°', '270°'])
    
    # Set radial axis (time) labels
    if len(time_bins) > 10:
        # Show fewer time labels to avoid clutter
        step = len(time_bins) // 10
        time_ticks = time_bins[::step]
    else:
        time_ticks = time_bins
    
    ax.set_yticks(time_ticks)
    ax.set_yticklabels([f'Frame {t}' for t in time_ticks])
    ax.set_ylabel('Time (frames)')
    
    # Add statistics text
    avg_cells = np.mean(cell_counts[cell_counts > 0]) if np.any(cell_counts > 0) else 0
    stats_text = f'Average cells per frame: {avg_cells:.1f}\nTotal frames: {len(time_bins)}'
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Ensemble outline polar plot saved to {output_path}")


def create_ensemble_outline_heatmap(trj, id_masks, output_path, num_angle_bins=36, time_bins=None, dpi=300, figsize_scale=1.0):
    """
    Create a heatmap showing ensemble cell outline changes over time.
    Alternative to polar plot, showing the same data in rectangular format.
    
    Coordinate system:
    - X-axis: Angle (0° to 360°)
    - Y-axis: Time (early frames at top, late frames at bottom)
    - 0° points to the right (positive x-axis)
    - 90° points up (positive y-axis)
    - 180° points left (negative x-axis) 
    - 270° points down (negative y-axis)
    """
    if trj is None or trj.empty or id_masks is None:
        logging.warning("Cannot create ensemble outline heatmap: missing or empty trajectory/mask data")
        return
    
    if id_masks.ndim != 3:
        logging.error(f"Invalid id_masks shape: {id_masks.shape}, expected 3D array")
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.measure import find_contours
    
    # Get dimensions
    height, width, num_frames = id_masks.shape
    
    # Determine time bins
    if time_bins is None:
        # Sample frames evenly
        if num_frames <= 20:
            time_bins = list(range(num_frames))
        else:
            step = num_frames // 20
            time_bins = list(range(0, num_frames, step))
    
    # Create angle bins (0 to 2π, centered on each bin)
    angles = np.linspace(0, 2*np.pi, num_angle_bins + 1)[:-1]
    angle_degrees = np.degrees(angles)
    angle_bin_centers = angles + (2*np.pi / num_angle_bins) / 2  # Center of each bin
    
    # Initialize ensemble outline data
    ensemble_outlines = np.zeros((len(time_bins), num_angle_bins))
    cell_counts = np.zeros(len(time_bins))
    
    for t_idx, frame in enumerate(time_bins):
        if frame >= num_frames:
            continue
            
        frame_mask = id_masks[:, :, frame]
        frame_cells = trj[trj['frame'] == frame]
        
        if frame_cells.empty:
            continue
        
        cell_outlines = []
        
        # Process each cell in this frame
        for _, cell_row in frame_cells.iterrows():
            cell_id = cell_row['particle']
            cell_mask = (frame_mask == cell_id)
            
            if not np.any(cell_mask):
                continue
            
            # Find contours
            contours = find_contours(cell_mask, 0.5)
            if not contours:
                continue
            
            # Use the largest contour
            largest_contour = max(contours, key=len)
            
            # Center the contour
            center_y, center_x = np.mean(largest_contour, axis=0)
            centered_contour = largest_contour - [center_y, center_x]
            
            # Convert to polar coordinates
            # Note: contour[:, 0] is y, contour[:, 1] is x in image coordinates
            x_coords = centered_contour[:, 1]  # x coordinates (column)
            y_coords = centered_contour[:, 0]  # y coordinates (row)
            
            # Calculate polar coordinates (r, theta)
            r_coords = np.sqrt(x_coords**2 + y_coords**2)
            theta_coords = np.arctan2(y_coords, x_coords)
            theta_coords = np.mod(theta_coords, 2*np.pi)
            
            # Bin the radius values by angle
            angle_binned_radii = np.zeros(num_angle_bins)
            angle_counts = np.zeros(num_angle_bins)
            
            for r, theta in zip(r_coords, theta_coords):
                # Find which angle bin this point belongs to
                # Normalize theta to [0, 2π) and find corresponding bin
                angle_bin = int(theta / (2*np.pi) * num_angle_bins)
                angle_bin = min(angle_bin, num_angle_bins - 1)  # Ensure within bounds
                
                angle_binned_radii[angle_bin] += r
                angle_counts[angle_bin] += 1
            
            valid_bins = angle_counts > 0
            if np.any(valid_bins):
                angle_binned_radii[valid_bins] /= angle_counts[valid_bins]
                cell_outlines.append(angle_binned_radii)
        
        if cell_outlines:
            ensemble_outlines[t_idx] = np.mean(cell_outlines, axis=0)
            cell_counts[t_idx] = len(cell_outlines)
    
    # Create the heatmap
    figsize = (14 * figsize_scale, 8 * figsize_scale)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the heatmap
    # Note: imshow origin='upper' means time increases downward
    # Fix the extent: X-axis should be 0 to 360 degrees, Y-axis should be frame indices
    im = ax.imshow(ensemble_outlines, cmap='viridis', aspect='auto', 
                   extent=[0, 360, len(time_bins)-1, 0], origin='upper')
    
    # Customize the plot
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Time (Frames)', fontsize=12)
    ax.set_title('Ensemble Cell Outline Changes Over Time\n(Heatmap View)', 
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Cell Radius (pixels)')
    
    # Set angle ticks (X-axis) - show degrees from 0 to 360
    # The extent is [0, 360, len(time_bins)-1, 0], so we need to map tick positions correctly
    angle_step = max(1, num_angle_bins // 8)
    angle_tick_positions = np.linspace(0, 360, num_angle_bins + 1)[::angle_step]
    angle_labels = [f'{int(angle)}°' for angle in angle_tick_positions]
    ax.set_xticks(angle_tick_positions)
    ax.set_xticklabels(angle_labels)
    
    # Set time ticks (Y-axis) - show frame numbers
    if len(time_bins) > 10:
        step = len(time_bins) // 10
        time_ticks = list(range(0, len(time_bins), step))
    else:
        time_ticks = list(range(len(time_bins)))
    
    time_labels = [f'Frame {time_bins[i]}' for i in time_ticks]
    ax.set_yticks(time_ticks)
    ax.set_yticklabels(time_labels)
    
    # Ensure axes are visible and properly formatted
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Ensure time axis goes from early frames (top) to late frames (bottom)
    ax.invert_yaxis()
    
    # Add statistics
    avg_cells = np.mean(cell_counts[cell_counts > 0]) if np.any(cell_counts > 0) else 0
    stats_text = f'Average cells per frame: {avg_cells:.1f} | Total frames: {len(time_bins)}'
    plt.figtext(0.5, 0.005, stats_text, ha='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Ensemble outline heatmap saved to {output_path}")


def create_comprehensive_analysis_plots(trj, cell_lineage, ancestry, id_masks, output_dir, main_app_state=None, fast_mode=False):
    """Create all comprehensive analysis plots and save them to the output directory.
    
    Args:
        trj: Trajectory DataFrame
        cell_lineage: Cell lineage dictionary
        ancestry: Ancestry dictionary
        id_masks: ID masks array
        output_dir: Output directory path
        main_app_state: Main application state
        fast_mode: If True, use fast mode for plot generation (reduces quality but much faster)
    """
    if trj is None or trj.empty:
        logging.warning("Cannot create analysis plots: trajectory is empty")
        return
    
    logging.info("Creating comprehensive analysis plots...")
    
    # Create analysis plots directory
    analysis_dir = os.path.join(output_dir, 'analysis_plots')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Get DPI from main app state or use defaults
    if main_app_state and 'params' in main_app_state:
        dpi = main_app_state['params'].get('Figure DPI', (300,))[0]
    else:
        dpi = 300
    
    # Fast mode optimizations
    if fast_mode:
        logging.info("Using fast mode for analysis plots generation")
        figsize_scale = 0.7  # Reduce figure sizes
    else:
        figsize_scale = 1.0
    
    try:
        # Cell count over time
        if (main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Cell Count Plot', (True,))[0]):
            create_cell_count_over_time_plot(trj, os.path.join(analysis_dir, 'cell_count_over_time.png'), dpi=dpi, figsize_scale=figsize_scale)
        
        # Event timeline
        if (main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Event Timeline Plot', (True,))[0]):
            create_event_timeline_plot(trj, cell_lineage, ancestry, os.path.join(analysis_dir, 'event_timeline.png'), dpi=dpi, figsize_scale=figsize_scale)
        
        # Cell cycle duration
        if (main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Cell Cycle Duration Plot', (True,))[0]):
            create_cell_cycle_duration_plot(trj, cell_lineage, os.path.join(analysis_dir, 'cell_cycle_duration.png'), dpi=dpi, figsize_scale=figsize_scale)
        
        # Track length distribution
        if (main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Track Length Distribution Plot', (True,))[0]):
            create_track_length_distribution_plot(trj, os.path.join(analysis_dir, 'track_length_distribution.png'), dpi=dpi, figsize_scale=figsize_scale)
        
        # Cell density heatmap
        if (main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Cell Density Heatmap', (True,))[0]):
            create_cell_density_heatmap(trj, id_masks, os.path.join(analysis_dir, 'cell_density_heatmap.png'), dpi=dpi, figsize_scale=figsize_scale)
        
        # Movement vector field
        if (main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Movement Vector Field', (True,))[0]):
            create_movement_vector_field(trj, os.path.join(analysis_dir, 'movement_vector_field.png'), dpi=dpi, figsize_scale=figsize_scale)
        
        # Spatial clustering
        if (main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Spatial Clustering Plot', (True,))[0]):
            create_spatial_clustering_analysis(trj, os.path.join(analysis_dir, 'spatial_clustering.png'), dpi=dpi, figsize_scale=figsize_scale)
        
        # Ensemble outline plots
        if (main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Ensemble Outline Plots', (True,))[0]):
            create_ensemble_outline_polar_plot(trj, id_masks, os.path.join(analysis_dir, 'ensemble_outline_polar.png'), dpi=dpi, figsize_scale=figsize_scale)
            create_ensemble_outline_heatmap(trj, id_masks, os.path.join(analysis_dir, 'ensemble_outline_heatmap.png'), dpi=dpi, figsize_scale=figsize_scale)
        
        # Publication lineage tree plots (PNG only)
        if (main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Publication Lineage Trees', (True,))[0]):
            create_publication_lineage_tree_plots(trj, id_masks, analysis_dir, cell_lineage, ancestry, dpi=dpi, figsize_scale=figsize_scale)
        
        # Cell distribution stacked bar plots
        if (main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Cell Distribution Stacked Bar Plot', (True,))[0]):
            categories = main_app_state['params'].get('Cell Distribution Plot Categories', ('both',))[0]
            create_cell_distribution_stacked_bar_plot(trj, cell_lineage, ancestry, 
                                                    os.path.join(analysis_dir, 'cell_distribution_stacked_bar.png'), 
                                                    categories=categories, dpi=dpi, figsize_scale=figsize_scale)
        
        # Simplified cell distribution plot
        if (main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Simplified Cell Distribution Plot', (True,))[0]):
            create_simplified_cell_distribution_plot(trj, cell_lineage, ancestry, 
                                                   os.path.join(analysis_dir, 'cell_distribution_simplified.png'), dpi=dpi, figsize_scale=figsize_scale)
        
        # Enhanced cell distribution plots (separate charts for track type and state with percentages)
        if (main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Enhanced Cell Distribution Plots', (True,))[0]):
            create_enhanced_cell_distribution_plots(trj, cell_lineage, ancestry, analysis_dir, dpi=dpi, figsize_scale=figsize_scale)
        
        # Phylogenetic tree plots (matplotlib only)
        if (main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save Phylogenetic Tree Plots', (True,))[0]):
            # Use matplotlib implementation directly
            _create_publication_lineage_tree_plot_matplotlib(trj, id_masks, os.path.join(analysis_dir, 'phylogenetic_tree_track_segments.png'), 
                                                           cell_lineage, ancestry, view_type='track_segments', dpi=dpi, figsize=(14 * figsize_scale, 10 * figsize_scale))
            _create_publication_lineage_tree_plot_matplotlib(trj, id_masks, os.path.join(analysis_dir, 'phylogenetic_tree_class_type.png'), 
                                                           cell_lineage, ancestry, view_type='class_type', dpi=dpi, figsize=(14 * figsize_scale, 10 * figsize_scale))
        
        # State transition diagram (if states exist)
        if ('state' in trj.columns and main_app_state and 'params' in main_app_state and 
            main_app_state['params'].get('Save State Analysis Plots', (True,))[0]):
            create_state_transition_diagram(trj, os.path.join(analysis_dir, 'state_transition_diagram.png'), dpi=dpi, figsize_scale=figsize_scale)
            create_state_distribution_over_time(trj, os.path.join(analysis_dir, 'state_distribution_over_time.png'), dpi=dpi, figsize_scale=figsize_scale)
        
        logging.info(f"All analysis plots saved to {analysis_dir}")
        
    except Exception as e:
        logging.error(f"Error creating analysis plots: {e}")
        import traceback
        logging.error(traceback.format_exc())


def create_gif_overlay_id_based(raw_images, id_masks, trj, output_path, alpha=0.6, fps=10, cell_visibility=None):
    """
    Create a GIF overlay showing cell IDs overlaid on raw images.
    Each cell ID gets a unique color.
    
    Args:
        raw_images: 3D numpy array (H, W, T) of raw images
        id_masks: 3D numpy array (H, W, T) of cell ID masks
        trj: DataFrame with trajectory data
        output_path: Path to save the GIF
        alpha: Transparency of the overlay (0-1)
        fps: Frames per second for the GIF
        cell_visibility: Dictionary mapping cell IDs to visibility (True/False)
    """
    if raw_images is None or id_masks is None or trj is None:
        logging.error("Cannot create GIF overlay: missing input data")
        return
    
    if raw_images.shape != id_masks.shape:
        logging.error(f"Shape mismatch: raw_images {raw_images.shape} vs id_masks {id_masks.shape}")
        return
    
    H, W, T = raw_images.shape
    
    # Get unique cell IDs for coloring, but filter by visibility
    unique_ids = np.unique(id_masks)
    unique_ids = unique_ids[unique_ids != 0]  # Remove background
    
    # Filter visible cells if cell_visibility is provided
    if cell_visibility is not None:
        visible_ids = [cell_id for cell_id in unique_ids if cell_visibility.get(cell_id, True)]
        logging.info(f"GIF overlay: Filtering {len(unique_ids)} total cells to {len(visible_ids)} visible cells")
        unique_ids = visible_ids
    else:
        logging.info(f"GIF overlay: No cell visibility filter provided, using all {len(unique_ids)} cells")
    
    if len(unique_ids) == 0:
        logging.warning("No visible cells found for GIF overlay")
        return
    
    # Generate colors for each ID
    colors = generate_distinct_colors(len(unique_ids), n_intensity_levels=3)
    id_to_color = {id_val: colors[i] for i, id_val in enumerate(unique_ids)}
    
    # Create frames for GIF
    frames = []
    
    for t in range(T):
        # Normalize raw image to 0-1 range first
        raw_frame = raw_images[:, :, t].astype(np.float32)
        if raw_frame.max() > raw_frame.min():
            raw_frame = (raw_frame - raw_frame.min()) / (raw_frame.max() - raw_frame.min())
        else:
            raw_frame = np.zeros_like(raw_frame, dtype=np.float32)
        
        # Convert to RGB (0-1 range)
        raw_rgb = np.stack([raw_frame] * 3, axis=-1)
        
        # Create overlay
        mask_frame = id_masks[:, :, t]
        overlay = raw_rgb.copy()
        
        for cell_id in unique_ids:
            if cell_id in mask_frame:
                color = id_to_color[cell_id]
                mask_region = (mask_frame == cell_id)
                for c in range(3):
                    overlay[mask_region, c] = (1 - alpha) * overlay[mask_region, c] + alpha * (color[c] / 255.0)
        
        # Convert to uint8 and add to frames
        frame_uint8 = img_as_ubyte(overlay)
        frames.append(frame_uint8)
    
    # Save GIF
    try:
        imageio.mimsave(output_path, frames, fps=fps)
        logging.info(f"Saved ID-based GIF overlay to {output_path}")
    except Exception as e:
        logging.error(f"Error saving GIF overlay: {e}")


def create_gif_overlay_state_based(raw_images, id_masks, trj, output_path, alpha=0.8, fps=10, cell_visibility=None):
    """
    Create a GIF overlay showing cell states overlaid on raw images.
    Each state class gets the same color. Missing states are assigned "unknown".
    
    Args:
        raw_images: 3D numpy array (H, W, T) of raw images
        id_masks: 3D numpy array (H, W, T) of cell ID masks
        trj: DataFrame with trajectory data (state column will be created if missing)
        output_path: Path to save the GIF
        alpha: Transparency of the overlay (0-1)
        fps: Frames per second for the GIF
        cell_visibility: Dictionary mapping cell IDs to visibility (True/False)
    """
    if raw_images is None or id_masks is None or trj is None:
        logging.error("Cannot create GIF overlay: missing input data")
        return
    
    if raw_images.shape != id_masks.shape:
        logging.error(f"Shape mismatch: raw_images {raw_images.shape} vs id_masks {id_masks.shape}")
        return
    
    H, W, T = raw_images.shape
    
    # Handle state information - create state column if missing
    trj_copy = trj.copy()
    if 'state' not in trj_copy.columns:
        logging.info("No 'state' column found in trajectory data. Creating state column with 'unknown' values.")
        trj_copy['state'] = 'unknown'
    else:
        # Fill missing state values with "unknown"
        trj_copy['state'] = trj_copy['state'].fillna("unknown")
    
    # Get unique states for coloring
    unique_states = trj_copy['state'].unique()
    if len(unique_states) == 0:
        logging.warning("No valid states found in trajectory data")
        return
    
    logging.info(f"Creating state-based GIF overlay with {len(unique_states)} unique states: {unique_states}")
    
    # Generate colors for each state with better distinction for common cell states
    state_to_color = generate_state_specific_colors(unique_states)
    logging.info(f"State to color mapping: {state_to_color}")
    
    # Create frames for GIF
    frames = []
    
    for t in range(T):
        # Normalize raw image to 0-1 range first
        raw_frame = raw_images[:, :, t].astype(np.float32)
        if raw_frame.max() > raw_frame.min():
            raw_frame = (raw_frame - raw_frame.min()) / (raw_frame.max() - raw_frame.min())
        else:
            raw_frame = np.zeros_like(raw_frame, dtype=np.float32)
        
        # Convert to RGB (0-1 range)
        raw_rgb = np.stack([raw_frame] * 3, axis=-1)
        
        # Create overlay
        mask_frame = id_masks[:, :, t]
        overlay = raw_rgb.copy()
        
        # Get trajectory data for this frame
        frame_trj = trj_copy[trj_copy['frame'] == t]
        
        cells_processed = 0
        for _, row in frame_trj.iterrows():
            cell_id = row['particle']
            state = row['state']
            
            # Check cell visibility if provided
            if cell_visibility is not None and not cell_visibility.get(cell_id, True):
                continue  # Skip invisible cells
            
            if pd.notna(cell_id) and cell_id in mask_frame:
                color = state_to_color[state]
                mask_region = (mask_frame == cell_id)
                # Use stronger color blending for better visibility
                for c in range(3):
                    # Blend with higher weight for the state color
                    overlay[mask_region, c] = (1 - alpha) * overlay[mask_region, c] + alpha * (color[c] / 255.0)
                    # Ensure minimum brightness for visibility
                    overlay[mask_region, c] = np.maximum(overlay[mask_region, c], 0.1)
                cells_processed += 1
        
        if t < 3:  # Log first 3 frames
            logging.info(f"Frame {t}: Processed {cells_processed} cells with states: {frame_trj['state'].unique()}")
            # Log detailed state information for debugging
            for _, row in frame_trj.iterrows():
                cell_id = row['particle']
                state = row['state']
                if pd.notna(cell_id) and cell_id in mask_frame:
                    color = state_to_color[state]
                    logging.info(f"  Cell {cell_id}: state='{state}', color={color}")
        
        # Convert to uint8 and add to frames
        frame_uint8 = img_as_ubyte(overlay)
        frames.append(frame_uint8)
    
    # Add legend overlay to each frame
    frames_with_legend = []
    for frame in frames:
        frame_with_legend = add_legend_overlay_to_frame(frame, unique_states, state_to_color)
        frames_with_legend.append(frame_with_legend)
    
    # Save GIF
    try:
        imageio.mimsave(output_path, frames_with_legend, fps=fps)
        logging.info(f"Saved state-based GIF overlay with legend overlay to {output_path}")
        
        # Also create a high-contrast version for better visibility
        high_contrast_path = output_path.replace('.gif', '_high_contrast.gif')
        frames_high_contrast = []
        
        for t in range(T):
            # Normalize raw image to 0-1 range first
            raw_frame = raw_images[:, :, t].astype(np.float32)
            if raw_frame.max() > raw_frame.min():
                raw_frame = (raw_frame - raw_frame.min()) / (raw_frame.max() - raw_frame.min())
            else:
                raw_frame = np.zeros_like(raw_frame, dtype=np.float32)
            
            # Convert to RGB (0-1 range)
            raw_rgb = np.stack([raw_frame] * 3, axis=-1)
            
            # Create high contrast overlay
            mask_frame = id_masks[:, :, t]
            overlay_hc = raw_rgb.copy()
            
            # Get trajectory data for this frame
            frame_trj = trj_copy[trj_copy['frame'] == t]
            
            for _, row in frame_trj.iterrows():
                cell_id = row['particle']
                state = row['state']
                
                # Check cell visibility if provided
                if cell_visibility is not None and not cell_visibility.get(cell_id, True):
                    continue
                
                if pd.notna(cell_id) and cell_id in mask_frame:
                    color = state_to_color[state]
                    mask_region = (mask_frame == cell_id)
                    # Use much stronger color blending for high contrast version
                    for c in range(3):
                        overlay_hc[mask_region, c] = 0.2 * overlay_hc[mask_region, c] + 0.8 * (color[c] / 255.0)
            
            # Convert to uint8 and add to frames
            frame_uint8 = img_as_ubyte(overlay_hc)
            frames_high_contrast.append(frame_uint8)
        
        # Add legend to high contrast frames
        frames_hc_with_legend = []
        for frame in frames_high_contrast:
            frame_with_legend = add_legend_overlay_to_frame(frame, unique_states, state_to_color)
            frames_hc_with_legend.append(frame_with_legend)
        
        imageio.mimsave(high_contrast_path, frames_hc_with_legend, fps=fps)
        logging.info(f"Saved high-contrast state-based GIF overlay to {high_contrast_path}")
        
    except Exception as e:
        logging.error(f"Error saving GIF overlay: {e}")


def add_legend_overlay_to_frame(frame, unique_states, state_to_color):
    """
    Add a legend overlay to a single frame at the bottom.
    
    Args:
        frame: numpy array of the frame (H, W, 3)
        unique_states: List of unique state names
        state_to_color: Dictionary mapping state names to RGB color tuples
    
    Returns:
        numpy array of the frame with legend overlay
    """
    H, W = frame.shape[:2]
    frame_with_legend = frame.copy()
    
    # Calculate legend dimensions
    legend_height = 60
    legend_y_start = H - legend_height - 20  # 20 pixels from bottom
    legend_x_start = 20  # 20 pixels from left
    
    # Create semi-transparent background for legend
    legend_bg = np.full((legend_height, W-40, 3), 240, dtype=np.uint8)  # Light gray background
    # Make it semi-transparent by blending with original frame
    alpha = 0.8
    frame_with_legend[legend_y_start:legend_y_start+legend_height, legend_x_start:legend_x_start+W-40] = \
        (legend_bg * alpha + frame[legend_y_start:legend_y_start+legend_height, legend_x_start:legend_x_start+W-40] * (1-alpha)).astype(np.uint8)
    
    # Draw legend border
    border_color = (128, 128, 128)
    border_thickness = 2
    # Top border
    frame_with_legend[legend_y_start:legend_y_start+border_thickness, legend_x_start:legend_x_start+W-40] = border_color
    # Bottom border
    frame_with_legend[legend_y_start+legend_height-border_thickness:legend_y_start+legend_height, legend_x_start:legend_x_start+W-40] = border_color
    # Left border
    frame_with_legend[legend_y_start:legend_y_start+legend_height, legend_x_start:legend_x_start+border_thickness] = border_color
    # Right border
    frame_with_legend[legend_y_start:legend_y_start+legend_height, legend_x_start+W-40-border_thickness:legend_x_start+W-40] = border_color
    
    # Add legend text using PIL
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_frame = Image.fromarray(frame_with_legend)
        draw = ImageDraw.Draw(pil_frame)
        
        # Try to use a larger font if available
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Draw title
        title_text = "Cell States"
        title_bbox = draw.textbbox((0, 0), title_text, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = legend_x_start + 10
        title_y = legend_y_start + 5
        draw.text((title_x, title_y), title_text, fill=(0, 0, 0), font=font)
        
        # Draw state color boxes and labels
        box_size = 16  # Increased box size for better visibility
        box_spacing = 120  # Increased spacing
        box_y = legend_y_start + 25
        max_boxes_per_row = (W - 80) // box_spacing  # Calculate how many boxes fit per row
        
        for i, state in enumerate(unique_states):
            row = i // max_boxes_per_row
            col = i % max_boxes_per_row
            box_x = legend_x_start + 10 + (col * box_spacing)
            box_y = legend_y_start + 25 + (row * 25)  # Increased row spacing
            
            # Check if we're still within the legend area
            if box_y + 25 > legend_y_start + legend_height:
                break
            
            # Draw color box with border for better visibility
            color = state_to_color[state]
            # Draw border first (black border)
            for y in range(box_y - 1, box_y + box_size + 1):
                for x in range(box_x - 1, box_x + box_size + 1):
                    if 0 <= y < H and 0 <= x < W:
                        if (y == box_y - 1 or y == box_y + box_size or 
                            x == box_x - 1 or x == box_x + box_size):
                            frame_with_legend[y, x] = (0, 0, 0)  # Black border
            
            # Draw colored box
            for y in range(box_y, box_y + box_size):
                for x in range(box_x, box_x + box_size):
                    if 0 <= y < H and 0 <= x < W:
                        frame_with_legend[y, x] = color
            
            # Draw state name
            state_text = str(state)
            draw.text((box_x + box_size + 5, box_y + 2), state_text, fill=(0, 0, 0), font=font)
        
        frame_with_legend = np.array(pil_frame)
        
    except ImportError:
        # Fallback if PIL is not available - just draw colored boxes
        logging.warning("PIL not available for legend text. Using basic color boxes only.")
        box_size = 16  # Increased box size for better visibility
        box_spacing = 120  # Increased spacing
        box_y = legend_y_start + 25
        
        for i, state in enumerate(unique_states):
            box_x = legend_x_start + 10 + (i * box_spacing)
            if box_x + box_spacing > W - 40:  # Wrap to next line if needed
                box_y += 25  # Increased row spacing
                box_x = legend_x_start + 10
            
            # Check if we're still within the legend area
            if box_y + 25 > legend_y_start + legend_height:
                break
            
            # Draw color box with border for better visibility
            color = state_to_color[state]
            # Draw border first (black border)
            for y in range(box_y - 1, box_y + box_size + 1):
                for x in range(box_x - 1, box_x + box_size + 1):
                    if 0 <= y < H and 0 <= x < W:
                        if (y == box_y - 1 or y == box_y + box_size or 
                            x == box_x - 1 or x == box_x + box_size):
                            frame_with_legend[y, x] = (0, 0, 0)  # Black border
            
            # Draw colored box
            for y in range(box_y, box_y + box_size):
                for x in range(box_x, box_x + box_size):
                    if 0 <= y < H and 0 <= x < W:
                        frame_with_legend[y, x] = color
    
    return frame_with_legend


def save_16bit_class_masks(id_masks, trj, output_dir, mask_prefix='class_mask'):
    """
    Save 16-bit masks for different classes (states) for each timeframe as PNG files.
    Missing state values are assigned "unknown" state.
    Args:
        id_masks: 3D numpy array (H, W, T) of cell ID masks
        trj: DataFrame with trajectory data including 'particle' column
        output_dir: Directory to save the masks
        mask_prefix: Prefix for mask filenames
    """
    if id_masks is None or trj is None:
        logging.error("Cannot save class masks: missing input data")
        return
    
    logging.info(f"Starting class mask generation. Trajectory shape: {trj.shape}, ID masks shape: {id_masks.shape}")
    logging.info(f"Available columns in trajectory: {list(trj.columns)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    H, W, T = id_masks.shape
    
    # Check if we have particle column
    if 'particle' not in trj.columns:
        logging.error("Cannot save class masks: 'particle' column not found in trajectory data")
        logging.info("Available columns: " + ", ".join(trj.columns))
        return
    
    # Handle state information - use "unknown" for missing values
    if 'state' in trj.columns:
        # Fill missing state values with "unknown"
        trj_copy = trj.copy()
        trj_copy['state'] = trj_copy['state'].fillna("unknown")
        unique_states = trj_copy['state'].unique()
        logging.info(f"Using state-based classification with 'unknown' for missing values. Found {len(unique_states)} unique states: {unique_states}")
        class_column = 'state'
        trj_to_use = trj_copy
    else:
        # No state column, use cell IDs as classes
        unique_states = trj['particle'].dropna().unique()
        logging.info(f"No state column found, using cell IDs as classes. Found {len(unique_states)} unique cells")
        class_column = 'particle'
        trj_to_use = trj
    
    if len(unique_states) == 0:
        logging.warning("No valid classes found in trajectory data")
        return
    
    # Create a mapping from state to class ID (1, 2, 3, ...)
    state_to_id = {state: i + 1 for i, state in enumerate(unique_states)}
    logging.info(f"State to ID mapping: {state_to_id}")
    
    masks_saved = 0
    for t in range(T):
        # Create class mask for this frame
        class_mask = np.zeros((H, W), dtype=np.uint16)
        
        # Get trajectory data for this frame
        frame_trj = trj_to_use[trj_to_use['frame'] == t]
        mask_frame = id_masks[:, :, t]
        
        cells_in_frame = 0
        for _, row in frame_trj.iterrows():
            cell_id = row['particle']
            state_value = row[class_column]
            
            if pd.notna(cell_id) and cell_id in mask_frame:
                class_id = state_to_id[state_value]
                class_mask[mask_frame == cell_id] = class_id
                cells_in_frame += 1
        
        if cells_in_frame > 0:
            # Save 16-bit mask as PNG
            mask_filename = f"{mask_prefix}_{t:03d}.png"
            mask_path = os.path.join(output_dir, mask_filename)
            
            try:
                imageio.imwrite(mask_path, class_mask.astype(np.uint16))
                masks_saved += 1
                logging.debug(f"Saved class mask for frame {t} with {cells_in_frame} cells")
            except Exception as e:
                logging.error(f"Error saving class mask for frame {t}: {e}")
    
    # Save class mapping
    mapping_filename = f"{mask_prefix}_class_mapping.txt"
    mapping_path = os.path.join(output_dir, mapping_filename)
    
    try:
        with open(mapping_path, 'w') as f:
            f.write("Class_ID\tState\n")
            for state_value, class_id in state_to_id.items():
                f.write(f"{class_id}\t{state_value}\n")
        logging.info(f"Saved class mapping to {mapping_path}")
    except Exception as e:
        logging.error(f"Error saving class mapping: {e}")
    
    logging.info(f"Saved {masks_saved} class masks to {output_dir}")


def create_publication_lineage_tree_plot(trj, id_masks, output_path, cell_lineage=None, ancestry=None,
                                        view_type='track_segments', line_color='black', line_width=2.0, 
                                        figsize=(14, 10), dpi=300):
    """
    Create publication-ready lineage tree plots using matplotlib.
    """
    return _create_publication_lineage_tree_plot_matplotlib(trj, id_masks, output_path, cell_lineage, ancestry,
                                                           view_type, line_color, line_width, figsize, dpi)








def create_publication_lineage_tree_plots(trj, id_masks, output_dir, cell_lineage=None, ancestry=None, dpi=300, figsize_scale=1.0):
    """
    Create both track segments and class type publication lineage tree plots using matplotlib (PNG only).
    
    Args:
        trj: Trajectory dataframe
        id_masks: 3D array of cell masks
        output_dir: Output directory path
        cell_lineage: Dictionary mapping parent IDs to list of child IDs
        ancestry: Dictionary mapping child IDs to list of parent IDs
    """
    if trj is None or trj.empty or id_masks is None:
        logging.warning("Cannot create publication lineage tree plots: missing or empty trajectory/mask data")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create track segments plot (single color) - PNG only
    track_segments_path = os.path.join(output_dir, 'lineage_tree_track_segments_publication.png')
    create_publication_lineage_tree_plot(
        trj, id_masks, track_segments_path, 
        cell_lineage=cell_lineage, ancestry=ancestry,
        view_type='track_segments', 
        line_color='black', 
        line_width=2.0,
        dpi=dpi,
        figsize=(14 * figsize_scale, 10 * figsize_scale)
    )
    
    # Create class type plot (colored by class) - PNG only
    class_type_path = os.path.join(output_dir, 'lineage_tree_class_type_publication.png')
    create_publication_lineage_tree_plot(
        trj, id_masks, class_type_path, 
        cell_lineage=cell_lineage, ancestry=ancestry,
        view_type='class_type', 
        line_width=2.0,
        dpi=dpi,
        figsize=(14 * figsize_scale, 10 * figsize_scale)
    )
    
    logging.info(f"Publication lineage tree plots saved to {output_dir} (PNG versions only)")


def _create_publication_lineage_tree_plot_matplotlib(trj, id_masks, output_path, cell_lineage=None, ancestry=None,
                                                    view_type='track_segments', line_color='black', line_width=2.0, 
                                                    figsize=(14, 10), dpi=300):
    """
    Fallback matplotlib-only version of the publication lineage tree plot.
    """
    if trj is None or trj.empty or id_masks is None:
        logging.warning("Cannot create publication lineage tree: missing or empty trajectory/mask data")
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    
    # Get dimensions
    height, width, num_frames = id_masks.shape
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    # Build lineage tree structure
    if cell_lineage is None or ancestry is None:
        logging.warning("No lineage data provided. Creating simple track segments plot.")
        # Fallback to simple track segments if no lineage data
        unique_tracks = trj['particle'].unique()
        track_y_positions = {}
        y_spacing = 2.0
        current_y = 0
        
        for track_id in sorted(unique_tracks):
            track_y_positions[track_id] = current_y
            current_y += y_spacing
        
        # Plot each track as simple horizontal line
        for track_id in unique_tracks:
            track_data = trj[trj['particle'] == track_id].sort_values('frame')
            if track_data.empty:
                continue
            
            frames = track_data['frame'].values
            y_pos = track_y_positions[track_id]
            color = line_color
            
            ax.plot(frames, [y_pos] * len(frames), color=color, linewidth=line_width, solid_capstyle='round')
            if len(frames) > 0:
                ax.text(frames[0] - 1, y_pos, str(track_id), 
                       fontsize=8, ha='right', va='center', color=color, weight='bold')
    else:
        # Create proper lineage tree with connections
        # Find root nodes (nodes with no parents)
        all_nodes = set(trj['particle'].unique())
        root_nodes = []
        
        for node in all_nodes:
            if node not in ancestry or not ancestry[node]:
                root_nodes.append(node)
        
        if not root_nodes:
            # If no clear roots, use nodes that appear earliest
            earliest_nodes = trj.groupby('particle')['frame'].min().nsmallest(5).index.tolist()
            root_nodes = earliest_nodes
        
        # Calculate y positions for tree layout
        track_y_positions = {}
        y_spacing = 3.0
        current_y = 0
        
        # Assign y positions starting from roots
        def assign_y_positions(node, visited=None):
            if visited is None:
                visited = set()
            
            if node in visited or node in track_y_positions:
                return
            
            visited.add(node)
            track_y_positions[node] = current_y
            
            # Recursively assign positions to children
            if node in cell_lineage:
                for child in cell_lineage[node]:
                    if child not in track_y_positions:
                        assign_y_positions(child, visited)
        
        # Assign positions for each root
        for root in root_nodes:
            assign_y_positions(root)
            current_y += y_spacing
        
        # For any remaining nodes, assign positions
        for node in all_nodes:
            if node not in track_y_positions:
                track_y_positions[node] = current_y
                current_y += y_spacing
        
        # For class type view, get state information
        state_colors = {}
        if view_type == 'class_type' and 'state' in trj.columns:
            unique_states = trj['state'].dropna().unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_states)))
            state_colors = {state: colors[i] for i, state in enumerate(unique_states)}
        
        # Plot track segments (horizontal lines)
        if view_type == 'track_segments':
            # All tracks in black (or specified color)
            for track_id in all_nodes:
                track_data = trj[trj['particle'] == track_id].sort_values('frame')
                if track_data.empty:
                    continue
                frames = track_data['frame'].values
                y_pos = track_y_positions[track_id]
                ax.plot(frames, [y_pos] * len(frames), color=line_color, linewidth=line_width, alpha=0.8, solid_capstyle='round')
        elif view_type == 'class_type' and 'state' in trj.columns:
            # Each track colored by its most common state
            for track_id in all_nodes:
                track_data = trj[trj['particle'] == track_id].sort_values('frame')
                if track_data.empty:
                    continue
                frames = track_data['frame'].values
                y_pos = track_y_positions[track_id]
                states = track_data['state'].dropna().values
                if len(states) > 0:
                    most_common_state = max(set(states), key=list(states).count)
                    color = state_colors.get(most_common_state, line_color)
                else:
                    color = line_color
                ax.plot(frames, [y_pos] * len(frames), color=color, linewidth=line_width, alpha=0.8, solid_capstyle='round')
        else:
            # Fallback: all tracks in black
            for track_id in all_nodes:
                track_data = trj[trj['particle'] == track_id].sort_values('frame')
                if track_data.empty:
                    continue
                frames = track_data['frame'].values
                y_pos = track_y_positions[track_id]
                ax.plot(frames, [y_pos] * len(frames), color=line_color, linewidth=line_width, alpha=0.8, solid_capstyle='round')
    
        # Plot connection lines between parents and children
        connector_color = 'gray'
        connector_style = '--'
        connector_width = 1.0
        
        for child_id, parents in ancestry.items():
            if child_id not in track_y_positions:
                continue
                
            child_data = trj[trj['particle'] == child_id].sort_values('frame')
            if child_data.empty:
                continue
            
            child_start_frame = child_data['frame'].min()
            child_y = track_y_positions[child_id]
            
            for parent_id in parents:
                if parent_id not in track_y_positions:
                    continue
                    
                parent_data = trj[trj['particle'] == parent_id].sort_values('frame')
                if parent_data.empty:
                    continue
                
                parent_end_frame = parent_data['frame'].max()
                parent_y = track_y_positions[parent_id]
                
                # Draw connector line from parent end to child start
                ax.plot([parent_end_frame, child_start_frame], [parent_y, child_y], 
                       color=connector_color, linestyle=connector_style, linewidth=connector_width, alpha=0.7)
    
    # Customize the plot
    ax.set_xlabel('Time (frames)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Track ID', fontsize=12, fontweight='bold')
    
    if view_type == 'track_segments':
        title = 'Lineage Tree - Track Segments'
    else:
        title = 'Lineage Tree - Class Type'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set axis limits with some padding
    all_frames = trj['frame'].values
    if len(all_frames) > 0:
        ax.set_xlim(min(all_frames) - 2, max(all_frames) + 2)
    
    if 'track_y_positions' in locals():
        ax.set_ylim(-1, max(track_y_positions.values()) + 1)
    
    # Remove y-axis ticks and labels for cleaner look
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='x')
    
    # Add legend for class type view
    if view_type == 'class_type' and 'state_colors' in locals() and state_colors:
        legend_elements = [plt.Line2D([0], [0], color=color, lw=line_width, label=state) 
                          for state, color in state_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Ensure tight layout
    plt.tight_layout()
    
    # Save with high quality - handle HTML files by converting to PNG
    if output_path.endswith('.html'):
        # Convert HTML path to PNG for matplotlib fallback
        png_path = output_path.replace('.html', '.png')
        plt.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        logging.info(f"HTML requested but matplotlib only supports PNG. Saved PNG instead: {png_path}")
    else:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()
    
    logging.info(f"Matplotlib lineage tree plot saved to {output_path} (view_type: {view_type})")
    
    return output_path

def save_time_based_offset_track_plot(trj, output_path, colormap_name='plasma', dpi=300):
    """
    Save a plot of all tracks offset so each cell's position at t=0 (first frame) is normalized to (0,0).
    Tracks are colored by elapsed frame (time) using a colormap.
    """
    if trj is None or trj.empty or 'x' not in trj.columns or 'y' not in trj.columns or 'frame' not in trj.columns or 'particle' not in trj.columns:
        return
    
    plt.figure(figsize=(8, 6))
    
    # Get the overall frame range for normalization
    all_frames = trj['frame'].unique()
    min_frame = all_frames.min()
    max_frame = all_frames.max()
    
    # Create colormap
    try:
        cmap = plt.cm.get_cmap(colormap_name)
    except ValueError:
        # Fallback to plasma if the specified colormap doesn't exist
        cmap = plt.cm.plasma
        print(f"Warning: Colormap '{colormap_name}' not found, using 'plasma' instead.")
    
    # Create a colorbar for reference
    norm = plt.Normalize(min_frame, max_frame)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    for pid, group in trj.groupby('particle'):
        if group.empty:
            continue
            
        # Sort by frame to ensure proper ordering
        group = group.sort_values('frame')
        
        # Get the first frame position (t=0) for this cell
        first_frame_data = group.iloc[0]
        first_x = first_frame_data['x']
        first_y = first_frame_data['y']
        
        # Offset all positions relative to this cell's first position
        xs = group['x'] - first_x
        ys = group['y'] - first_y
        frames = group['frame']
        
        if len(xs) == 0:
            continue
        
        # Plot each segment with color based on frame
        for i in range(len(xs) - 1):
            # Color based on the frame at the start of this segment
            frame_val = frames.iloc[i]
            color = cmap(norm(frame_val))
            
            # Plot line segment
            plt.plot(xs.iloc[i:i+2], ys.iloc[i:i+2], color=color, linewidth=1.5, alpha=0.8)
            
            # Add markers at each point
            plt.plot(xs.iloc[i], ys.iloc[i], 'o', color=color, markersize=3, alpha=0.8)
        
        # Add marker for the last point
        if len(xs) > 0:
            last_frame = frames.iloc[-1]
            last_color = cmap(norm(last_frame))
            plt.plot(xs.iloc[-1], ys.iloc[-1], 'o', color=last_color, markersize=3, alpha=0.8)
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('x axis [μm]')
    plt.ylabel('y axis [μm]')
    plt.title(f'Time-based Offset Track Plot\nNumber of tracks: {len(trj["particle"].unique())}')
    
    # Add colorbar
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
    cbar.set_label('Frame Number', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def create_cell_distribution_stacked_bar_plot(trj, cell_lineage, ancestry, output_path, categories='all', dpi=300, figsize_scale=1.0):
    """
    Create a comprehensive stacked bar plot showing cell distribution across multiple categories over time.
    
    Args:
        trj: Trajectory DataFrame with columns ['particle', 'frame', 'x', 'y', 'state']
        cell_lineage: Dictionary mapping parent IDs to lists of daughter IDs
        ancestry: Dictionary mapping child IDs to lists of parent IDs
        output_path: Output file path for the plot
        categories: String specifying which categories to include ('all', 'track_type', 'state', 'both')
    """
    if trj is None or trj.empty:
        return
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Determine which categories to include
    include_track_type = categories in ['all', 'track_type', 'both']
    include_state = categories in ['all', 'state', 'both']
    
    if not include_track_type and not include_state:
        logging.warning("No valid categories specified for cell distribution plot")
        return
    
    # Get all unique frames
    frames = sorted(trj['frame'].unique())
    
    # Initialize data structures for plotting
    plot_data = {}
    category_names = []
    
    if include_track_type:
        # Categorize tracks by type
        all_track_ids = set(trj['particle'].unique())
        
        # Identify mitosis and fusion related IDs
        mitosis_parents = {int(p_id) for p_id, d_ids in cell_lineage.items() if len(d_ids) >= 2}
        mitosis_daughters = {int(d_id) for d_ids in cell_lineage.values() if len(d_ids) >= 2 for d_id in d_ids}
        fusion_children = {int(c_id) for c_id, p_ids in ancestry.items() if len(p_ids) >= 2}
        fusion_parents = {int(p_id) for p_ids in ancestry.values() if len(p_ids) >= 2 for p_id in p_ids}
        
        event_related_ids = mitosis_parents.union(mitosis_daughters).union(fusion_children).union(fusion_parents)
        singular_ids = all_track_ids - event_related_ids
        
        # Create track type mapping
        track_type_map = {}
        for track_id in all_track_ids:
            if track_id in mitosis_parents:
                track_type_map[track_id] = 'Mitosis Parent'
            elif track_id in mitosis_daughters:
                track_type_map[track_id] = 'Mitosis Daughter'
            elif track_id in fusion_children:
                track_type_map[track_id] = 'Fusion Child'
            elif track_id in fusion_parents:
                track_type_map[track_id] = 'Fusion Parent'
            else:
                track_type_map[track_id] = 'Singular'
        
        # Add track type to trajectory data
        trj_with_track_type = trj.copy()
        trj_with_track_type['track_type'] = trj_with_track_type['particle'].map(track_type_map)
        
        # Count track types per frame
        track_type_counts = trj_with_track_type.groupby(['frame', 'track_type']).size().unstack(fill_value=0)
        
        # Add track type data to plot
        for track_type in track_type_counts.columns:
            category_name = f"Track Type: {track_type}"
            category_names.append(category_name)
            plot_data[category_name] = track_type_counts[track_type].reindex(frames, fill_value=0).values
    
    if include_state:
        # Count states per frame (if state column exists)
        if 'state' in trj.columns:
            state_counts = trj.groupby(['frame', 'state']).size().unstack(fill_value=0)
            
            # Add state data to plot
            for state in state_counts.columns:
                category_name = f"State: {state}"
                category_names.append(category_name)
                plot_data[category_name] = state_counts[state].reindex(frames, fill_value=0).values
        else:
            logging.warning("No 'state' column found in trajectory data for state-based distribution")
    
    if not plot_data:
        logging.warning("No data available for cell distribution plot")
        return
    
    # Create the stacked bar plot
    figsize = (14 * figsize_scale, 8 * figsize_scale)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for stacking
    categories_list = list(plot_data.keys())
    data_matrix = np.array([plot_data[cat] for cat in categories_list])
    
    # Create color palette
    if len(categories_list) <= 10:
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories_list)))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, len(categories_list)))
    
    # Create stacked bars
    bottom = np.zeros(len(frames))
    bars = []
    
    for i, (category, color) in enumerate(zip(categories_list, colors)):
        bar = ax.bar(frames, data_matrix[i], bottom=bottom, 
                    label=category, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        bars.append(bar)
        bottom += data_matrix[i]
    
    # Customize the plot
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Number of Cells', fontsize=12)
    ax.set_title('Cell Distribution Over Time', fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Customize legend
    if len(categories_list) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        # For many categories, create a more compact legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    
    # Adjust layout to accommodate legend
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Cell distribution stacked bar plot saved to {output_path}")
    
    # Log summary statistics
    total_cells = sum(data_matrix.sum() for data_matrix in plot_data.values())
    logging.info(f"Cell distribution plot summary: {len(categories_list)} categories, {len(frames)} frames, {total_cells:.0f} total cell observations")


def create_enhanced_cell_distribution_plots(trj, cell_lineage, ancestry, output_dir, dpi=300, figsize_scale=1.0):
    """
    Create enhanced cell distribution plots with separate charts for track type and state,
    including both absolute counts and percentages.
    
    Args:
        trj: Trajectory DataFrame with columns ['particle', 'frame', 'x', 'y', 'state']
        cell_lineage: Dictionary mapping parent IDs to lists of daughter IDs
        ancestry: Dictionary mapping child IDs to lists of parent IDs
        output_dir: Output directory for the plots
    """
    if trj is None or trj.empty:
        return
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    
    # Get all unique frames
    frames = sorted(trj['frame'].unique())
    
    # Create track type data
    all_track_ids = set(trj['particle'].unique())
    
    # Identify mitosis and fusion related IDs
    mitosis_parents = {int(p_id) for p_id, d_ids in cell_lineage.items() if len(d_ids) >= 2}
    mitosis_daughters = {int(d_id) for d_ids in cell_lineage.values() if len(d_ids) >= 2 for d_id in d_ids}
    fusion_children = {int(c_id) for c_id, p_ids in ancestry.items() if len(p_ids) >= 2}
    fusion_parents = {int(p_id) for p_ids in ancestry.values() if len(p_ids) >= 2 for p_id in p_ids}
    
    event_related_ids = mitosis_parents.union(mitosis_daughters).union(fusion_children).union(fusion_parents)
    singular_ids = all_track_ids - event_related_ids
    
    # Create track type mapping
    track_type_map = {}
    for track_id in all_track_ids:
        if track_id in mitosis_parents:
            track_type_map[track_id] = 'Mitosis Parent'
        elif track_id in mitosis_daughters:
            track_type_map[track_id] = 'Mitosis Daughter'
        elif track_id in fusion_children:
            track_type_map[track_id] = 'Fusion Child'
        elif track_id in fusion_parents:
            track_type_map[track_id] = 'Fusion Parent'
        else:
            track_type_map[track_id] = 'Singular'
    
    # Add track type to trajectory data
    trj_with_track_type = trj.copy()
    trj_with_track_type['track_type'] = trj_with_track_type['particle'].map(track_type_map)
    
    # Count track types per frame
    track_type_counts = trj_with_track_type.groupby(['frame', 'track_type']).size().unstack(fill_value=0)
    
    # Create Track Type Distribution Plot (Absolute Counts)
    figsize = (14 * figsize_scale, 8 * figsize_scale)
    fig, ax = plt.subplots(figsize=figsize)
    
    colors_track = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#FFD93D', '#6C5CE7']  # Green, Red, Cyan, Yellow, Purple
    bottom = np.zeros(len(frames))
    
    for i, track_type in enumerate(track_type_counts.columns):
        data = track_type_counts[track_type].reindex(frames, fill_value=0).values
        ax.bar(frames, data, bottom=bottom, label=track_type, 
               color=colors_track[i % len(colors_track)], alpha=0.8, edgecolor='black', linewidth=0.5)
        bottom += data
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Number of Cells', fontsize=12)
    ax.set_title('Cell Distribution by Track Type (Absolute Counts)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cell_distribution_track_type_counts.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # Create Track Type Distribution Plot (Percentages)
    figsize = (14 * figsize_scale, 8 * figsize_scale)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate percentages
    track_type_percentages = track_type_counts.div(track_type_counts.sum(axis=1), axis=0) * 100
    
    bottom = np.zeros(len(frames))
    
    for i, track_type in enumerate(track_type_counts.columns):
        data = track_type_percentages[track_type].reindex(frames, fill_value=0).values
        ax.bar(frames, data, bottom=bottom, label=track_type, 
               color=colors_track[i % len(colors_track)], alpha=0.8, edgecolor='black', linewidth=0.5)
        bottom += data
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Percentage of Cells (%)', fontsize=12)
    ax.set_title('Cell Distribution by Track Type (Percentages)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cell_distribution_track_type_percentages.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # Create State Distribution Plot (if state data exists)
    if 'state' in trj.columns:
        state_counts = trj.groupby(['frame', 'state']).size().unstack(fill_value=0)
        
        # Create State Distribution Plot (Absolute Counts)
        figsize = (14 * figsize_scale, 8 * figsize_scale)
        fig, ax = plt.subplots(figsize=figsize)
        
        colors_state = plt.cm.Set3(np.linspace(0, 1, len(state_counts.columns)))
        bottom = np.zeros(len(frames))
        
        for i, state in enumerate(state_counts.columns):
            data = state_counts[state].reindex(frames, fill_value=0).values
            ax.bar(frames, data, bottom=bottom, label=state, 
                   color=colors_state[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            bottom += data
        
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Number of Cells', fontsize=12)
        ax.set_title('Cell Distribution by State (Absolute Counts)', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cell_distribution_state_counts.png'), dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # Create State Distribution Plot (Percentages)
        figsize = (14 * figsize_scale, 8 * figsize_scale)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate percentages
        state_percentages = state_counts.div(state_counts.sum(axis=1), axis=0) * 100
        
        bottom = np.zeros(len(frames))
        
        for i, state in enumerate(state_counts.columns):
            data = state_percentages[state].reindex(frames, fill_value=0).values
            ax.bar(frames, data, bottom=bottom, label=state, 
                   color=colors_state[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            bottom += data
        
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Percentage of Cells (%)', fontsize=12)
        ax.set_title('Cell Distribution by State (Percentages)', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cell_distribution_state_percentages.png'), dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logging.info(f"State distribution plots saved to {output_dir}")
    else:
        logging.warning("No 'state' column found in trajectory data for state-based distribution")
    
    # Create Combined Summary Plot
    figsize = (16 * figsize_scale, 12 * figsize_scale)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Track Type Counts
    bottom = np.zeros(len(frames))
    for i, track_type in enumerate(track_type_counts.columns):
        data = track_type_counts[track_type].reindex(frames, fill_value=0).values
        ax1.bar(frames, data, bottom=bottom, label=track_type, 
               color=colors_track[i % len(colors_track)], alpha=0.8, edgecolor='black', linewidth=0.5)
        bottom += data
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title('Track Type Distribution (Counts)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Track Type Percentages
    bottom = np.zeros(len(frames))
    for i, track_type in enumerate(track_type_counts.columns):
        data = track_type_percentages[track_type].reindex(frames, fill_value=0).values
        ax2.bar(frames, data, bottom=bottom, label=track_type, 
               color=colors_track[i % len(colors_track)], alpha=0.8, edgecolor='black', linewidth=0.5)
        bottom += data
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Track Type Distribution (Percentages)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)
    
    # Plot 3: State Counts (if available)
    if 'state' in trj.columns:
        bottom = np.zeros(len(frames))
        for i, state in enumerate(state_counts.columns):
            data = state_counts[state].reindex(frames, fill_value=0).values
            ax3.bar(frames, data, bottom=bottom, label=state, 
                   color=colors_state[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            bottom += data
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Number of Cells')
        ax3.set_title('State Distribution (Counts)', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: State Percentages
        bottom = np.zeros(len(frames))
        for i, state in enumerate(state_counts.columns):
            data = state_percentages[state].reindex(frames, fill_value=0).values
            ax4.bar(frames, data, bottom=bottom, label=state, 
                   color=colors_state[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            bottom += data
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Percentage (%)')
        ax4.set_title('State Distribution (Percentages)', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 100)
    else:
        # If no state data, show total cell count and track type summary
        total_counts = trj.groupby('frame')['particle'].nunique()
        ax3.plot(frames, [total_counts.get(frame, 0) for frame in frames], 
                'o-', linewidth=2, markersize=4, color='blue')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Total Number of Cells')
        ax3.set_title('Total Cell Count Over Time', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Summary statistics
        track_type_summary = track_type_counts.sum()
        ax4.pie(track_type_summary.values, labels=track_type_summary.index, autopct='%1.1f%%')
        ax4.set_title('Overall Track Type Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cell_distribution_summary.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Enhanced cell distribution plots saved to {output_dir}")


def create_simplified_cell_distribution_plot(trj, cell_lineage, ancestry, output_path, dpi=300, figsize_scale=1.0):
    """
    Create a simplified cell distribution plot focusing on the most important categories.
    This version is more readable when there are many categories.
    """
    if trj is None or trj.empty:
        return
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Get all unique frames
    frames = sorted(trj['frame'].unique())
    
    # Categorize tracks by type (simplified)
    all_track_ids = set(trj['particle'].unique())
    
    # Identify mitosis and fusion related IDs
    mitosis_parents = {int(p_id) for p_id, d_ids in cell_lineage.items() if len(d_ids) >= 2}
    mitosis_daughters = {int(d_id) for d_ids in cell_lineage.values() if len(d_ids) >= 2 for d_id in d_ids}
    fusion_children = {int(c_id) for c_id, p_ids in ancestry.items() if len(p_ids) >= 2}
    fusion_parents = {int(p_id) for p_ids in ancestry.values() if len(p_ids) >= 2 for p_id in p_ids}
    
    event_related_ids = mitosis_parents.union(mitosis_daughters).union(fusion_children).union(fusion_parents)
    singular_ids = all_track_ids - event_related_ids
    
    # Create simplified track type mapping
    track_type_map = {}
    for track_id in all_track_ids:
        if track_id in mitosis_parents or track_id in mitosis_daughters:
            track_type_map[track_id] = 'Mitosis'
        elif track_id in fusion_children or track_id in fusion_parents:
            track_type_map[track_id] = 'Fusion'
        else:
            track_type_map[track_id] = 'Singular'
    
    # Add track type to trajectory data
    trj_with_track_type = trj.copy()
    trj_with_track_type['track_type'] = trj_with_track_type['particle'].map(track_type_map)
    
    # Count track types per frame
    track_type_counts = trj_with_track_type.groupby(['frame', 'track_type']).size().unstack(fill_value=0)
    
    # Create the plot
    figsize = (14 * figsize_scale, 10 * figsize_scale)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Track Type Distribution
    colors_track = ['#2E8B57', '#FF6B6B', '#4ECDC4']  # Green, Red, Cyan
    bottom = np.zeros(len(frames))
    
    for i, track_type in enumerate(['Singular', 'Mitosis', 'Fusion']):
        if track_type in track_type_counts.columns:
            data = track_type_counts[track_type].reindex(frames, fill_value=0).values
            ax1.bar(frames, data, bottom=bottom, label=track_type, 
                   color=colors_track[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            bottom += data
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title('Cell Distribution by Track Type', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: State Distribution (if available)
    if 'state' in trj.columns:
        state_counts = trj.groupby(['frame', 'state']).size().unstack(fill_value=0)
        
        # Limit to top 5 states for readability
        top_states = state_counts.sum().nlargest(5).index
        state_counts_filtered = state_counts[top_states]
        
        colors_state = plt.cm.Set3(np.linspace(0, 1, len(top_states)))
        bottom = np.zeros(len(frames))
        
        for i, state in enumerate(top_states):
            data = state_counts_filtered[state].reindex(frames, fill_value=0).values
            ax2.bar(frames, data, bottom=bottom, label=state, 
                   color=colors_state[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            bottom += data
        
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Number of Cells')
        ax2.set_title('Cell Distribution by State (Top 5)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        # If no state data, show total cell count over time
        total_counts = trj.groupby('frame')['particle'].nunique()
        ax2.plot(frames, [total_counts.get(frame, 0) for frame in frames], 
                'o-', linewidth=2, markersize=4, color='blue')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Total Number of Cells')
        ax2.set_title('Total Cell Count Over Time', fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Simplified cell distribution plot saved to {output_path}")


def _save_temporal_outline_stack_fast(id_masks, output_path, colormap_name="plasma", alpha=0.4, dpi=300, scale_factor=3, figsize_scale=1.0):
    """
    ULTRA-FAST version of temporal outline stack generation for very large datasets.
    This version sacrifices some visual quality for significant speed improvements by:
    - Using simplified contour detection
    - Reducing glow effects
    - Using lower resolution for large datasets
    - Minimizing memory allocations
    - Using more efficient data structures
    
    Args:
        id_masks: 3D numpy array (H, W, T) of cell ID masks
        output_path: Path to save the output PNG
        colormap_name: Name of the matplotlib colormap to use
        alpha: Transparency of the outline overlays (0-1)
        dpi: DPI for the output image (higher = sharper)
        scale_factor: Factor to scale up the image resolution (higher = larger, sharper image)
        figsize_scale: Scale factor for image dimensions (higher = larger image)
    """
    H, W, T = id_masks.shape
    
    # Apply figsize_scale to scale_factor
    scale_factor = scale_factor * figsize_scale
    
    # For fast mode, limit scale factor to prevent memory issues
    scale_factor = min(scale_factor, 2)
    
    # For very large datasets, use even lower resolution
    if H * W * T > 50000000:  # 50M pixels threshold
        scale_factor = 1
        logging.info(f"Very large dataset detected ({H*W*T} pixels), using scale_factor=1 for fast mode")
    
    # Scale up the image dimensions
    H_scaled = int(H * scale_factor)
    W_scaled = int(W * scale_factor)
    
    # Start with white background
    outline_img = np.ones((H_scaled, W_scaled, 4), dtype=np.float32)
    outline_img[:, :, 3] = 1.0
    
    cmap = plt.get_cmap(colormap_name)
    
    # Pre-compute offsets for efficiency (simplified)
    main_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # Simple cross pattern
    
    # Process ALL frames (fixed: was sampling frames before)
    for t in range(T):
        mask = id_masks[:, :, t]
        unique_cells = np.unique(mask)
        unique_cells = unique_cells[unique_cells != 0]
        
        if len(unique_cells) == 0:
            continue
        
        color = cmap(t / max(T - 1, 1))
        
        # Process cells with simplified approach (but process all cells, not just first 50)
        for cell_id in unique_cells:
            binary = (mask == cell_id).astype(np.uint8)
            
            if not np.any(binary):
                continue
            
            # Use simplified contour detection
            try:
                contours = find_contours(binary, 0.5)
            except:
                continue
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                # Scale and simplify contour
                contour_scaled = contour * scale_factor
                contour_int = np.round(contour_scaled).astype(int)
                
                # Draw simplified outline (no glow, just main outline)
                # Fixed: process all contour points, not every other one
                for i in range(len(contour_int) - 1):
                    y0, x0 = contour_int[i]
                    y1, x1 = contour_int[i + 1]
                    
                    rr, cc = line(y0, x0, y1, x1)
                    
                    # Apply simple outline
                    for dy, dx in main_offsets:
                        rr_outline = rr + dy
                        cc_outline = cc + dx
                        
                        valid = (rr_outline >= 0) & (rr_outline < H_scaled) & (cc_outline >= 0) & (cc_outline < W_scaled)
                        rr_valid = rr_outline[valid]
                        cc_valid = cc_outline[valid]
                        
                        if len(rr_valid) > 0:
                            outline_img[rr_valid, cc_valid, :3] = color[:3]
                            outline_img[rr_valid, cc_valid, 3] = 0.8
                
                # Draw closing segment if contour is closed (like standard mode)
                if len(contour_int) > 2:
                    y0, x0 = contour_int[-1]
                    y1, x1 = contour_int[0]
                    
                    rr, cc = line(y0, x0, y1, x1)
                    
                    # Apply simple outline to closing segment
                    for dy, dx in main_offsets:
                        rr_outline = rr + dy
                        cc_outline = cc + dx
                        
                        valid = (rr_outline >= 0) & (rr_outline < H_scaled) & (cc_outline >= 0) & (cc_outline < W_scaled)
                        rr_valid = rr_outline[valid]
                        cc_valid = cc_outline[valid]
                        
                        if len(rr_valid) > 0:
                            outline_img[rr_valid, cc_valid, :3] = color[:3]
                            outline_img[rr_valid, cc_valid, 3] = 0.8
    
    outline_img_uint8 = (np.clip(outline_img, 0, 1) * 255).astype(np.uint8)
    
    # Create simplified colorbar
    colorbar_width = max(100, W_scaled // 15)
    
    # Simplified tick positions (but more comprehensive than before)
    if T <= 10:
        tick_positions = list(range(T))
        tick_labels = [f'{t}' for t in range(T)]
    elif T <= 50:
        tick_positions = list(range(0, T, 5))
        tick_labels = [f'{t}' for t in tick_positions]
    else:
        tick_positions = list(range(0, T, 10))
        tick_labels = [f'{t}' for t in tick_positions]
    
    if T-1 not in tick_positions:
        tick_positions.append(T-1)
        tick_labels.append(f'{T-1}')
    
    # Create simplified colorbar using PIL
    from PIL import Image, ImageDraw, ImageFont
    
    colorbar_pil = Image.new('RGBA', (colorbar_width, H_scaled), (255, 255, 255, 255))
    draw = ImageDraw.Draw(colorbar_pil)
    
    gradient_width = colorbar_width // 4
    gradient_x = 5
    
    # Draw gradient (process all rows, not every other one)
    for y in range(H_scaled):
        frame_ratio = 1.0 - (y / H_scaled)
        frame_num = frame_ratio * (T - 1)
        color = cmap(frame_num / max(T - 1, 1))
        color_rgb = tuple(int(c * 255) for c in color[:3])
        draw.line([(gradient_x, y), (gradient_x + gradient_width, y)], fill=color_rgb, width=1)
    
    # Draw labels
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for pos, label in zip(tick_positions, tick_labels):
        frame_ratio = 1.0 - (pos / (T - 1))
        y_pos = int(frame_ratio * H_scaled)
        
        tick_x = gradient_x + gradient_width + 5
        draw.line([(tick_x, y_pos), (tick_x + 8, y_pos)], fill=(0, 0, 0, 255), width=2)
        draw.text((tick_x + 12, y_pos - 6), label, fill=(0, 0, 0, 255), font=font)
    
    # Convert to numpy and combine
    colorbar_img = np.array(colorbar_pil)
    
    if colorbar_img.shape[0] != H_scaled:
        from skimage.transform import resize
        colorbar_img = resize(colorbar_img, (H_scaled, colorbar_img.shape[1]), 
                            preserve_range=True, order=1).astype(np.uint8)  # Use order=1 for better quality
    
    # Add padding and combine
    padding_width = 5
    padding = np.ones((H_scaled, padding_width, 4), dtype=np.uint8) * 255
    padding[:, :, 3] = 255
    
    combined_img = np.concatenate([outline_img_uint8, padding, colorbar_img], axis=1)
    
    # Save with DPI information using PIL for better quality control
    from PIL import Image
    pil_img = Image.fromarray(combined_img)
    
    # Save with DPI information
    pil_img.save(output_path, dpi=(dpi, dpi), quality=95)
    
    logging.info(f"Saved fast temporal outline stack to {output_path} (fast_mode=True, scale_factor={scale_factor}, dpi={dpi}, figsize_scale={figsize_scale})")
