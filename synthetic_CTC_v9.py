# Author: Claude (Original) / Extended by Gemini / Enhanced by User Request
# Date: 2025-05-28
# Description: Generate synthetic cell tracking data with mitosis, fusion, irregular shapes,
#              tunable initial cell sparseness, configurable mitosis timing,
#              separate event-specific masks (configurable via params) using a dictionary-based
#              approach for future expandability, and an all-objects instance mask (each object retaining its unique ID).
#              Cells in mitosis/fusion event masks are excluded from cell_id_mask during those event frames.
#              Fixes IndexError for float array indexing and NameError for tracking_df.
#              Added detailed comments for all parameters in main().
#              Added Cell Tracking Challenge (CTC) compliant output generation (TIFF masks and res_track.txt).
#              Added flexible output for user-specific software input (raw PNGs and 16-bit instance mask PNGs).
#              V6: Added creation of GT/SEG folder by copying TRA masks for SEG metric compatibility.
#              V7: Corrected user_sw_raw_output_dirname_pattern and user_sw_mask_output_dirname_pattern defaults.
#              V7.1: Changed GT tracking file name from res_track.txt to man_track.txt for CTC compliance.
#              V7.2: Corrected CTC GT path generation to avoid extra subdirectory.
#              V8.1 (This Update): Made parent assignment in fusion events deterministic.
#              V8.2 (This Update): Restored missing directory variable definitions to fix NameError.

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon as skimage_polygon, disk as skimage_disk
from skimage.io import imsave
import os
import pandas as pd
import random
import math  # For sqrt
import tifffile  # For saving 16-bit TIFF images for CTC
import traceback  # For better error reporting in visualization
import shutil  # For copying files for SEG GT


def _generate_polygon_vertices(center_x, center_y, avg_radius, num_vertices=8, irregularity_factor=0.3):
    """
    Generates vertices for an irregular polygon centered at (center_x, center_y).
    irregularity_factor (0 to ~0.9): 0 for near circle, higher for more irregular.
    """
    irregularity_factor = np.clip(irregularity_factor, 0, 0.9)
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    angle_segment = 2 * np.pi / num_vertices
    angular_perturbation_range = angle_segment * 0.4
    angles += np.random.uniform(-angular_perturbation_range, angular_perturbation_range, num_vertices)

    radial_perturbations = np.random.uniform(-irregularity_factor, irregularity_factor, num_vertices)
    radii = avg_radius * (1 + radial_perturbations)

    min_allowed_radius_proportional = avg_radius * 0.2
    radii = np.maximum(radii, min_allowed_radius_proportional)
    radii = np.maximum(radii, 2.0)  # Absolute minimum radius for any vertex point

    verts_x = center_x + radii * np.cos(angles)
    verts_y = center_y + radii * np.sin(angles)

    return np.column_stack((verts_x, verts_y))


def _schedule_mitosis_frame(cell_frame_start, n_frames, mitosis_timing_config, generation=0):
    """
    Enhanced mitosis scheduling with configurable timing intervals.
    """
    if generation == 0:
        min_lifespan = mitosis_timing_config.get('initial_min_lifespan', 15)
        max_lifespan = mitosis_timing_config.get('initial_max_lifespan', 40)
    else:
        min_lifespan = mitosis_timing_config.get('daughter_min_lifespan', 12)
        max_lifespan = mitosis_timing_config.get('daughter_max_lifespan', 35)

    min_daughter_lifespan = mitosis_timing_config.get('min_daughter_survival', 8)

    earliest_mitosis = cell_frame_start + min_lifespan
    latest_mitosis = min(
        cell_frame_start + max_lifespan,
        n_frames - 1 - min_daughter_lifespan
    )

    if earliest_mitosis > latest_mitosis:
        return None

    mitosis_frame = random.randint(earliest_mitosis, latest_mitosis)

    generation_delay = mitosis_timing_config.get('generation_delay_factor', 0)
    if generation_delay > 0:
        additional_delay = int(generation * generation_delay)
        mitosis_frame = min(mitosis_frame + additional_delay, latest_mitosis)

    return mitosis_frame


def generate_synthetic_dataset(
        output_path,
        ctc_output_path_base="CTC_GT_Data",
        ctc_sequence_name="01",
        user_sw_raw_output_dirname_pattern="{ctc_sequence_name}",
        user_sw_mask_output_dirname_pattern="{ctc_sequence_name}_mask_all",
        user_sw_raw_filename_prefix="frame_",
        user_sw_mask_filename_prefix="mask_",
        user_sw_output_file_extension="png",
        n_frames=60,
        image_size=512,
        initial_cell_behaviors=[(20, 0), (5, 1)],
        min_initial_separation=5,
        num_fusion_events=2,
        max_fusion_distance_factor=1.2,
        daughter_spread_factor=0.4,
        noise_level=0.1,
        speed=3.0,
        cell_avg_radius_range=(8, 15),
        cell_irregularity=0.4,
        num_polygon_vertices=8,
        random_seed=42,
        file_extension="png",
        max_placement_retries=100,
        mitosis_timing_config=None,
        event_mask_configs=None,
        create_gt_seg_folder=True
):
    """
    Generate a synthetic dataset with various outputs.
    """
    if mitosis_timing_config is None:
        mitosis_timing_config = {
            'initial_min_lifespan': 15, 'initial_max_lifespan': 40,
            'daughter_min_lifespan': 12, 'daughter_max_lifespan': 35,
            'min_daughter_survival': 8, 'generation_delay_factor': 2,
            'inter_division_interval': 5
        }

    if event_mask_configs is None:
        event_mask_configs = {"mitosis": 3, "fusion": 1}

    np.random.seed(random_seed)
    random.seed(random_seed)
    masked_event_types = list(event_mask_configs.keys())
    print(f"Event types with dedicated masks: {masked_event_types}")
    print(f"Configured event mask durations: {event_mask_configs}")

    # --- Original Output Directories ---
    raw_dir_original = os.path.join(output_path, "synthetic_cells_raw")
    cell_id_mask_dir_original = os.path.join(output_path, "synthetic_cells_mask_cell_id")
    all_objects_mask_dir_original = os.path.join(output_path, "synthetic_cells_mask_all_objects")
    event_mask_dirs_original = {}
    for event_name in masked_event_types:
        dir_path = os.path.join(output_path, f"synthetic_cells_mask_{event_name}")
        os.makedirs(dir_path, exist_ok=True)
        event_mask_dirs_original[event_name] = dir_path
    os.makedirs(raw_dir_original, exist_ok=True)
    os.makedirs(cell_id_mask_dir_original, exist_ok=True)
    os.makedirs(all_objects_mask_dir_original, exist_ok=True)

    # --- CTC Output Directories ---
    ctc_sequence_specific_gt_dir = os.path.join(output_path, ctc_output_path_base)
    ctc_tra_dir = os.path.join(ctc_sequence_specific_gt_dir, "TRA")
    os.makedirs(ctc_tra_dir, exist_ok=True)
    print(f"CTC compliant GT TRA masks and man_track.txt will be saved to: {ctc_tra_dir}")

    ctc_seg_dir = None
    if create_gt_seg_folder:
        ctc_seg_dir = os.path.join(ctc_sequence_specific_gt_dir, "SEG")
        os.makedirs(ctc_seg_dir, exist_ok=True)
        print(f"CTC compliant GT SEG masks will be created in: {ctc_seg_dir}")

    # --- User Software Specific Output Directories ---
    user_sw_raw_dirname = user_sw_raw_output_dirname_pattern.format(ctc_sequence_name=ctc_sequence_name)
    user_sw_raw_dir = os.path.join(output_path, user_sw_raw_dirname)
    user_sw_mask_dirname = user_sw_mask_output_dirname_pattern.format(ctc_sequence_name=ctc_sequence_name)
    user_sw_mask_dir = os.path.join(output_path, user_sw_mask_dirname)
    os.makedirs(user_sw_raw_dir, exist_ok=True)
    os.makedirs(user_sw_mask_dir, exist_ok=True)
    print(f"User software specific raw images will be saved to: {user_sw_raw_dir}")
    print(f"User software specific instance masks will be saved to: {user_sw_mask_dir}")

    cells = []
    next_id = 1
    tracking_data_for_csv = []
    mitosis_data_for_csv = []
    fusion_data_for_csv = []
    gt_track_entries = []
    lineage_last_division = {}
    placed_initial_cells_info = []
    initial_cells_generated_count = 0

    # --- Initial Cell Placement ---
    for count, potential in initial_cell_behaviors:
        for _ in range(count):
            placed_successfully = False
            for retry in range(max_placement_retries):
                border_margin = cell_avg_radius_range[1] + min_initial_separation + 5
                x_prop = np.random.uniform(border_margin, image_size - border_margin)
                y_prop = np.random.uniform(border_margin, image_size - border_margin)
                avg_radius_prop = np.random.uniform(cell_avg_radius_range[0], cell_avg_radius_range[1])
                is_too_close = False
                for ex_x, ex_y, ex_radius in placed_initial_cells_info:
                    distance = math.sqrt((x_prop - ex_x) ** 2 + (y_prop - ex_y) ** 2)
                    required_distance = avg_radius_prop + ex_radius + min_initial_separation
                    if distance < required_distance:
                        is_too_close = True
                        break
                if not is_too_close:
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = speed * np.cos(angle)
                    dy = speed * np.sin(angle)
                    cell_id = next_id
                    next_id += 1
                    mitosis_frame_val = None
                    if potential > 0:
                        mitosis_frame_val = _schedule_mitosis_frame(0, n_frames, mitosis_timing_config, generation=0)
                    initial_event_timers = {event_name: 0 for event_name in masked_event_types}
                    cells.append({
                        'id': cell_id, 'x': x_prop, 'y': y_prop, 'avg_radius': avg_radius_prop,
                        'dx': dx, 'dy': dy, 'active': True, 'frame_start': 0,
                        'frame_end': n_frames - 1,
                        'parent_id': 0,
                        'fused_from_ids': None,
                        'generation': 0, 'division_potential': potential,
                        'mitosis_at_frame': mitosis_frame_val,
                        'mask_value': cell_id,
                        'prev_x': x_prop, 'prev_y': y_prop,
                        'lineage_root': cell_id,
                        'event_mask_timers': initial_event_timers
                    })
                    placed_initial_cells_info.append((x_prop, y_prop, avg_radius_prop))
                    initial_cells_generated_count += 1
                    placed_successfully = True
                    break
            if not placed_successfully:
                print(
                    f"Warning: Could not place an initial cell (potential: {potential}) after {max_placement_retries} retries.")
    print(f"Successfully placed {initial_cells_generated_count} initial cells.")

    # --- Schedule Fusion Events ---
    fusion_scheduled_frames = {}
    if num_fusion_events > 0 and initial_cells_generated_count >= 2:
        possible_fusion_frames = list(range(5, n_frames - 10))
        if not possible_fusion_frames:
            print("Warning: n_frames too small for fusion. No fusions will be scheduled.")
            num_fusion_events = 0
        actual_fusion_events_scheduled = 0
        for _ in range(num_fusion_events):
            if not possible_fusion_frames: break
            chosen_frame = random.choice(possible_fusion_frames)
            fusion_scheduled_frames.setdefault(chosen_frame, 0)
            fusion_scheduled_frames[chosen_frame] += 1
            actual_fusion_events_scheduled += 1
            if len(possible_fusion_frames) > 1:
                try:
                    possible_fusion_frames.remove(chosen_frame)
                except ValueError:
                    pass
        if actual_fusion_events_scheduled < num_fusion_events:
            print(
                f"Warning: Could only schedule {actual_fusion_events_scheduled} fusions out of {num_fusion_events} requested.")
            num_fusion_events = actual_fusion_events_scheduled

    # --- Main Simulation Loop (Frame by Frame) ---
    for frame in range(n_frames):
        raw_image_original_format = np.zeros((image_size, image_size), dtype=np.uint8)
        cell_id_mask_image_original_format = np.zeros((image_size, image_size), dtype=np.uint16)
        all_objects_mask_image_for_user_and_original = np.zeros((image_size, image_size), dtype=np.uint16)
        ctc_tracking_mask_image = np.zeros((image_size, image_size), dtype=np.uint16)
        current_event_mask_images_original_format = {
            event_name: np.zeros((image_size, image_size), dtype=np.uint16)
            for event_name in masked_event_types
        }
        newly_created_this_frame_meta = []
        cells_to_inactivate_ids = set()

        # --- Fusion Event Handling ---
        if frame in fusion_scheduled_frames:
            num_fusions_to_attempt_this_frame = fusion_scheduled_frames[frame]
            fusions_completed_this_frame = 0
            active_candidates_for_fusion = []
            for idx, c_dict in enumerate(cells):
                if (c_dict['active'] and
                        frame >= c_dict['frame_start'] and frame <= c_dict['frame_end'] and
                        (c_dict['mitosis_at_frame'] != frame if c_dict['mitosis_at_frame'] is not None else True) and
                        c_dict['id'] not in cells_to_inactivate_ids):
                    active_candidates_for_fusion.append((idx, c_dict))
            if len(active_candidates_for_fusion) >= 2:
                potential_fusion_pairs = []
                for i in range(len(active_candidates_for_fusion)):
                    for j in range(i + 1, len(active_candidates_for_fusion)):
                        idx1, c1 = active_candidates_for_fusion[i]
                        idx2, c2 = active_candidates_for_fusion[j]
                        if c1['id'] == c2['id'] or c1['id'] in cells_to_inactivate_ids or c2[
                            'id'] in cells_to_inactivate_ids:
                            continue
                        dist = math.sqrt((c1['x'] - c2['x']) ** 2 + (c1['y'] - c2['y']) ** 2)
                        if dist <= (c1['avg_radius'] + c2['avg_radius']) * max_fusion_distance_factor:
                            potential_fusion_pairs.append(((idx1, c1), (idx2, c2)))
                random.shuffle(potential_fusion_pairs)
                for (idx1, c1), (idx2, c2) in potential_fusion_pairs:
                    if fusions_completed_this_frame >= num_fusions_to_attempt_this_frame:
                        break
                    if c1['id'] in cells_to_inactivate_ids or c2['id'] in cells_to_inactivate_ids:
                        continue

                    # --- DETERMINISTIC PARENT ASSIGNMENT ---
                    if c1['id'] > c2['id']:
                        c1, c2 = c2, c1
                        idx1, idx2 = idx2, idx1

                    if "fusion" in event_mask_configs:
                        cells[idx1]['event_mask_timers']['fusion'] = event_mask_configs["fusion"]
                        cells[idx2]['event_mask_timers']['fusion'] = event_mask_configs["fusion"]

                    cells_to_inactivate_ids.add(c1['id'])
                    cells_to_inactivate_ids.add(c2['id'])
                    fused_id = next_id
                    next_id += 1
                    new_x = (c1['x'] + c2['x']) / 2
                    new_y = (c1['y'] + c2['y']) / 2
                    new_avg_radius = np.sqrt(c1['avg_radius'] ** 2 + c2['avg_radius'] ** 2)
                    new_dx = (c1['dx'] + c2['dx']) / 2
                    new_dy = (c1['dy'] + c2['dy']) / 2
                    fused_cell_meta = {
                        'id': fused_id, 'x': new_x, 'y': new_y, 'avg_radius': new_avg_radius,
                        'dx': new_dx, 'dy': new_dy, 'active': True,
                        'frame_start': frame + 1,
                        'frame_end': n_frames - 1,
                        'parent_id': c1['id'],
                        'fused_from_ids': sorted([c1['id'], c2['id']]),
                        'generation': max(c1['generation'], c2['generation']) + 1,
                        'division_potential': 0,
                        'mitosis_at_frame': None,
                        'mask_value': fused_id,
                        'prev_x': new_x, 'prev_y': new_y,
                        'lineage_root': c1.get('lineage_root', c1['id']),
                        'event_mask_timers': {name: 0 for name in masked_event_types}
                    }
                    newly_created_this_frame_meta.append(fused_cell_meta)
                    fusion_data_for_csv.append({
                        'frame': frame, 'parent1_id': c1['id'], 'parent2_id': c2['id'],
                        'fused_cell_id': fused_id, 'fused_cell_frame_start': frame + 1
                    })
                    fusions_completed_this_frame += 1

        # --- Mitosis Event Handling ---
        for cell_idx, cell_dict_ref in enumerate(cells):
            if (cell_dict_ref['id'] not in cells_to_inactivate_ids and
                    cell_dict_ref['active'] and
                    frame >= cell_dict_ref['frame_start'] and frame <= cell_dict_ref['frame_end'] and
                    cell_dict_ref['division_potential'] > 0 and
                    cell_dict_ref['mitosis_at_frame'] == frame):
                lineage_root = cell_dict_ref.get('lineage_root', cell_dict_ref['id'])
                last_division_frame_for_lineage = lineage_last_division.get(lineage_root, -float('inf'))
                inter_division_interval_needed = mitosis_timing_config.get('inter_division_interval', 5)
                if frame - last_division_frame_for_lineage >= inter_division_interval_needed:
                    if "mitosis" in event_mask_configs:
                        cells[cell_idx]['event_mask_timers']['mitosis'] = event_mask_configs["mitosis"]
                    cells_to_inactivate_ids.add(cell_dict_ref['id'])
                    lineage_last_division[lineage_root] = frame
                    parent_cell_ref = cells[cell_idx]
                    daughter_potential = parent_cell_ref['division_potential'] - 1
                    base_angle = np.random.uniform(0, 2 * np.pi)
                    dist_from_parent = parent_cell_ref['avg_radius'] * daughter_spread_factor
                    daughter_ids_temp = []
                    for i in range(2):
                        curr_angle = base_angle + (i * np.pi) + np.random.uniform(-np.pi / 8, np.pi / 8)
                        new_x = parent_cell_ref['x'] + dist_from_parent * np.cos(curr_angle)
                        new_y = parent_cell_ref['y'] + dist_from_parent * np.sin(curr_angle)
                        d_move_angle = curr_angle + np.random.uniform(-np.pi / 16, np.pi / 16)
                        new_dx = speed * np.cos(d_move_angle)
                        new_dy = speed * np.sin(d_move_angle)
                        new_radius = parent_cell_ref['avg_radius'] * np.random.uniform(0.8, 0.9)
                        daughter_id = next_id
                        next_id += 1
                        daughter_ids_temp.append(daughter_id)
                        d_mitosis_frame = None
                        if daughter_potential > 0:
                            d_gen = parent_cell_ref['generation'] + 1
                            d_mitosis_frame = _schedule_mitosis_frame(
                                frame + 1, n_frames, mitosis_timing_config, generation=d_gen
                            )
                        daughter_cell_meta = {
                            'id': daughter_id, 'x': new_x, 'y': new_y, 'avg_radius': new_radius,
                            'dx': new_dx, 'dy': new_dy, 'active': True,
                            'frame_start': frame + 1, 'frame_end': n_frames - 1,
                            'parent_id': parent_cell_ref['id'],
                            'fused_from_ids': None,
                            'generation': parent_cell_ref['generation'] + 1,
                            'division_potential': daughter_potential,
                            'mitosis_at_frame': d_mitosis_frame,
                            'mask_value': daughter_id,
                            'prev_x': new_x, 'prev_y': new_y,
                            'lineage_root': lineage_root,
                            'event_mask_timers': {name: 0 for name in masked_event_types}
                        }
                        newly_created_this_frame_meta.append(daughter_cell_meta)
                    mitosis_data_for_csv.append({
                        'frame': frame, 'parent_id': parent_cell_ref['id'],
                        'daughter1_id': daughter_ids_temp[0], 'daughter2_id': daughter_ids_temp[1],
                        'daughters_frame_start': frame + 1
                    })
                else:
                    new_earliest_mitosis_frame = last_division_frame_for_lineage + inter_division_interval_needed
                    min_survival_for_daughters = mitosis_timing_config.get('min_daughter_survival', 8)
                    if new_earliest_mitosis_frame < n_frames - 1 - min_survival_for_daughters:
                        cells[cell_idx]['mitosis_at_frame'] = new_earliest_mitosis_frame
                    else:
                        cells[cell_idx]['mitosis_at_frame'] = None

        # --- Update Cell Positions and Draw ---
        for cell_idx, cell in enumerate(cells):
            is_generally_active_this_frame = (
                    cell['active'] and
                    frame >= cell['frame_start'] and
                    frame <= cell['frame_end'] and
                    not (cell['id'] in cells_to_inactivate_ids and cell['frame_end'] == frame)
            )
            is_final_active_frame_due_to_event = cell['id'] in cells_to_inactivate_ids and cell['frame_end'] == frame
            if cell['frame_start'] == frame:
                cells[cell_idx]['prev_x'], cells[cell_idx]['prev_y'] = cell['x'], cell['y']
            elif frame > cell['frame_start'] and cell['active'] and frame <= cell['frame_end']:
                cells[cell_idx]['prev_x'], cells[cell_idx]['prev_y'] = cell['x'], cell['y']
                new_x_tentative = cell['x'] + cell['dx'] + np.random.normal(0, noise_level * speed)
                new_y_tentative = cell['y'] + cell['dy'] + np.random.normal(0, noise_level * speed)
                if (new_x_tentative - cell['avg_radius'] < 0 and cell['dx'] < 0) or \
                        (new_x_tentative + cell['avg_radius'] >= image_size and cell['dx'] > 0):
                    cells[cell_idx]['dx'] *= -1
                    new_x_tentative = cell['x'] + cells[cell_idx]['dx']
                if (new_y_tentative - cell['avg_radius'] < 0 and cell['dy'] < 0) or \
                        (new_y_tentative + cell['avg_radius'] >= image_size and cell['dy'] > 0):
                    cells[cell_idx]['dy'] *= -1
                    new_y_tentative = cell['y'] + cells[cell_idx]['dy']
                cells[cell_idx]['x'] = np.clip(new_x_tentative, 0, image_size - 1)
                cells[cell_idx]['y'] = np.clip(new_y_tentative, 0, image_size - 1)
                current_angle = np.arctan2(cells[cell_idx]['dy'], cells[cell_idx]['dx'])
                current_angle += np.random.normal(0, 0.15)
                cells[cell_idx]['dx'] = speed * np.cos(current_angle)
                cells[cell_idx]['dy'] = speed * np.sin(current_angle)
            should_render_this_cell = (
                    is_generally_active_this_frame or
                    is_final_active_frame_due_to_event or
                    any(timer > 0 for timer in cell.get('event_mask_timers', {}).values())
            )
            if should_render_this_cell:
                current_cell_x = cells[cell_idx]['x']
                current_cell_y = cells[cell_idx]['y']
                current_avg_radius = cells[cell_idx]['avg_radius']
                vertices = _generate_polygon_vertices(current_cell_x, current_cell_y, current_avg_radius,
                                                      num_vertices=num_polygon_vertices,
                                                      irregularity_factor=cell_irregularity)
                poly_r_float = np.clip(vertices[:, 1], 0, image_size - 1)
                poly_c_float = np.clip(vertices[:, 0], 0, image_size - 1)
                rr, cc = skimage_polygon(poly_r_float, poly_c_float, shape=(image_size, image_size))
                if rr.size == 0 or cc.size == 0:
                    center_x_draw = int(round(np.clip(current_cell_x, 0, image_size - 1)))
                    center_y_draw = int(round(np.clip(current_cell_y, 0, image_size - 1)))
                    fallback_radius = max(1, int(round(current_avg_radius * 0.75)))
                    rr_fallback, cc_fallback = skimage_disk((center_y_draw, center_x_draw), fallback_radius,
                                                            shape=(image_size, image_size))
                    rr, cc = rr_fallback, cc_fallback
                if rr.size > 0 and cc.size > 0:
                    is_in_mitosis_event_mask_phase = cell.get('event_mask_timers', {}).get("mitosis", 0) > 0
                    is_in_fusion_event_mask_phase = cell.get('event_mask_timers', {}).get("fusion", 0) > 0
                    if is_generally_active_this_frame or is_final_active_frame_due_to_event:
                        raw_image_original_format[rr, cc] = 200
                        all_objects_mask_image_for_user_and_original[rr, cc] = cell['mask_value']
                        ctc_tracking_mask_image[rr, cc] = cell['mask_value']
                        if not (is_in_mitosis_event_mask_phase or is_in_fusion_event_mask_phase):
                            cell_id_mask_image_original_format[rr, cc] = cell['mask_value']
                    for event_name, timer_value in cell.get('event_mask_timers', {}).items():
                        if timer_value > 0:
                            if event_name in current_event_mask_images_original_format:
                                current_event_mask_images_original_format[event_name][rr, cc] = cell['mask_value']
                            cells[cell_idx]['event_mask_timers'][event_name] -= 1
            if is_generally_active_this_frame or is_final_active_frame_due_to_event:
                tracking_data_for_csv.append({
                    'frame': frame, 'particle': cell['id'], 'y': cell['y'], 'x': cell['x'],
                    'avg_radius': cell['avg_radius'],
                    'parent_id_csv': cell['parent_id'],
                    'fused_from_id1': cell['fused_from_ids'][0] if cell['fused_from_ids'] else None,
                    'fused_from_id2': cell['fused_from_ids'][1] if cell['fused_from_ids'] else None,
                    'generation': cell['generation'],
                    'division_potential_remaining': cell['division_potential'],
                    'lineage_root': cell.get('lineage_root', cell['id'])
                })

        # --- Post-frame processing ---
        temp_cells_next_frame = []
        for idx, cell_dict_current_frame in enumerate(cells):
            if cell_dict_current_frame['id'] in cells_to_inactivate_ids:
                cells[idx]['active'] = False
                cells[idx]['frame_end'] = frame
                gt_track_entries.append({
                    'L': cells[idx]['id'],
                    'B': cells[idx]['frame_start'],
                    'E': cells[idx]['frame_end'],
                    'P': cells[idx]['parent_id']
                })
            is_needed_for_event_mask_countdown = any(
                timer > 0 for timer in cells[idx].get('event_mask_timers', {}).values())
            if cells[idx]['active'] or cells[idx]['frame_end'] >= frame or is_needed_for_event_mask_countdown:
                temp_cells_next_frame.append(cells[idx])
        cells = temp_cells_next_frame
        cells.extend(newly_created_this_frame_meta)

        if frame % 10 == 0 or frame == n_frames - 1:
            print(f"...completed frame {frame + 1}/{n_frames}")

        # --- Save images for the current frame ---
        imsave(os.path.join(raw_dir_original, f"frame_{frame:03d}.{file_extension}"), raw_image_original_format,
               check_contrast=False)
        imsave(os.path.join(cell_id_mask_dir_original, f"mask_cell_id_{frame:03d}.{file_extension}"),
               cell_id_mask_image_original_format.astype(np.uint16), check_contrast=False)
        imsave(os.path.join(all_objects_mask_dir_original, f"mask_all_objects_{frame:03d}.{file_extension}"),
               all_objects_mask_image_for_user_and_original.astype(np.uint16), check_contrast=False)
        for event_name, mask_image in current_event_mask_images_original_format.items():
            save_path = os.path.join(event_mask_dirs_original[event_name],
                                     f"mask_{event_name}_{frame:03d}.{file_extension}")
            imsave(save_path, mask_image.astype(np.uint16), check_contrast=False)

        ctc_tra_mask_filename = os.path.join(ctc_tra_dir, f"mask{frame:03d}.tif")
        tifffile.imwrite(ctc_tra_mask_filename, ctc_tracking_mask_image.astype(np.uint16), imagej=True)

        if create_gt_seg_folder and ctc_seg_dir:
            ctc_seg_mask_filename = os.path.join(ctc_seg_dir, f"man_seg{frame:03d}.tif")
            tifffile.imwrite(ctc_seg_mask_filename, ctc_tracking_mask_image.astype(np.uint16), imagej=True)

        user_sw_raw_img_filename = os.path.join(user_sw_raw_dir,
                                                f"{user_sw_raw_filename_prefix}{frame:03d}.{user_sw_output_file_extension}")
        imsave(user_sw_raw_img_filename, raw_image_original_format, check_contrast=False)

        user_sw_mask_filename = os.path.join(user_sw_mask_dir,
                                             f"{user_sw_mask_filename_prefix}{frame:03d}.{user_sw_output_file_extension}")
        imsave(user_sw_mask_filename, all_objects_mask_image_for_user_and_original.astype(np.uint16),
               check_contrast=False)

    # --- After all frames are processed ---
    for cell_dict_final in cells:
        if cell_dict_final['active']:
            already_added = any(
                entry['L'] == cell_dict_final['id'] for entry in gt_track_entries)
            if not already_added:
                gt_track_entries.append({
                    'L': cell_dict_final['id'],
                    'B': cell_dict_final['frame_start'],
                    'E': n_frames - 1,
                    'P': cell_dict_final['parent_id']
                })

    tracking_df_original = pd.DataFrame(tracking_data_for_csv)
    if not mitosis_data_for_csv:
        mitosis_df_original = pd.DataFrame(
            columns=['frame', 'parent_id', 'daughter1_id', 'daughter2_id', 'daughters_frame_start'])
    else:
        mitosis_df_original = pd.DataFrame(mitosis_data_for_csv)
    if not fusion_data_for_csv:
        fusion_df_original = pd.DataFrame(
            columns=['frame', 'parent1_id', 'parent2_id', 'fused_cell_id', 'fused_cell_frame_start'])
    else:
        fusion_df_original = pd.DataFrame(fusion_data_for_csv)

    tracking_df_original.to_csv(os.path.join(output_path, "tracking_ground_truth.csv"), index=False)
    mitosis_df_original.to_csv(os.path.join(output_path, "mitosis_ground_truth.csv"), index=False)
    fusion_df_original.to_csv(os.path.join(output_path, "fusion_ground_truth.csv"), index=False)

    # Save the GT tracking file as man_track.txt
    gt_track_file_path = os.path.join(ctc_tra_dir, "man_track.txt")
    with open(gt_track_file_path, 'w') as f_gt_track:
        for entry in sorted(gt_track_entries, key=lambda x: x['L']):
            f_gt_track.write(f"{entry['L']} {entry['B']} {entry['E']} {entry['P']}\n")
    print(f"CTC man_track.txt saved to: {gt_track_file_path}")

    print(f"\nDataset Generation Summary:")
    print(f"- Generated {n_frames} frames. Initial cells attempted: {sum(c for c, p in initial_cell_behaviors)}")
    print(f"- Mitosis events recorded (for CSV): {len(mitosis_data_for_csv)}")
    print(f"- Fusion events recorded (for CSV): {len(fusion_data_for_csv)}")
    print(f"- Total unique cell IDs generated: {next_id - 1}")
    print(f"- Original raw images saved to: {raw_dir_original}")
    print(f"- Original Cell ID masks saved to: {cell_id_mask_dir_original}")
    print(f"- Original All Objects (Instance) masks saved to: {all_objects_mask_dir_original}")
    for event_name, dir_path in event_mask_dirs_original.items():
        print(f"- Original '{event_name}' event masks saved to: {dir_path}")
    print(f"- CTC GT sequence directory: {ctc_sequence_specific_gt_dir}")  # Corrected variable
    print(f"  - CTC tracking TIFF masks saved to: {ctc_tra_dir}")
    if ctc_seg_dir:
        print(f"  - CTC segmentation TIFF masks saved to: {ctc_seg_dir}")
    print(f"  - CTC man_track.txt saved to: {ctc_tra_dir}")
    print(f"- User software specific raw images saved to: {user_sw_raw_dir}")
    print(f"- User software specific instance masks saved to: {user_sw_mask_dir}")

    return {
        'tracking_data_original': tracking_df_original,
        'mitosis_data_original': mitosis_df_original,
        'fusion_data_original': fusion_df_original,
        'gt_track_ctc_entries': gt_track_entries,
        'cells_final_state_count': len(cells),
        'mitosis_timing_config_used': mitosis_timing_config,
        'event_mask_configs_used': event_mask_configs,
        'user_sw_raw_dir_path': user_sw_raw_dir,
        'user_sw_mask_dir_path': user_sw_mask_dir,
        'ctc_tra_dir_path': ctc_tra_dir,
        'ctc_seg_dir_path': ctc_seg_dir
    }


def main():
    output_base_dir = "./synthetic_cell_dataset_ctc_v9"
    if os.path.exists(output_base_dir):
        # import shutil
        # shutil.rmtree(output_base_dir) # Comment out if you want to append or keep old data
        pass  # Keep existing directory if you want to add more sequences or outputs
    os.makedirs(output_base_dir, exist_ok=True)

    params = {
        'n_frames': 50,
        'image_size': 512,
        'initial_cell_behaviors': [(5, 0), (3, 1), (1, 2)],  # (count, division_potential)
        'min_initial_separation': 8,  # Min distance between centers of initially placed cells
        'num_fusion_events': 1,  # Number of fusion events to attempt to schedule
        'event_mask_configs': {"mitosis": 2, "fusion": 1},  # Duration (frames) for event-specific masks
        'mitosis_timing_config': {
            'initial_min_lifespan': 10,  # Min frames before first gen cell can divide
            'initial_max_lifespan': 25,  # Max frames before first gen cell must divide (if potential > 0)
            'daughter_min_lifespan': 8,  # Min frames before daughter cell can divide
            'daughter_max_lifespan': 20,  # Max frames before daughter cell must divide
            'min_daughter_survival': 5,  # Min frames a daughter must exist after mitosis for parent to divide
            'generation_delay_factor': 1,  # Additional frames delay per generation for mitosis
            'inter_division_interval': 4  # Min frames between divisions in the same lineage
        },
        'max_fusion_distance_factor': 1.1,  # Factor of sum of radii for fusion eligibility
        'daughter_spread_factor': 0.3,  # Factor of parent radius for daughter initial separation
        'noise_level': 0.1,  # Randomness in cell movement
        'speed': 1.8,  # Base speed of cells
        'cell_avg_radius_range': (7, 11),  # Range for average cell radius
        'cell_irregularity': 0.25,  # Factor for how irregular cell shapes are (0=circle)
        'num_polygon_vertices': 7,  # Number of vertices for cell polygons
        'random_seed': 2025,  # Seed for reproducibility
        'file_extension': "png",  # For original separate masks/raw images
        'max_placement_retries': 100,  # Retries for initial cell placement if too close

        # CTC Output Configuration:
        # This should be the name of the specific sequence's GT folder, e.g., "01_GT", "Synthetic_SEQ01_GT"
        # It will be created inside `output_base_dir`.
        'ctc_output_path_base': "Synthetic_SEQ01_GT",

        # Base name of the sequence, e.g., "01", "Synthetic_SEQ01".
        # Used for user-specific output folders and internal consistency.
        'ctc_sequence_name': "Synthetic_SEQ01",

        # User Software Specific Output (relative to output_base_dir):
        # Pattern for the directory holding raw images for your software.
        # {ctc_sequence_name} will be replaced by the value of 'ctc_sequence_name'.
        'user_sw_raw_output_dirname_pattern': "{ctc_sequence_name}",  # Default: e.g., "Synthetic_SEQ01"

        # Pattern for the directory holding instance masks for your software.
        'user_sw_mask_output_dirname_pattern': "{ctc_sequence_name}_mask_all",
        # Default: e.g., "Synthetic_SEQ01_mask_all"

        'user_sw_raw_filename_prefix': "frame_",  # Prefix for raw image files for your software
        'user_sw_mask_filename_prefix': "mask_",  # Prefix for mask files for your software
        'user_sw_output_file_extension': "png",  # File extension for user software images/masks

        'create_gt_seg_folder': True  # If True, creates GT/SEG folder by copying TRA masks
    }

    print(f"Starting dataset generation with CTC output and user-specific software output...")
    results = generate_synthetic_dataset(output_base_dir, **params)

    print(f"\n--- Output Check ---")
    print(f"Original tracking data rows: {len(results['tracking_data_original'])}")
    if not results['mitosis_data_original'].empty:
        print(f"Original mitosis events: {len(results['mitosis_data_original'])}")
    else:
        print("No mitosis events occurred (original CSV).")
    if not results['fusion_data_original'].empty:
        print(f"Original fusion events: {len(results['fusion_data_original'])}")
    else:
        print("No fusion events occurred (original CSV).")
    print(f"CTC man_track.txt entries: {len(results['gt_track_ctc_entries'])}")

    # --- Visualization of a sample frame from each output type ---
    frame_to_viz = params['n_frames'] // 2
    try:
        user_sw_raw_dir_viz = results['user_sw_raw_dir_path']
        user_sw_mask_dir_viz = results['user_sw_mask_dir_path']
        ctc_tra_dir_viz = results['ctc_tra_dir_path']
        ctc_seg_dir_viz = results['ctc_seg_dir_path']

        num_plots = 0
        plot_paths_titles = []

        user_sw_raw_path_viz = os.path.join(user_sw_raw_dir_viz,
                                            f"{params['user_sw_raw_filename_prefix']}{frame_to_viz:03d}.{params['user_sw_output_file_extension']}")
        if os.path.exists(user_sw_raw_path_viz):
            plot_paths_titles.append(
                {'path': user_sw_raw_path_viz, 'title': f"UserSW Raw Fr {frame_to_viz}", 'cmap': 'gray'})
            num_plots += 1
        else:
            print(f"Viz: UserSW Raw not found at {user_sw_raw_path_viz}")

        user_sw_mask_path_viz = os.path.join(user_sw_mask_dir_viz,
                                             f"{params['user_sw_mask_filename_prefix']}{frame_to_viz:03d}.{params['user_sw_output_file_extension']}")
        if os.path.exists(user_sw_mask_path_viz):
            plot_paths_titles.append(
                {'path': user_sw_mask_path_viz, 'title': f"UserSW Mask Fr {frame_to_viz}", 'cmap': 'nipy_spectral'})
            num_plots += 1
        else:
            print(f"Viz: UserSW Mask not found at {user_sw_mask_path_viz}")

        if os.path.exists(ctc_tra_dir_viz):
            ctc_tra_mask_path_viz = os.path.join(ctc_tra_dir_viz, f"mask{frame_to_viz:03d}.tif")
            if os.path.exists(ctc_tra_mask_path_viz):
                plot_paths_titles.append(
                    {'path': ctc_tra_mask_path_viz, 'title': f"CTC TRA Mask Fr {frame_to_viz}", 'cmap': 'nipy_spectral',
                     'is_tif': True})
                num_plots += 1
            else:
                print(f"Viz: CTC TRA Mask not found at {ctc_tra_mask_path_viz}")

        if ctc_seg_dir_viz and os.path.exists(ctc_seg_dir_viz) and params['create_gt_seg_folder']:
            ctc_seg_mask_path_viz = os.path.join(ctc_seg_dir_viz, f"man_seg{frame_to_viz:03d}.tif")
            if os.path.exists(ctc_seg_mask_path_viz):
                plot_paths_titles.append(
                    {'path': ctc_seg_mask_path_viz, 'title': f"CTC SEG Mask Fr {frame_to_viz}", 'cmap': 'nipy_spectral',
                     'is_tif': True})
                num_plots += 1
            else:
                print(f"Viz: CTC SEG Mask not found at {ctc_seg_mask_path_viz}")

        if num_plots == 0:
            print("No images found for visualization summary.")
            return

        fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5.5))
        if num_plots == 1 and not isinstance(axs, np.ndarray): axs = [axs]  # Ensure axs is iterable

        for ax_idx, plot_info in enumerate(plot_paths_titles):
            if ax_idx < len(axs):
                img_data = tifffile.imread(plot_info['path']) if plot_info.get('is_tif') else plt.imread(
                    plot_info['path'])
                axs[ax_idx].imshow(img_data, cmap=plot_info['cmap'])
                axs[ax_idx].set_title(plot_info['title'])
                axs[ax_idx].axis('off')
            else:
                print(
                    f"Warning: Not enough subplots for all images. Expected {num_plots}, got {len(axs)} for ax_idx {ax_idx}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f"Visualization Summary - Frame {frame_to_viz} - Seed {params['random_seed']}", fontsize=14)
        plt.savefig(os.path.join(output_base_dir,
                                 f"visualization_summary_frame_{frame_to_viz}_seed{params['random_seed']}.png"))
        plt.close(fig)
        print(f"\nSaved a sample visualization summary for frame {frame_to_viz} to {output_base_dir}")

    except Exception as e:
        print(f"\nError during sample visualization for frame {frame_to_viz}: {e}")
        traceback.print_exc()

    print(f"\nAll processing complete! Dataset saved to {output_base_dir}")


if __name__ == "__main__":
    main()
