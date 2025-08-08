# Author: ImagineA / Andrei Rares
# Date: 2018-08-18
# Original forward tracking algorithm.
# Reworked by Gemini for backward tracking to handle multi-generational mitosis (May 2024).
# Refined by Gemini for simpler lineage construction and ID handling (May 2024).
# Version 2: Added defensive checks and more detailed logging for ID assignment.
# Version 3 (this update): Refined fusion detection for parent to select the "best" (e.g., nearest) valid daughter pair.
# Version 4 (this update): Corrected parent_particle update logic for multi-generational linking.
# Version 5 (User Request): Added Forward tracking mode and refactored into forward/backward specific functions.
# Version 6 (User Request - this update): Added detailed logging to detect_and_link_mitosis_forward for debugging multi-generational parent assignment.
# Version 7 (User Request - this update): Added "Forward (No Mitosis)" tracking mode.
# Version 8 (Gemini): Added tp.quiet() to suppress trackpy's default print statements.
# Version 9 (Gemini - This Update): Added 'original_mask_label' to get_region_info.

import numpy as np
import pandas as pd
import trackpy as tp
from collections import defaultdict
from datetime import datetime
from os.path import join, split, normpath
from skimage.measure import regionprops
import logging
from itertools import combinations

# Global variable for assigning new daughter IDs
next_available_daughter_id = -1


def initialize_next_daughter_id(trj):
    """Initializes the global next_available_daughter_id based on the maximum particle ID in the trajectory."""
    global next_available_daughter_id
    if not trj.empty and 'particle' in trj.columns and trj['particle'].notna().any():
        valid_ids = pd.to_numeric(trj['particle'], errors='coerce').dropna()
        if not valid_ids.empty:
            next_available_daughter_id = int(valid_ids.max()) + 1
        else:
            next_available_daughter_id = 1
    else:
        next_available_daughter_id = 1
    logging.info(f"TRACKING_MODE_COMMON: next_available_daughter_id initialized to: {next_available_daughter_id}")


def get_new_daughter_id():
    """Returns a new unique ID for a daughter cell track."""
    global next_available_daughter_id
    if next_available_daughter_id == -1:
        logging.warning(
            "TRACKING_MODE_COMMON: next_available_daughter_id not initialized! Defaulting to 10000.")
        next_available_daughter_id = 10000  # Fallback, should be initialized by initialize_next_daughter_id

    new_id = next_available_daughter_id
    next_available_daughter_id += 1
    logging.debug(
        f"TRACKING_MODE_COMMON: get_new_daughter_id() returning: {new_id}. Next will be: {next_available_daughter_id}")
    return new_id


def initialize_experiment_parameters(path_in, current_max_displacement, current_max_absence_interval):
    """Initializes parameters for the tracking experiment."""
    n_active_features = 8  # Number of features used for linking (excluding x, y)
    idx_start_active_features = 1  # Index of the first active feature in col_tuple['original']
    col_tuple = {
        'original': ['frame', 'y', 'x', 'equivalent_diameter', 'perimeter', 'eccentricity',
                     'orientation_x_2_sin', 'orientation_x_2_cos', 'true_solidity',
                     'solidity', 'area', 'mean_intensity', 'angle', 'original_mask_label'], # Added original_mask_label
        'extra': ['bbox_top', 'bbox_left', 'bbox_bottom', 'bbox_right']
    }
    col_tuple['weighted'] = ['wtd_{}'.format(x) for x in col_tuple['original']]

    # Define weights for features used in linking
    col_weights = defaultdict(
        lambda: 1.0,  # Default weight for unlisted features
        {
            'frame': (float(current_max_displacement) / float(current_max_absence_interval))
            if current_max_absence_interval > 0 else (float(current_max_displacement) or 10.0),
            'equivalent_diameter': 0.75,
            'perimeter': 0.25,
            'eccentricity': 5.0,
            'orientation_x_2_sin': 5.0,
            'orientation_x_2_cos': 5.0,
            'true_solidity': 5.0 * np.pi  # A custom feature combining diameter and perimeter
        }
    )
    # Prepare output path (though not directly used in this function, it's part of original init)
    in_path_part_1, in_path_part_2 = split(normpath(path_in))
    exp_time = datetime.now()
    out_path = join(in_path_part_1,
                    '{}_Exp_{}-{:02d}-{:02d}T{:02d}{:02d}{:02d}'.format(in_path_part_2,
                                                                        exp_time.year, exp_time.month, exp_time.day,
                                                                        exp_time.hour, exp_time.minute,
                                                                        exp_time.second))
    return n_active_features, idx_start_active_features, col_tuple, col_weights, \
        current_max_displacement, current_max_absence_interval, out_path


def get_region_info(region, frame_index, col_weights, col_tuple):
    """Extracts feature information from a single skimage.measure.regionprops region."""
    # Basic geometric and intensity properties
    area = region.area if region.area > 0 else 1e-6  # Avoid division by zero
    eq_diameter = region.equivalent_diameter if region.equivalent_diameter > 0 else 1e-6
    perimeter = max(1, region.perimeter)  # Avoid division by zero for true_solidity

    feat_dict = {
        'y': region.centroid[0], 'x': region.centroid[1],
        'equivalent_diameter': eq_diameter, 'perimeter': perimeter,
        'eccentricity': region.eccentricity,
        'orientation_x_2_sin': np.sin(2 * region.orientation),  # For rotational invariance
        'orientation_x_2_cos': np.cos(2 * region.orientation),  # For rotational invariance
        'true_solidity': eq_diameter / perimeter if perimeter > 0 else 0,  # Custom solidity measure
        'solidity': region.solidity,  # Standard solidity
        'area': area,
        'mean_intensity': region.mean_intensity,  # Used for matching mask label to region
        'angle': region.orientation,  # Original orientation angle
        'frame': frame_index,
        'original_mask_label': region.label # <<<--- THIS IS THE ADDED LINE
    }
    # Calculate weighted features for linking
    weighted_features_list = [('wtd_{}'.format(name), col_weights[name] * feat_dict[name])
                              for name in col_tuple['original'] if name in feat_dict]
    feat_dict.update(dict(weighted_features_list))
    # Add bounding box info
    feat_dict.update({
        'bbox_top': region.bbox[0], 'bbox_left': region.bbox[1],
        'bbox_bottom': region.bbox[2], 'bbox_right': region.bbox[3]
    })
    return feat_dict


def calculate_initial_cell_info(raw_masks, n_active_features, idx_start_active_features,
                                col_tuple, col_weights, mode="backward"):
    """
    Extracts cell features from masks.
    Iterates frames backward for 'backward' mode, forward for 'forward' mode.
    """
    logging.info(
        f'TRACKING_MODE_COMMON: ================= Extracting cell features ({mode.capitalize()} Pass) =================')
    features_list = []
    regions_found = False

    frame_iterator = range(raw_masks.shape[2] - 1, -1, -1) if mode == "backward" else range(raw_masks.shape[2])

    for i_frame_chrono in frame_iterator:
        current_mask_frame = raw_masks[:, :, i_frame_chrono].astype(np.int32)
        # Pass the raw mask frame as intensity_image to ensure region.label is from the raw mask
        frame_regions = regionprops(current_mask_frame, intensity_image=current_mask_frame)


        current_frame_features = [get_region_info(region, i_frame_chrono, col_weights, col_tuple)
                                  for region in frame_regions if
                                  region.label != 0] # region.label is the original mask label
        if current_frame_features:
            features_list.extend(current_frame_features)
            regions_found = True

    if not regions_found:
        logging.error(f"TRACKING_MODE_COMMON ({mode}): No valid regions found in any masks! Cannot proceed.")
        return pd.DataFrame()

    features_df = pd.DataFrame(features_list)
    # Sort for trackpy:
    # Backward mode: sort descending by frame (trackpy sees it as ascending time)
    # Forward mode: sort ascending by frame (trackpy sees it as ascending time)
    sort_ascending = True if mode == "forward" else False
    features_df.sort_values(by='frame', ascending=sort_ascending, inplace=True)
    logging.info(f"TRACKING_MODE_COMMON ({mode}): Extracted {len(features_df)} features, sorted for linking.")
    return features_df


def detect_and_link_mitosis_forward(trj, image_shape,
                                    mitosis_max_dist_factor,
                                    mitosis_area_sum_min_factor,
                                    mitosis_area_sum_max_factor,
                                    mitosis_daughter_area_similarity):
    """
    CORRECTED version of forward mitosis detection that properly handles multi-generational mitosis.

    Key fixes:
    1. Only rename tracks from the mitosis frame forward (not the entire track history)
    2. Preserve parent-child relationships for previous generations
    3. Properly handle track segment splitting at mitosis points
    4. End parent tracks at mitosis points
    """
    logging.info(
        'FORWARD_TRACKING_CORRECTED: ================= Detecting Mitosis Events (Multi-generational) =================')

    # Ensure 'particle' is int and 'parent_particle' exists and is nullable int
    trj['particle'] = trj['particle'].astype(int)
    if 'parent_particle' not in trj.columns:
        trj['parent_particle'] = pd.NA
    trj['parent_particle'] = trj['parent_particle'].astype(pd.Int64Dtype())

    # Get sorted unique frame numbers
    chronological_frames = sorted(trj['frame'].unique())
    if len(chronological_frames) < 2:
        logging.info("FORWARD_TRACKING_CORRECTED: Not enough frames for mitosis detection.")
        return

    # Iterate through frame transitions chronologically
    for frame_idx_chrono in range(len(chronological_frames) - 1):
        parent_frame_chrono = chronological_frames[frame_idx_chrono]
        daughter_frame_chrono = chronological_frames[frame_idx_chrono + 1]

        logging.debug(
            f"FORWARD_TRACKING_CORRECTED: Mitosis check: PFC {parent_frame_chrono} -> DFC {daughter_frame_chrono}")

        # Get cells in the current parent frame
        cells_in_pfc = trj[trj['frame'] == parent_frame_chrono]
        cells_in_dfc_for_transition = trj[trj['frame'] == daughter_frame_chrono]

        if cells_in_pfc.empty or len(cells_in_dfc_for_transition) < 2:
            continue

        processed_parents_in_pfc_this_transition = set()

        # Iterate through each potential parent cell in the parent frame
        for _, parent_candidate_series in cells_in_pfc.iterrows():
            parent_p_id = int(parent_candidate_series['particle'])
            if parent_p_id in processed_parents_in_pfc_this_transition:
                continue

            logging.debug(
                f"  FORWARD_TRACKING_CORRECTED: Potential Parent P (ID {parent_p_id}) at PFC {parent_frame_chrono}.")

            best_mitosis_for_this_P = {
                'd1_idx': None, 'd2_idx': None,
                'd1_current_particle_id': None, 'd2_current_particle_id': None,
                'score': float('inf'),
                'dist_d1_to_p': 0, 'dist_d2_to_p': 0
            }

            # Get indices of DFC cells that are currently unassigned
            unassigned_dfc_indices_now = cells_in_dfc_for_transition.index[
                trj.loc[cells_in_dfc_for_transition.index, 'parent_particle'].isna()
            ]

            if len(unassigned_dfc_indices_now) < 2:
                logging.debug(
                    f"    Parent {parent_p_id}: Not enough unassigned daughters in DFC {daughter_frame_chrono}.")
                continue

            # Consider all combinations of two unassigned daughters
            for d1_main_trj_idx, d2_main_trj_idx in combinations(unassigned_dfc_indices_now, 2):
                d1_series = trj.loc[d1_main_trj_idx]
                d2_series = trj.loc[d2_main_trj_idx]
                current_particle_id_d1 = int(d1_series['particle'])
                current_particle_id_d2 = int(d2_series['particle'])

                # Calculate distances and check criteria
                dist_d1_to_p = np.sqrt((d1_series['x'] - parent_candidate_series['x']) ** 2 +
                                       (d1_series['y'] - parent_candidate_series['y']) ** 2)
                dist_d2_to_p = np.sqrt((d2_series['x'] - parent_candidate_series['x']) ** 2 +
                                       (d2_series['y'] - parent_candidate_series['y']) ** 2)
                parent_diameter = parent_candidate_series['equivalent_diameter']

                if parent_diameter <= 1e-6:
                    continue

                # Distance criterion
                if not (dist_d1_to_p < parent_diameter * mitosis_max_dist_factor and
                        dist_d2_to_p < parent_diameter * mitosis_max_dist_factor):
                    continue

                # Area criteria
                area_sum_d1d2 = d1_series['area'] + d2_series['area']
                parent_p_area = parent_candidate_series['area']
                if parent_p_area <= 1e-6:
                    continue
                area_ratio_sum_vs_parent = area_sum_d1d2 / parent_p_area

                if d1_series['area'] <= 1e-6 or d2_series['area'] <= 1e-6:
                    continue
                d_area_similarity_ratio = min(d1_series['area'], d2_series['area']) / max(
                    1e-6, max(d1_series['area'], d2_series['area']))

                logging.debug(
                    f"      Candidate DPair for P_ID {parent_p_id}: D1_curr_ID {current_particle_id_d1}, D2_curr_ID {current_particle_id_d2}:")
                logging.debug(
                    f"        Dist D1-P: {dist_d1_to_p:.1f}, D2-P: {dist_d2_to_p:.1f} (Crit: < {parent_diameter * mitosis_max_dist_factor:.1f})")
                logging.debug(
                    f"        AreaSumFactor: {area_ratio_sum_vs_parent:.2f} (Crit: {mitosis_area_sum_min_factor}-{mitosis_area_sum_max_factor})")
                logging.debug(
                    f"        DaughterSimFactor: {d_area_similarity_ratio:.2f} (Crit: >{mitosis_daughter_area_similarity})")

                # Check if all criteria are met
                if (mitosis_area_sum_min_factor < area_ratio_sum_vs_parent < mitosis_area_sum_max_factor and
                        d_area_similarity_ratio > mitosis_daughter_area_similarity):
                    current_score = (dist_d1_to_p + dist_d2_to_p) / 2
                    if current_score < best_mitosis_for_this_P['score']:
                        best_mitosis_for_this_P.update({
                            'score': current_score, 'd1_idx': d1_main_trj_idx, 'd2_idx': d2_main_trj_idx,
                            'd1_current_particle_id': current_particle_id_d1,
                            'd2_current_particle_id': current_particle_id_d2,
                            'dist_d1_to_p': dist_d1_to_p, 'dist_d2_to_p': dist_d2_to_p
                        })
                        logging.debug(f"        New best pair for P_ID {parent_p_id} found. Score: {current_score:.2f}")

            # Process the best mitosis event found for this parent
            if best_mitosis_for_this_P['d1_idx'] is not None:
                chosen_d1_id = best_mitosis_for_this_P['d1_current_particle_id']
                chosen_d2_id = best_mitosis_for_this_P['d2_current_particle_id']

                # Get new unique IDs for the daughter tracks
                new_daughter_id1 = get_new_daughter_id()
                new_daughter_id2 = get_new_daughter_id()

                # Critical check for ID clashes
                if (new_daughter_id1 == parent_p_id or new_daughter_id2 == parent_p_id or
                        new_daughter_id1 == new_daughter_id2):
                    logging.error(f"CRITICAL ID CLASH PREVENTED: Retrying ID generation.")
                    new_daughter_id1 = get_new_daughter_id()
                    new_daughter_id2 = get_new_daughter_id()
                    if (new_daughter_id1 == parent_p_id or new_daughter_id2 == parent_p_id or
                            new_daughter_id1 == new_daughter_id2):
                        logging.error(f"ID CLASH PERSISTS: SKIPPING MITOSIS for parent {parent_p_id}.")
                        continue

                logging.info(
                    f"  FORWARD_TRACKING_CORRECTED: MITOSIS CONFIRMED for Parent {parent_p_id} at frame {parent_frame_chrono}")
                logging.info(f"    Original daughter IDs: {chosen_d1_id}, {chosen_d2_id}")
                logging.info(f"    New daughter IDs: {new_daughter_id1}, {new_daughter_id2}")

                # === CORRECTED SECTION: Proper multi-generational handling ===

                # 1. FIRST: Handle existing grandchildren BEFORE any renaming
                #    Find all tracks that have chosen daughters as parents
                grandchildren_d1_mask = trj['parent_particle'] == chosen_d1_id
                grandchildren_d2_mask = trj['parent_particle'] == chosen_d2_id

                logging.debug(f"    Found {grandchildren_d1_mask.sum()} grandchildren of D1 ({chosen_d1_id})")
                logging.debug(f"    Found {grandchildren_d2_mask.sum()} grandchildren of D2 ({chosen_d2_id})")

                # Update grandchildren to point to new daughter IDs
                trj.loc[grandchildren_d1_mask, 'parent_particle'] = new_daughter_id1
                trj.loc[grandchildren_d2_mask, 'parent_particle'] = new_daughter_id2

                # 2. CRITICAL FIX: Split tracks at mitosis point
                #    Rename ONLY segments from daughter_frame_chrono onwards

                # For daughter 1: rename only from daughter_frame_chrono onwards
                d1_future_mask = (trj['particle'] == chosen_d1_id) & (trj['frame'] >= daughter_frame_chrono)
                logging.debug(
                    f"    Renaming D1 future segments: {d1_future_mask.sum()} rows from frame {daughter_frame_chrono}")
                trj.loc[d1_future_mask, 'particle'] = new_daughter_id1

                # Set parent for the new daughter track
                d1_new_mask = trj['particle'] == new_daughter_id1
                trj.loc[d1_new_mask, 'parent_particle'] = parent_p_id

                # For daughter 2: rename only from daughter_frame_chrono onwards
                d2_future_mask = (trj['particle'] == chosen_d2_id) & (trj['frame'] >= daughter_frame_chrono)
                logging.debug(
                    f"    Renaming D2 future segments: {d2_future_mask.sum()} rows from frame {daughter_frame_chrono}")
                trj.loc[d2_future_mask, 'particle'] = new_daughter_id2

                # Set parent for the new daughter track
                d2_new_mask = trj['particle'] == new_daughter_id2
                trj.loc[d2_new_mask, 'parent_particle'] = parent_p_id

                # 3. IMPORTANT: End the parent track at the mitosis frame
                #    Remove parent track entries beyond parent_frame_chrono
                parent_future_mask = (trj['particle'] == parent_p_id) & (trj['frame'] > parent_frame_chrono)
                if parent_future_mask.any():
                    logging.debug(f"    Removing {parent_future_mask.sum()} parent track entries beyond mitosis frame")
                    trj.drop(trj[parent_future_mask].index, inplace=True)

                # 4. Verify that original daughter IDs still exist for pre-mitosis history
                d1_pre_mitosis = (trj['particle'] == chosen_d1_id) & (trj['frame'] < daughter_frame_chrono)
                d2_pre_mitosis = (trj['particle'] == chosen_d2_id) & (trj['frame'] < daughter_frame_chrono)

                logging.debug(f"    Pre-mitosis history preserved:")
                logging.debug(
                    f"      D1 ({chosen_d1_id}): {d1_pre_mitosis.sum()} rows before frame {daughter_frame_chrono}")
                logging.debug(
                    f"      D2 ({chosen_d2_id}): {d2_pre_mitosis.sum()} rows before frame {daughter_frame_chrono}")
                logging.debug(f"    Post-mitosis tracks created:")
                logging.debug(f"      New D1 ({new_daughter_id1}): {(trj['particle'] == new_daughter_id1).sum()} rows")
                logging.debug(f"      New D2 ({new_daughter_id2}): {(trj['particle'] == new_daughter_id2).sum()} rows")

                processed_parents_in_pfc_this_transition.add(parent_p_id)
            else:
                logging.debug(
                    f"    Parent {parent_p_id}: No suitable daughter pair found after checking all combinations.")

    # Post-processing: Reset index after any row deletions
    trj.reset_index(drop=True, inplace=True)

    # Validation
    validation_issues = validate_forward_tracking_consistency(trj)
    if validation_issues:
        logging.warning("Validation found issues after mitosis detection - this may indicate problems")
    else:
        logging.info("Forward mitosis detection completed successfully - validation passed")


def validate_forward_tracking_consistency(trj):
    """
    Enhanced validation that checks for proper parent-child relationships
    across multiple generations.
    """
    validation_issues = []

    # Check for self-parenting
    if 'parent_particle' in trj.columns:
        self_parents = trj[trj['particle'] == trj['parent_particle']]
        if not self_parents.empty:
            validation_issues.append(f"Self-parenting detected: {self_parents['particle'].unique()}")

    # Check for orphaned children (children whose parents don't exist in the trajectory)
    children_with_parents = trj.dropna(
        subset=['parent_particle']) if 'parent_particle' in trj.columns else pd.DataFrame()
    all_particle_ids = set(trj['particle'].unique())

    for _, row in children_with_parents.iterrows():
        parent_id = int(row['parent_particle'])
        if parent_id not in all_particle_ids:
            validation_issues.append(f"Orphaned child {row['particle']} has non-existent parent {parent_id}")

    # Check for duplicate particle IDs in the same frame
    frame_particle_counts = trj.groupby(['frame', 'particle']).size()
    duplicates = frame_particle_counts[frame_particle_counts > 1]
    if not duplicates.empty:
        validation_issues.append(f"Duplicate particle IDs in same frame: {duplicates.index.tolist()}")

    # Check temporal consistency (children should not appear before parents)
    if 'parent_particle' in trj.columns:
        for parent_id in trj['parent_particle'].dropna().unique():
            parent_id = int(parent_id)
            parent_frames = trj[trj['particle'] == parent_id]['frame']
            children_frames = trj[trj['parent_particle'] == parent_id]['frame']

            if not parent_frames.empty and not children_frames.empty:
                parent_first_frame = parent_frames.min()
                parent_last_frame = parent_frames.max()
                children_first_frame = children_frames.min()
                children_last_frame = children_frames.max()

                # Children should start at or after parent's last frame (mitosis point)
                if children_first_frame < parent_last_frame:
                    logging.debug(
                        f"Child appears during parent lifetime: Parent {parent_id} active {parent_first_frame}-{parent_last_frame}, children start at {children_first_frame}")

                # Check for gaps in parent-child timeline
                if children_first_frame > parent_last_frame + 1:
                    validation_issues.append(
                        f"Gap between parent end and children start: Parent {parent_id} ends at {parent_last_frame}, children start at {children_first_frame}")

    if validation_issues:
        logging.warning("Forward tracking validation issues:")
        for issue in validation_issues:
            logging.warning(f"  - {issue}")
    else:
        logging.info("Forward tracking validation passed - no issues detected")

    return validation_issues

def detect_and_link_fusions_backward(trj, image_shape,
                                     fusion_max_dist_factor,
                                     fusion_area_sum_min_factor,
                                     fusion_area_sum_max_factor,
                                     fusion_daughter_area_similarity):
    """
    Detects fusion events (chronological splits when viewed backward) in backward-linked trajectories.
    """
    logging.info(
        'BACKWARD_TRACKING: ================= Detecting Fusion Events (Parent picks best daughter pair) =================')
    trj['particle'] = trj['particle'].astype(int)
    if 'parent_particle' not in trj.columns:
        trj['parent_particle'] = pd.NA
    trj['parent_particle'] = trj['parent_particle'].astype(pd.Int64Dtype())

    chronological_frames = sorted(trj['frame'].unique())
    if len(chronological_frames) < 2:
        logging.info("BACKWARD_TRACKING: Not enough frames for fusion detection (need at least 2).")
        return

    for frame_idx_chrono in range(len(chronological_frames) - 1, 0, -1):
        daughter_frame_chrono = chronological_frames[frame_idx_chrono]
        parent_frame_chrono = chronological_frames[frame_idx_chrono - 1]

        logging.debug(
            f"BACKWARD_TRACKING: Fusion check: PFC {parent_frame_chrono} (Potential Parents) <- DFC {daughter_frame_chrono} (Potential Daughters)")

        cells_in_pfc = trj[trj['frame'] == parent_frame_chrono]
        cells_in_dfc_for_transition = trj[trj['frame'] == daughter_frame_chrono]

        if cells_in_pfc.empty or len(cells_in_dfc_for_transition) < 2:
            continue

        processed_parents_in_pfc_this_transition = set()

        for _, parent_candidate_series in cells_in_pfc.iterrows():
            parent_p_id = int(parent_candidate_series['particle'])
            if parent_p_id in processed_parents_in_pfc_this_transition:
                continue

            logging.debug(f"  BACKWARD_TRACKING: Potential Parent P (ID {parent_p_id}) at PFC {parent_frame_chrono}.")

            best_fusion_for_this_P = {
                'd1_idx': None, 'd2_idx': None,
                'd1_current_particle_id': None, 'd2_current_particle_id': None,
                'score': float('inf'),
                'dist_d1_to_p': 0, 'dist_d2_to_p': 0
            }

            unassigned_dfc_indices_now = cells_in_dfc_for_transition.index[
                trj.loc[cells_in_dfc_for_transition.index, 'parent_particle'].isna()
            ]

            if len(unassigned_dfc_indices_now) < 2:
                logging.debug(
                    f"    Parent {parent_p_id}: Not enough unassigned daughters in DFC {daughter_frame_chrono} ({len(unassigned_dfc_indices_now)} found).")
                continue

            logging.debug(
                f"    Parent {parent_p_id}: Found {len(unassigned_dfc_indices_now)} unassigned daughters to consider.")

            for d1_main_trj_idx, d2_main_trj_idx in combinations(unassigned_dfc_indices_now, 2):
                d1_series = trj.loc[d1_main_trj_idx]
                d2_series = trj.loc[d2_main_trj_idx]

                current_particle_id_d1 = int(d1_series['particle'])
                current_particle_id_d2 = int(d2_series['particle'])

                dist_d1_to_p = np.sqrt((d1_series['x'] - parent_candidate_series['x']) ** 2 + (
                        d1_series['y'] - parent_candidate_series['y']) ** 2)
                dist_d2_to_p = np.sqrt((d2_series['x'] - parent_candidate_series['x']) ** 2 + (
                        d2_series['y'] - parent_candidate_series['y']) ** 2)
                parent_diameter = parent_candidate_series['equivalent_diameter']

                if parent_diameter <= 1e-6: continue

                if not (dist_d1_to_p < parent_diameter * fusion_max_dist_factor and \
                        dist_d2_to_p < parent_diameter * fusion_max_dist_factor):
                    continue

                area_sum_d1d2 = d1_series['area'] + d2_series['area']
                parent_p_area = parent_candidate_series['area']
                if parent_p_area <= 1e-6: continue
                area_ratio_sum_vs_parent = area_sum_d1d2 / parent_p_area

                if d1_series['area'] <= 1e-6 or d2_series['area'] <= 1e-6: continue
                d_area_similarity_ratio = min(d1_series['area'], d2_series['area']) / max(1e-6, max(d1_series['area'],
                                                                                                    d2_series['area']))
                logging.debug(
                    f"      Candidate DPair for P_ID {parent_p_id}: D1_curr_ID {current_particle_id_d1}, D2_curr_ID {current_particle_id_d2}:")
                logging.debug(
                    f"        Dist D1-P: {dist_d1_to_p:.1f}, D2-P: {dist_d2_to_p:.1f} (Crit: < {parent_diameter * fusion_max_dist_factor:.1f})")
                logging.debug(
                    f"        AreaSumFactor: {area_ratio_sum_vs_parent:.2f} (Crit: {fusion_area_sum_min_factor}-{fusion_area_sum_max_factor})")
                logging.debug(
                    f"        DaughterSimFactor: {d_area_similarity_ratio:.2f} (Crit: >{fusion_daughter_area_similarity})")

                if (fusion_area_sum_min_factor < area_ratio_sum_vs_parent < fusion_area_sum_max_factor and
                        d_area_similarity_ratio > fusion_daughter_area_similarity):

                    current_score = (dist_d1_to_p + dist_d2_to_p) / 2
                    if current_score < best_fusion_for_this_P['score']:
                        best_fusion_for_this_P['score'] = current_score
                        best_fusion_for_this_P['d1_idx'] = d1_main_trj_idx
                        best_fusion_for_this_P['d2_idx'] = d2_main_trj_idx
                        best_fusion_for_this_P['d1_current_particle_id'] = current_particle_id_d1
                        best_fusion_for_this_P['d2_current_particle_id'] = current_particle_id_d2
                        best_fusion_for_this_P['dist_d1_to_p'] = dist_d1_to_p
                        best_fusion_for_this_P['dist_d2_to_p'] = dist_d2_to_p
                        logging.debug(f"        New best pair for P_ID {parent_p_id} found. Score: {current_score:.2f}")

            if best_fusion_for_this_P['d1_idx'] is not None:
                chosen_d1_id = best_fusion_for_this_P['d1_current_particle_id']
                chosen_d2_id = best_fusion_for_this_P['d2_current_particle_id']
                new_daughter_id1 = get_new_daughter_id()
                new_daughter_id2 = get_new_daughter_id()

                if new_daughter_id1 == parent_p_id or new_daughter_id2 == parent_p_id or new_daughter_id1 == new_daughter_id2:
                    logging.error(
                        f"CRITICAL ID CLASH PREVENTED (Parent {parent_p_id} choosing best pair): P={parent_p_id}, NewD1={new_daughter_id1}, NewD2={new_daughter_id2}. Retrying.")
                    new_daughter_id1 = get_new_daughter_id()
                    new_daughter_id2 = get_new_daughter_id()
                    if new_daughter_id1 == parent_p_id or new_daughter_id2 == parent_p_id or new_daughter_id1 == new_daughter_id2:
                        logging.error(
                            f"CRITICAL ID CLASH PERSISTS (Parent {parent_p_id} choosing best pair) after retry: P={parent_p_id}, NewD1={new_daughter_id1}, NewD2={new_daughter_id2}. SKIPPING FUSION for this parent.")
                        continue

                logging.info(
                    f"  BACKWARD_TRACKING: FUSION (Parent {parent_p_id} chose best pair)! P_ID {parent_p_id} (PFC {parent_frame_chrono}) <- "
                    f"D1 (orig_curr_ID {chosen_d1_id} -> new {new_daughter_id1}), "
                    f"D2 (orig_curr_ID {chosen_d2_id} -> new {new_daughter_id2}) from DFC {daughter_frame_chrono}. "
                    f"Score: {best_fusion_for_this_P['score']:.2f}, Dists: {best_fusion_for_this_P['dist_d1_to_p']:.1f}, {best_fusion_for_this_P['dist_d2_to_p']:.1f}")

                trj.loc[trj['parent_particle'] == chosen_d1_id, 'parent_particle'] = new_daughter_id1
                trj.loc[trj['parent_particle'] == chosen_d2_id, 'parent_particle'] = new_daughter_id2

                d1_segment_mask = (trj['particle'] == chosen_d1_id) & (trj['frame'] >= daughter_frame_chrono)
                trj.loc[d1_segment_mask, 'particle'] = new_daughter_id1
                trj.loc[(trj['particle'] == new_daughter_id1) & (trj['frame'] >= daughter_frame_chrono) & trj[
                    'parent_particle'].isna(), 'parent_particle'] = parent_p_id

                d2_segment_mask = (trj['particle'] == chosen_d2_id) & (trj['frame'] >= daughter_frame_chrono)
                trj.loc[d2_segment_mask, 'particle'] = new_daughter_id2
                trj.loc[(trj['particle'] == new_daughter_id2) & (trj['frame'] >= daughter_frame_chrono) & trj[
                    'parent_particle'].isna(), 'parent_particle'] = parent_p_id

                processed_parents_in_pfc_this_transition.add(parent_p_id)
            else:
                logging.debug(
                    f"    Parent {parent_p_id}: No suitable daughter pair found after checking all combinations.")


def reindex_particles_and_get_map(trj_df, min_id_val, mode="backward"):
    """
    Re-indexes 'particle' IDs sequentially starting from min_id_val.
    """
    logging.info(
        f'TRACKING_MODE_COMMON ({mode}): ================= Re-indexing final particle IDs from {min_id_val} =================')
    if 'particle' not in trj_df.columns or trj_df.empty:
        logging.warning(
            f"TRACKING_MODE_COMMON ({mode}): Trajectory empty or no 'particle' column. Skipping re-indexing.")
        return {}
    trj_df['particle'] = trj_df['particle'].astype(int)

    temp_trj_for_sort = trj_df.copy()
    if 'y' not in temp_trj_for_sort.columns:
        temp_trj_for_sort['y_for_sort'] = 0
    else:
        temp_trj_for_sort['y_for_sort'] = temp_trj_for_sort['y']
    if 'x' not in temp_trj_for_sort.columns:
        temp_trj_for_sort['x_for_sort'] = 0
    else:
        temp_trj_for_sort['x_for_sort'] = temp_trj_for_sort['x']

    temp_trj_for_sort.sort_values(by=['frame', 'y_for_sort', 'x_for_sort', 'particle'], inplace=True)
    sorted_old_ids_to_remap = temp_trj_for_sort['particle'].unique()

    old_to_new_id_map = {old_id: min_id_val + i for i, old_id in enumerate(sorted_old_ids_to_remap)}
    logging.debug(f"TRACKING_MODE_COMMON ({mode}): Re-indexing map (old_id -> new_id): Count {len(old_to_new_id_map)}")

    trj_df['particle'] = trj_df['particle'].map(old_to_new_id_map)
    if trj_df['particle'].isna().any():
        num_na = trj_df['particle'].isna().sum()
        logging.error(
            f"TRACKING_MODE_COMMON ({mode}): {num_na} NaN values in 'particle' after re-indexing! Check map. Example NaN rows:")
        logging.error(trj_df[trj_df['particle'].isna()].head())
        unmapped_ids = set(temp_trj_for_sort['particle'].unique()) - set(old_to_new_id_map.keys())
        if unmapped_ids:
            logging.error(f"    Original IDs that were in trj but not in map keys (should be empty): {unmapped_ids}")
        trj_df['particle'] = trj_df['particle'].fillna(-999).astype(int)
    else:
        trj_df['particle'] = trj_df['particle'].astype(int)

    if not trj_df.empty and 'particle' in trj_df.columns and trj_df['particle'].notna().all():
        min_p_id_final = trj_df["particle"].min()
        max_p_id_final = trj_df["particle"].max()
        if min_p_id_final == -999:
            logging.warning(
                f'TRACKING_MODE_COMMON ({mode}): Re-indexing done, but NaNs were filled with -999. New ID range might be affected.')
        else:
            logging.info(
                f'TRACKING_MODE_COMMON ({mode}): Re-indexing done. New ID range: [{min_p_id_final}-{max_p_id_final}]')
    return old_to_new_id_map


def _track_cells_common_logic(path_in, raw_masks, min_cell_id, search_range, memory,
                              neighbor_strategy, mitosis_max_dist_factor,
                              mitosis_area_sum_min_factor, mitosis_area_sum_max_factor,
                              mitosis_daughter_area_similarity, mode="backward",
                              perform_mitosis_detection=True):
    """
    Common logic for both forward and backward tracking.
    """
    global next_available_daughter_id
    next_available_daughter_id = -1

    logging.info(
        f'{mode.upper()}_TRACKING: ================= Starting {mode.capitalize()} Cell Tracking =================')
    if not perform_mitosis_detection:
        logging.info(f'{mode.upper()}_TRACKING: Mitosis/Fusion detection will be SKIPPED for this run.')

    n_active_features, idx_start_active_features, col_tuple, col_weights, _, _, _ = \
        initialize_experiment_parameters(path_in, search_range, memory)
    empty_trj_cols = col_tuple['original'] + col_tuple['weighted'] + col_tuple['extra'] + ['particle',
                                                                                           'parent_particle']

    features_df = calculate_initial_cell_info(raw_masks, n_active_features, idx_start_active_features,
                                              col_tuple, col_weights, mode=mode)
    if features_df.empty:
        return pd.DataFrame(columns=empty_trj_cols), col_tuple, col_weights, {}

    logging.info(
        f"{mode.upper()}_TRACKING: Linking features: sr={search_range}, mem={memory}, strat={neighbor_strategy}")
    pos_cols = col_tuple['weighted'][idx_start_active_features:(idx_start_active_features + n_active_features)]
    missing_cols = [col for col in pos_cols if col not in features_df.columns]
    if missing_cols:
        logging.warning(f"{mode.upper()}_TRACKING: Missing weighted cols: {missing_cols}. Trying 'x', 'y'.")
        pos_cols = ['x', 'y'] if 'x' in features_df.columns and 'y' in features_df.columns else []
    if not pos_cols:
        logging.error(f"{mode.upper()}_TRACKING: No suitable position columns for linking.")
        return pd.DataFrame(columns=empty_trj_cols), col_tuple, col_weights, {}

    # Suppress trackpy's default print statements
    tp.quiet()
    logging.info(f"{mode.upper()}_TRACKING: Trackpy output silenced via tp.quiet().")

    # Add debugging information about the input data
    logging.info(f"{mode.upper()}_TRACKING: Input features_df shape: {features_df.shape}")
    logging.info(f"{mode.upper()}_TRACKING: Features columns: {list(features_df.columns)}")
    logging.info(f"{mode.upper()}_TRACKING: Position columns for linking: {pos_cols}")
    logging.info(f"{mode.upper()}_TRACKING: Frame range: {features_df['frame'].min()} to {features_df['frame'].max()}")
    logging.info(f"{mode.upper()}_TRACKING: Unique frames: {sorted(features_df['frame'].unique())}")
    logging.info(f"{mode.upper()}_TRACKING: Search range: {search_range}, Memory: {memory}, Strategy: {neighbor_strategy}")
    
    # Validate input data
    if features_df.empty:
        logging.error(f"{mode.upper()}_TRACKING: Input features_df is empty!")
        return pd.DataFrame(columns=empty_trj_cols), col_tuple, col_weights, {}
    
    if not all(col in features_df.columns for col in ['x', 'y', 'frame']):
        logging.error(f"{mode.upper()}_TRACKING: Missing required columns (x, y, frame) in features_df")
        return pd.DataFrame(columns=empty_trj_cols), col_tuple, col_weights, {}
    
    # Check for NaN values in critical columns
    nan_check = features_df[['x', 'y', 'frame']].isnull().sum()
    if nan_check.any():
        logging.error(f"{mode.upper()}_TRACKING: Found NaN values in critical columns: {nan_check.to_dict()}")
        return pd.DataFrame(columns=empty_trj_cols), col_tuple, col_weights, {}

    try:
        trj_linked = tp.link(features_df, search_range=search_range, pos_columns=pos_cols,
                             t_column='frame', memory=memory, neighbor_strategy=neighbor_strategy,
                             link_strategy='recursive')
    except Exception as e:
        logging.error(f"{mode.upper()}_TRACKING: Trackpy linking error: {e}")
        logging.error(f"{mode.upper()}_TRACKING: Error traceback: {traceback.format_exc()}")
        return pd.DataFrame(columns=empty_trj_cols), col_tuple, col_weights, {}

    if trj_linked.empty:
        logging.warning(f"{mode.upper()}_TRACKING: Trackpy linking resulted in empty trajectory.")
        logging.warning(f"{mode.upper()}_TRACKING: This could be due to:")
        logging.warning(f"{mode.upper()}_TRACKING: 1. Search range too small for the data")
        logging.warning(f"{mode.upper()}_TRACKING: 2. Memory parameter causing issues")
        logging.warning(f"{mode.upper()}_TRACKING: 3. Insufficient features for linking")
        logging.warning(f"{mode.upper()}_TRACKING: 4. Data format issues")
        logging.warning(f"{mode.upper()}_TRACKING: Input had {len(features_df)} features across {features_df['frame'].nunique()} frames")
        return pd.DataFrame(columns=empty_trj_cols), col_tuple, col_weights, {}
    logging.info(f"{mode.upper()}_TRACKING: Initial {mode} linking: {trj_linked['particle'].nunique()} tracks.")

    initialize_next_daughter_id(trj_linked)

    if perform_mitosis_detection:
        if mode == "backward":
            detect_and_link_fusions_backward(
                trj_linked, raw_masks.shape[:2], mitosis_max_dist_factor,
                mitosis_area_sum_min_factor, mitosis_area_sum_max_factor, mitosis_daughter_area_similarity
            )
        elif mode == "forward":
            detect_and_link_mitosis_forward(
                trj_linked, raw_masks.shape[:2], mitosis_max_dist_factor,
                mitosis_area_sum_min_factor, mitosis_area_sum_max_factor, mitosis_daughter_area_similarity
            )
    else:
        if 'parent_particle' not in trj_linked.columns:
            trj_linked['parent_particle'] = pd.NA
        trj_linked['parent_particle'] = trj_linked['parent_particle'].astype(pd.Int64Dtype())
        logging.info(f"{mode.upper()}_TRACKING: Mitosis/Fusion detection skipped as per configuration.")


    final_id_map = reindex_particles_and_get_map(trj_linked, min_cell_id, mode=mode)

    if 'parent_particle' in trj_linked.columns and not trj_linked['parent_particle'].isna().all():
        trj_linked['parent_particle'] = pd.to_numeric(trj_linked['parent_particle'], errors='coerce') \
            .map(final_id_map).astype(pd.Int64Dtype())

    final_cell_lineage = {}
    if 'parent_particle' in trj_linked.columns and trj_linked['parent_particle'].notna().any():
        trj_for_lineage_build = trj_linked.dropna(subset=['particle', 'parent_particle']).copy()
        if not trj_for_lineage_build.empty:
            trj_for_lineage_build['particle'] = trj_for_lineage_build['particle'].astype(int)
            trj_for_lineage_build['parent_particle'] = trj_for_lineage_build['parent_particle'].astype(int)
            final_cell_lineage = trj_for_lineage_build.groupby('parent_particle')['particle'] \
                .apply(lambda x: sorted(list(x.unique()))).to_dict()

    logging.info(f"{mode.upper()}_TRACKING: Final cell_lineage map has {len(final_cell_lineage)} parent entries.")
    if len(final_cell_lineage) < 20:
        logging.debug(
            f"{mode.upper()}_TRACKING: Final Lineage (first few entries): {dict(list(final_cell_lineage.items())[:5])}")
    else:
        logging.debug(
            f"{mode.upper()}_TRACKING: Sample of Final Lineage: {dict(list(final_cell_lineage.items())[len(final_cell_lineage) // 2: len(final_cell_lineage) // 2 + 5])}")

    trj_linked.sort_values(by=['frame', 'particle'], inplace=True)
    trj_linked.reset_index(drop=True, inplace=True)

    if 'parent_particle' in trj_linked.columns:
        self_parented_rows = trj_linked[trj_linked['particle'] == trj_linked['parent_particle']]
        if not self_parented_rows.empty:
            logging.error(
                f"CRITICAL ERROR ({mode.upper()}): Found {len(self_parented_rows)} rows where particle == parent_particle!")
            logging.error(self_parented_rows[['frame', 'particle', 'parent_particle']].head())
            for p_id, d_ids in final_cell_lineage.items():
                if p_id in d_ids:
                    logging.error(f"  Lineage map also shows self-parenting for ID {p_id}: daughters {d_ids}")

    logging.info(f'{mode.upper()}_TRACKING: Cell tracking finished.')
    initialize_next_daughter_id(trj_linked)
    logging.info(
        f"TRACKING_MODE_COMMON: next_available_daughter_id re-initialized after {mode} tracking to: {next_available_daughter_id}")
    return trj_linked, col_tuple, col_weights, final_cell_lineage


def track_cells_backward(path_in, raw_masks, min_cell_id=1, search_range=51, memory=5, neighbor_strategy='KDTree',
                         mitosis_max_dist_factor=0.9, mitosis_area_sum_min_factor=0.7,
                         mitosis_area_sum_max_factor=2.5, mitosis_daughter_area_similarity=0.5):
    """Wrapper for backward cell tracking."""
    return _track_cells_common_logic(path_in, raw_masks, min_cell_id, search_range, memory,
                                     neighbor_strategy, mitosis_max_dist_factor,
                                     mitosis_area_sum_min_factor, mitosis_area_sum_max_factor,
                                     mitosis_daughter_area_similarity, mode="backward",
                                     perform_mitosis_detection=True)


def track_cells_forward(path_in, raw_masks, min_cell_id=1, search_range=51, memory=5, neighbor_strategy='KDTree',
                        mitosis_max_dist_factor=0.9, mitosis_area_sum_min_factor=0.7,
                        mitosis_area_sum_max_factor=2.5, mitosis_daughter_area_similarity=0.5):
    """Wrapper for forward cell tracking."""
    return _track_cells_common_logic(path_in, raw_masks, min_cell_id, search_range, memory,
                                     neighbor_strategy, mitosis_max_dist_factor,
                                     mitosis_area_sum_min_factor, mitosis_area_sum_max_factor,
                                     mitosis_daughter_area_similarity, mode="forward",
                                     perform_mitosis_detection=True)


def track_cells(path_in, raw_masks, tracking_mode="Backward", min_cell_id=1,
                search_range=51, memory=5, neighbor_strategy='KDTree',
                mitosis_max_dist_factor=0.9, mitosis_area_sum_min_factor=0.7,
                mitosis_area_sum_max_factor=2.5, mitosis_daughter_area_similarity=0.5):
    """
    Main dispatcher function for cell tracking.
    Calls either forward or backward tracking based on tracking_mode.
    """
    if tracking_mode == "Forward":
        logging.info("Dispatching to FORWARD tracking (with mitosis detection).")
        return track_cells_forward(path_in, raw_masks, min_cell_id, search_range, memory,
                                   neighbor_strategy, mitosis_max_dist_factor,
                                   mitosis_area_sum_min_factor, mitosis_area_sum_max_factor,
                                   mitosis_daughter_area_similarity)
    elif tracking_mode == "Backward":
        logging.info("Dispatching to BACKWARD tracking (with fusion detection).")
        return track_cells_backward(path_in, raw_masks, min_cell_id, search_range, memory,
                                    neighbor_strategy, mitosis_max_dist_factor,
                                    mitosis_area_sum_min_factor, mitosis_area_sum_max_factor,
                                    mitosis_daughter_area_similarity)
    elif tracking_mode == "Basic": # "Forward (No Mitosis)"
        logging.info("Dispatching to FORWARD tracking (NO mitosis detection).")
        return _track_cells_common_logic(path_in, raw_masks, min_cell_id, search_range, memory,
                                         neighbor_strategy, mitosis_max_dist_factor,
                                         mitosis_area_sum_min_factor, mitosis_area_sum_max_factor,
                                         mitosis_daughter_area_similarity, mode="forward",
                                         perform_mitosis_detection=False)
    else:
        logging.error(f"Unknown tracking_mode: {tracking_mode}. Defaulting to Backward.")
        return track_cells_backward(path_in, raw_masks, min_cell_id, search_range, memory,
                                    neighbor_strategy, mitosis_max_dist_factor,
                                    mitosis_area_sum_min_factor, mitosis_area_sum_max_factor,
                                    mitosis_daughter_area_similarity)


def debug_mitosis_events(trj, save_path=None):
    """
    Diagnostic function to analyze mitosis events in the trajectory.
    Call this after tracking to see what happened.
    """
    logging.info("=== DEBUGGING MITOSIS EVENTS ===")

    if 'parent_particle' not in trj.columns:
        logging.info("No parent_particle column - no mitosis events detected")
        return

    # Find all parent-child relationships
    children_with_parents = trj.dropna(subset=['parent_particle']).copy()

    if children_with_parents.empty:
        logging.info("No parent-child relationships found")
        return

    # Group by parent to find mitosis events
    parent_groups = children_with_parents.groupby('parent_particle')

    mitosis_events = []
    for parent_id, group in parent_groups:
        unique_children = group['particle'].unique()
        if len(unique_children) >= 2:
            # This is a mitosis event
            parent_frames = trj[trj['particle'] == parent_id]['frame']
            children_frames = group['frame']

            mitosis_info = {
                'parent_id': int(parent_id),
                'children_ids': [int(x) for x in unique_children],
                'parent_frame_range': (parent_frames.min(), parent_frames.max()) if not parent_frames.empty else (None,
                                                                                                                  None),
                'children_frame_range': (children_frames.min(), children_frames.max()),
                'mitosis_frame': parent_frames.max() if not parent_frames.empty else None,
                'children_start_frame': children_frames.min()
            }
            mitosis_events.append(mitosis_info)

    logging.info(f"Found {len(mitosis_events)} mitosis events:")

    for i, event in enumerate(mitosis_events):
        logging.info(f"  Mitosis {i + 1}:")
        logging.info(f"    Parent: {event['parent_id']} (frames {event['parent_frame_range']})")
        logging.info(f"    Children: {event['children_ids']} (start frame {event['children_start_frame']})")
        if event['mitosis_frame'] is not None:
            gap = event['children_start_frame'] - event['mitosis_frame']
            logging.info(f"    Gap between parent end and children start: {gap} frames")

        # Check for multi-generational issues
        for child_id in event['children_ids']:
            grandchildren = trj[trj['parent_particle'] == child_id]
            if not grandchildren.empty:
                grandchild_ids = grandchildren['particle'].unique()
                logging.info(f"      Child {child_id} has grandchildren: {[int(x) for x in grandchild_ids]}")

    # Check for orphaned IDs (tracks that appear once and disappear)
    particle_counts = trj['particle'].value_counts()
    orphaned = particle_counts[particle_counts == 1]
    if not orphaned.empty:
        logging.warning(
            f"Found {len(orphaned)} tracks with only 1 timepoint (potential orphans): {orphaned.index.tolist()}")

    # Save detailed analysis if requested
    if save_path:
        analysis = []
        for _, row in trj.iterrows():
            analysis.append({
                'frame': row['frame'],
                'particle': row['particle'],
                'parent_particle': row.get('parent_particle', None),
                'x': row.get('x', None),
                'y': row.get('y', None)
            })

        import json
        with open(save_path, 'w') as f:
            json.dump({
                'mitosis_events': mitosis_events,
                'trajectory_data': analysis,
                'summary': {
                    'total_tracks': trj['particle'].nunique(),
                    'total_timepoints': len(trj),
                    'mitosis_events': len(mitosis_events),
                    'orphaned_tracks': len(orphaned)
                }
            }, f, indent=2)
        logging.info(f"Detailed analysis saved to {save_path}")

    return mitosis_events
