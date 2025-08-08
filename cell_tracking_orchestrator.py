# cell_tracking_orchestrator.py
# Manages the cell tracking process and subsequent cell state classification.
# V9 (Gemini - This Update): Corrected the tracking mode dispatcher to recognize "ILP".

import logging
import time
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QApplication
import pyqtgraph as pg
import os
import io
import json
import traceback
import sys
from cell_tracking import track_cells
from cell_drawing import prepare_mask_colors
from skimage.measure import regionprops, label as skimage_label
import cell_ui_callbacks

# --- TRACKASTRA IMPORT ---
try:
    from cell_tracking_trackastra import track_with_trackastra
    TRACKASTRA_AVAILABLE = True
except ImportError:
    TRACKASTRA_AVAILABLE = False
    logging.warning("cell_tracking_trackastra.py not found. Trackastra tracking mode will be unavailable.")
# --- END TRACKASTRA IMPORT ---

# --- ILP IMPORT ---
try:
    from cell_tracking_ilp import TrackingGraph, solve_ilp, reconstruct_trajectories, ILP_AVAILABLE
except ImportError:
    ILP_AVAILABLE = False
    logging.warning("cell_tracking_ilp.py not found. ILP tracking mode will be unavailable.")


def perform_cell_state_classification(main_app_state, ui_elements):
    """
    Classifies cells based on their overlap with class-specific masks.
    """
    # This function's logic remains the same
    trj = main_app_state.get('trj')
    id_masks = main_app_state.get('id_masks')
    loaded_class_specific_masks = main_app_state.get('loaded_class_specific_masks')
    params = main_app_state.get('params')
    track_states = main_app_state.get('track_states')
    cell_visibility = main_app_state.get('cell_visibility', {})

    if trj is None or trj.empty or id_masks is None or not loaded_class_specific_masks:
        logging.info("Skipping cell state classification: Missing trajectory, ID masks, or class-specific masks.")
        track_states.clear()
        if trj is not None and not trj.empty and 'particle' in trj.columns:
            for cell_id_val in pd.to_numeric(trj['particle'], errors='coerce').dropna().astype(int).unique():
                track_states[cell_id_val] = "N/A"  # Default if no classification
        return

    logging.info("Starting cell state classification...")
    track_states.clear()

    try:
        class_names_list = json.loads(params['Class Definitions (JSON list)'][0])
        if not isinstance(class_names_list, list) or not all(isinstance(cn, str) for cn in class_names_list):
            raise ValueError("Class Definitions must be a JSON list of strings.")
    except (json.JSONDecodeError, ValueError) as e_json_cls:
        logging.warning(
            f"Invalid Class Definitions ('{params['Class Definitions (JSON list)'][0]}'): {e_json_cls}. Skipping state classification.")
        if 'particle' in trj.columns:
            for cid_val_ts_err in pd.to_numeric(trj['particle'], errors='coerce').dropna().astype(int).unique():
                track_states[cid_val_ts_err] = "N/A"
        return

    unique_track_ids = pd.to_numeric(trj['particle'], errors='coerce').dropna().astype(int).unique()
    classification_threshold_distance = params.get('Classification Max Centroid Distance', (30.0, 'float'))[0]

    for track_id in unique_track_ids:
        if not cell_visibility.get(track_id, True):
            track_states[track_id] = "Hidden"
            continue

        dominant_class_for_track = "Unclassified"
        class_scores = pd.Series(dtype=int)

        track_frames_data = trj[pd.to_numeric(trj['particle'], errors='coerce') == track_id]

        for _, row_data in track_frames_data.iterrows():
            frame_idx = int(row_data['frame'])
            if not (0 <= frame_idx < id_masks.shape[2]):
                continue

            current_cell_mask_generic_binary = (id_masks[:, :, frame_idx] == track_id)
            if not np.any(current_cell_mask_generic_binary):
                continue

            props_generic_list = regionprops(current_cell_mask_generic_binary.astype(np.uint8))
            if not props_generic_list:
                continue
            centroid_generic = props_generic_list[0].centroid

            for class_name in class_names_list:
                if class_name in loaded_class_specific_masks and \
                        0 <= frame_idx < loaded_class_specific_masks[class_name].shape[2]:

                    class_mask_frame_binary = (loaded_class_specific_masks[class_name][:, :, frame_idx] > 0)
                    if not np.any(class_mask_frame_binary):
                        continue

                    labeled_class_mask_frame = skimage_label(class_mask_frame_binary)
                    props_class_list = regionprops(labeled_class_mask_frame)

                    min_dist_to_class_obj = float('inf')
                    for prop_c in props_class_list:
                        dist = np.sqrt(np.sum((np.array(centroid_generic) - np.array(prop_c.centroid)) ** 2))
                        if dist < min_dist_to_class_obj:
                            min_dist_to_class_obj = dist

                    if min_dist_to_class_obj < classification_threshold_distance:
                        class_scores[class_name] = class_scores.get(class_name, 0) + 1

        if not class_scores.empty:
            dominant_class_for_track = class_scores.idxmax()

        track_states[track_id] = dominant_class_for_track
    logging.info("Cell state classification finished.")


def initiate_cell_tracking(main_app_state, ui_elements, ui_callbacks, ui_actions):
    """
    Orchestrates the cell tracking process. Now includes a dispatcher for ILP and Trackastra modes.
    """
    from cell_undoredo import clear_undo_redo_stacks  # Keep local import if preferred

    # Handle None UI elements for batch processing
    if ui_elements is None:
        ui_elements = {}
        batch_mode = True
    else:
        batch_mode = False

    clear_undo_redo_stacks(ui_elements)

    if main_app_state.get('raw_masks') is None:
        if not batch_mode:
            QMessageBox.warning(ui_elements.get('win'), "No Data", "Please open a folder with mask data first.")
        return

    # Only perform UI operations if not in batch mode
    if not batch_mode:
        table_cell_selection = ui_elements.get('table_cell_selection')
        if table_cell_selection:
            table_cell_selection.setRowCount(0)

        pi_raw_img = ui_elements.get('pi_raw_img')
        pi_mask = ui_elements.get('pi_mask')

        # Clear existing plot items more robustly
        for item_dict_key, plot_widget_ref_key in [
            ('cell_ids_raw_img', 'pi_raw_img'),
            ('cell_ids_mask', 'pi_mask'),
            ('track_plots_per_cell', 'pi_raw_img')
        ]:
            item_dict = main_app_state.get(item_dict_key, {})
            plot_widget_ref = ui_elements.get(plot_widget_ref_key)
            if plot_widget_ref:
                for key_to_remove in list(item_dict.keys()):  # Iterate over a copy of keys
                    if item_dict[key_to_remove].scene():  # Check if item is in a scene
                        try:
                            plot_widget_ref.removeItem(item_dict[key_to_remove])
                        except Exception as e_remove:
                            logging.debug(f"Error removing item {key_to_remove} from {plot_widget_ref_key}: {e_remove}")
                item_dict.clear()  # Clear the dictionary itself

        # Disable Evaluate button before tracking
        eval_button_pre_track = ui_elements.get('button_widgets_map', {}).get('evaluate_tracking_ctc')
        if eval_button_pre_track:
            eval_button_pre_track.setEnabled(False)

    main_app_state['cell_visibility'].clear()
    main_app_state['cell_frame_presence'].clear()
    main_app_state['track_data_per_frame'].clear()
    main_app_state['cell_y'].clear()
    main_app_state['cell_x'].clear()
    main_app_state['cell_lineage'].clear()
    main_app_state['ancestry'].clear()
    main_app_state['track_states'].clear()
    main_app_state['next_available_daughter_id'] = 1  # Reset for new tracking
    main_app_state['last_saved_res_path'] = None  # Reset last saved RES path

    main_app_state['merged_masks'] = main_app_state['raw_masks'].copy()

    start_time = time.time()
    main_app_state['captured_tracking_log'] = ""  # Reset captured log
    log_capture_string_io = io.StringIO()

    tracking_logger = logging.getLogger()  # Get root logger
    original_handlers = tracking_logger.handlers[:]  # Store original handlers
    original_level = tracking_logger.level  # Store original level

    temp_log_handler = logging.StreamHandler(log_capture_string_io)
    temp_log_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s'))
    temp_log_handler.setLevel(logging.INFO)  # Capture INFO and above for tracking log

    tracking_logger.addHandler(temp_log_handler)

    tracking_successful_flag = False
    try:
        with pg.BusyCursor():  # Ensure BusyCursor is used
            params = main_app_state['params']
            tracking_mode_input_val = params['Tracking Mode'][0]

            # V9 FIX: Corrected valid modes list to match UI ("ILP")
            valid_modes_list = ["Forward", "Backward", "Basic", "ILP", "Trackastra"]
            tracking_mode_to_use_val = "Backward"  # Default
            if tracking_mode_input_val in valid_modes_list:
                tracking_mode_to_use_val = tracking_mode_input_val
            else:
                logging.warning(f"Invalid tracking mode '{tracking_mode_input_val}'. Defaulting to 'Backward'.")

            path_for_tracking_param = main_app_state['path_in']
            if params['Enable Advanced File Structure'][0]:
                path_for_tracking_param = os.path.join(main_app_state['path_in'], params['Raw Image Folder Name'][0])

            # --- New logic block to dispatch to different tracking methods ---
            if tracking_mode_to_use_val == "Trackastra":
                if not TRACKASTRA_AVAILABLE:
                    if not batch_mode:
                        QMessageBox.critical(ui_elements.get('win'), "Module Not Found",
                                             "The 'trackastra' library or cell_tracking_trackastra.py could not be imported. Trackastra tracking is unavailable.")
                    raise ImportError("Trackastra module missing.")

                trj_result, col_tuple_result, col_weights_result, cell_lineage_result = track_with_trackastra(
                    main_app_state
                )

            # V9 FIX: Corrected elif to check for "ILP"
            elif tracking_mode_to_use_val == "ILP":
                if not ILP_AVAILABLE:
                    msg = (
                        "The 'ILP' tracking mode requires the 'cell_tracking_ilp.py' module, but it could not be found.\n\n"
                        "Please ensure that 'cell_tracking_ilp.py' is in the same directory as the main application script.\n\n"
                        "Check the log console for more detailed import error messages.")
                    if not batch_mode:
                        QMessageBox.critical(ui_elements.get('win'), "Module Not Found", msg)
                    raise ImportError("ILP module missing, cannot run ILP tracking.")

                logging.info("--- Starting ILP (Graph-based) Tracking ---")
                graph = TrackingGraph(
                    masks=main_app_state['raw_masks'],
                    max_distance=params['ILP Max Search Distance'][0],
                    cost_weight_transition=params['ILP Transition Cost Weight'][0],
                    cost_weight_mitosis=params['ILP Mitosis Cost'][0],
                    cost_weight_fusion=params['ILP Fusion Cost'][0],
                    cost_appearance=params['ILP Appearance Cost'][0],
                    cost_disappearance=params['ILP Disappearance Cost'][0]
                )
                graph.build()

                # --- FIX: Get solver and Gurobi params from UI ---
                selected_solver = params.get('ILP Solver', ('scipy',))[0]
                gurobi_params = None
                if selected_solver == 'gurobi':
                    gurobi_params = {
                        "WLSACCESSID": params.get('Gurobi WLSACCESSID', ('',))[0],
                        "WLSSECRET": params.get('Gurobi WLSSECRET', ('',))[0],
                        "LICENSEID": params.get('Gurobi LICENSEID', (0,))[0]
                    }

                selected_edges = solve_ilp(graph, solver=selected_solver, gurobi_params=gurobi_params)

                if selected_edges is None:
                    if not batch_mode:
                        QMessageBox.warning(ui_elements.get('win'), "ILP Solver Failed",
                                            "The ILP solver failed to find a solution. Check the logs for details.")
                    trj_result, cell_lineage_result = pd.DataFrame(), {}
                else:
                    trj_result, cell_lineage_result = reconstruct_trajectories(graph, selected_edges)
                col_tuple_result, col_weights_result = {}, {}
                
                # Get the background label that was used during ILP tracking
                from cell_tracking_ilp import determine_background_label
                background_label_result = determine_background_label(main_app_state['raw_masks'])


            else:  # Existing trackpy-based methods
                logging.info(f"--- Starting Trackpy-based Tracking (Mode: {tracking_mode_to_use_val}) ---")
                trj_result, col_tuple_result, col_weights_result, cell_lineage_result = track_cells(
                    path_for_tracking_param,
                    main_app_state['merged_masks'],
                    tracking_mode=tracking_mode_to_use_val,
                    min_cell_id=1,
                    search_range=params['Trackpy search range'][0],
                    memory=params['Trackpy memory'][0],
                    neighbor_strategy=params['Trackpy neighbor strategy'][0],
                    mitosis_max_dist_factor=params['Mitosis Max Distance Factor'][0],
                    mitosis_area_sum_min_factor=params['Mitosis Area Sum Min Factor'][0],
                    mitosis_area_sum_max_factor=params['Mitosis Area Sum Max Factor'][0],
                    mitosis_daughter_area_similarity=params['Mitosis Daughter Area Similarity'][0]
                )

            # --- The rest of the function processes the results from either method ---
            main_app_state['trj'] = trj_result
            main_app_state['col_tuple'] = col_tuple_result
            main_app_state['col_weights'] = col_weights_result

            if isinstance(cell_lineage_result, dict) and not main_app_state['trj'].empty and 'particle' in \
                    main_app_state['trj']:
                main_app_state['cell_lineage'] = {
                    int(k): [int(i) for i in v] for k, v in cell_lineage_result.items() if v
                }
            else:
                main_app_state['cell_lineage'] = {}

            # Rebuild ancestry from the final cell_lineage
            main_app_state['ancestry'].clear()
            if main_app_state['cell_lineage']:
                for parent_id_cl, daughter_ids_cl in main_app_state['cell_lineage'].items():
                    for daughter_id_cl in daughter_ids_cl:
                        main_app_state['ancestry'].setdefault(daughter_id_cl, []).append(parent_id_cl)
                for child_id_anc, parent_ids_anc in main_app_state['ancestry'].items():
                    main_app_state['ancestry'][child_id_anc] = sorted(list(set(parent_ids_anc)))

            main_app_state['has_lineage'] = bool(main_app_state['cell_lineage'] or main_app_state['ancestry'])
            tracking_successful_flag = True

            if main_app_state['trj'] is None or main_app_state['trj'].empty:
                if not batch_mode:
                    QMessageBox.information(ui_elements.get('win'), "Tracking Info", "No cells were tracked.")
                    eval_button_no_tracks = ui_elements.get('button_widgets_map', {}).get('evaluate_tracking_ctc')
                    if eval_button_no_tracks:
                        eval_button_no_tracks.setEnabled(False)
            else:
                main_app_state['initial_trj'] = main_app_state['trj'].copy()

                # Use the background label detected during tracking
                if tracking_mode_to_use_val == "ILP":
                    id_masks_res, cell_ids_res, color_list_res, background_id_res = prepare_mask_colors(
                        main_app_state['merged_masks'], main_app_state['trj']
                    )
                    # Override background_id with the detected one from ILP tracking
                    if background_label_result is not None:
                        background_id_res = background_label_result
                else:
                    id_masks_res, cell_ids_res, color_list_res, background_id_res = prepare_mask_colors(
                        main_app_state['merged_masks'], main_app_state['trj']
                    )
                main_app_state['id_masks'] = id_masks_res
                main_app_state['cell_ids'] = cell_ids_res
                main_app_state['color_list'] = color_list_res
                main_app_state['background_id'] = background_id_res
                main_app_state['cell_color_idx'] = {cell_id: idx for idx, cell_id in enumerate(cell_ids_res)}
                main_app_state['id_masks_initial'] = main_app_state['id_masks'].copy() if main_app_state[
                                                                                              'id_masks'] is not None else None

                perform_cell_state_classification(main_app_state, ui_elements)
                
                # Ensure the trajectory data has a 'state' column populated from track_states
                if 'track_states' in main_app_state and main_app_state['track_states']:
                    track_states = main_app_state['track_states']
                    # Add state column to trajectory data
                    main_app_state['trj']['state'] = main_app_state['trj']['particle'].map(track_states).fillna("unknown")
                    logging.info(f"Added state column to trajectory data with {len(track_states)} unique states")
                else:
                    # If no track_states available, create default state column
                    main_app_state['trj']['state'] = 'unknown'
                    logging.info("No track states available, created default 'unknown' state column")

                if not all(col in main_app_state['trj'].columns for col in ['particle', 'frame', 'x', 'y']):
                    logging.error("Trajectory missing required columns post-tracking.")
                else:
                    for i_frame_val_coords in sorted(main_app_state['trj']['frame'].unique()):
                        frame_data_coords = main_app_state['trj'][main_app_state['trj']['frame'] == i_frame_val_coords]
                        for _, row_coords in frame_data_coords.iterrows():
                            cell_id_val_coords = int(row_coords['particle'])
                            main_app_state['cell_y'][cell_id_val_coords, i_frame_val_coords] = row_coords['y']
                            main_app_state['cell_x'][cell_id_val_coords, i_frame_val_coords] = row_coords['x']

                if 'particle' in main_app_state['trj'].columns:
                    trj_temp_p_int_vis = main_app_state['trj'].copy()
                    trj_temp_p_int_vis['particle'] = pd.to_numeric(trj_temp_p_int_vis['particle'],
                                                                   errors='coerce').dropna().astype(int)
                    main_app_state['cell_frame_presence'] = trj_temp_p_int_vis.groupby('particle')['frame'].apply(
                        set).to_dict()
                    unique_particles_vis = trj_temp_p_int_vis['particle'].unique()
                    main_app_state['cell_visibility'] = {cid_vis: True for cid_vis in unique_particles_vis if
                                                         cid_vis != main_app_state['background_id']}
                    
                    # Safety check: ensure all cells in id_masks are also in cell_visibility
                    if main_app_state.get('id_masks') is not None:
                        all_mask_ids = set(np.unique(main_app_state['id_masks']))
                        all_mask_ids.discard(main_app_state['background_id'])  # Remove background
                        for mask_id in all_mask_ids:
                            if mask_id not in main_app_state['cell_visibility']:
                                main_app_state['cell_visibility'][mask_id] = True
                                logging.info(f"Added missing cell ID {mask_id} to cell_visibility")
                else:
                    main_app_state['cell_frame_presence'], main_app_state['cell_visibility'] = {}, {}

                min_duration_param = params.get('Min Tracklet Duration', (1,))[0]
                if min_duration_param > 1 and not main_app_state['trj'].empty and 'particle' in main_app_state[
                    'trj'].columns:
                    track_lengths = main_app_state['trj'].groupby('particle')['frame'].nunique()
                    short_ids_filter = track_lengths[track_lengths < min_duration_param].index
                    short_ids_filter_int = pd.to_numeric(short_ids_filter, errors='coerce').dropna().astype(int)
                    
                    if len(short_ids_filter_int) > 0:
                        logging.info(f"Min Tracklet Duration filtering: {len(short_ids_filter_int)} cells with duration < {min_duration_param} frames will be hidden")
                        logging.info(f"Short track IDs: {sorted(short_ids_filter_int)}")
                        logging.info(f"Track lengths for short tracks: {track_lengths[track_lengths < min_duration_param].to_dict()}")
                    
                    for s_id_f in short_ids_filter_int:
                        if s_id_f in main_app_state['cell_visibility']:
                            main_app_state['cell_visibility'][s_id_f] = False
                            logging.info(f"Set cell {s_id_f} to invisible due to short track length")
                        if main_app_state['id_masks'] is not None and main_app_state[
                            'id_masks_initial'] is not None and s_id_f != main_app_state['background_id']:
                            main_app_state['id_masks'][main_app_state['id_masks_initial'] == s_id_f] = main_app_state[
                                'background_id']

                # Only perform UI operations if not in batch mode
                if not batch_mode:
                    v_raw_img = ui_elements.get('v_raw_img')
                    raw_imgs_data = main_app_state.get('raw_imgs')
                    current_frame_idx_val = main_app_state.get('current_frame_index', 0)

                    if not (raw_imgs_data is not None and 0 <= current_frame_idx_val < raw_imgs_data.shape[2]):
                        main_app_state['current_frame_index'] = 0
                    if raw_imgs_data is not None and raw_imgs_data.shape[2] > 0 and v_raw_img:
                        v_raw_img.setCurrentIndex(main_app_state['current_frame_index'])

                    unique_particles_for_plot_items = main_app_state['trj']['particle'].unique() if 'particle' in \
                                                                                                    main_app_state[
                                                                                                        'trj'] else []
                    for cell_id_val_pi in unique_particles_for_plot_items:
                        cell_id_int_pi = int(cell_id_val_pi)
                        if cell_id_int_pi == main_app_state['background_id']: continue

                        for item_dict_key_pi, anchor_pos_pi, plot_widget_ref_key_pi in [
                            ('cell_ids_raw_img', (0.5, 1), 'pi_raw_img'),
                            ('cell_ids_mask', (0.5, 0.5), 'pi_mask'),
                            ('track_plots_per_cell', None, 'pi_raw_img')
                        ]:
                            item_dict_pi = main_app_state.get(item_dict_key_pi, {})
                            plot_widget_item_pi = ui_elements.get(plot_widget_ref_key_pi)

                            if plot_widget_item_pi:
                                if cell_id_int_pi not in item_dict_pi:
                                    item_dict_pi[cell_id_int_pi] = pg.TextItem(
                                        anchor=anchor_pos_pi) if anchor_pos_pi else pg.PlotDataItem()
                                if not item_dict_pi[cell_id_int_pi].scene():
                                    plot_widget_item_pi.addItem(item_dict_pi[cell_id_int_pi])

                    if ui_callbacks:
                        ui_callbacks.handle_populate_cell_table(main_app_state, ui_elements, ui_callbacks)

                    v_mask = ui_elements.get('v_mask')
                    if v_mask and main_app_state['id_masks'] is not None and not main_app_state['trj'].empty and \
                            main_app_state['color_list'] and main_app_state['cell_color_idx']:

                        all_drawable_ids_lut = [int(i_lut) for i_lut in main_app_state['cell_color_idx'].keys() if
                                                isinstance(i_lut, (int, float)) and pd.notna(i_lut)]
                        lut_size_val = max(all_drawable_ids_lut) + 1 if all_drawable_ids_lut else 1
                        actual_lut_val = np.zeros((lut_size_val, 3), dtype=np.uint8)

                        bg_col_idx_lut = main_app_state['cell_color_idx'].get(main_app_state['background_id'])
                        bg_color_lut = main_app_state['color_list'][
                            bg_col_idx_lut] if bg_col_idx_lut is not None and 0 <= bg_col_idx_lut < len(
                            main_app_state['color_list']) else (20, 20, 20)

                        if main_app_state['background_id'] >= 0 and main_app_state['background_id'] < lut_size_val:
                            actual_lut_val[main_app_state['background_id']] = bg_color_lut

                        for cid_val_lut, c_idx_lut in main_app_state['cell_color_idx'].items():
                            if isinstance(cid_val_lut, (int, float)) and pd.notna(cid_val_lut):
                                cid_val_int_lut = int(cid_val_lut)
                                if cid_val_int_lut >= 0 and cid_val_int_lut < lut_size_val and c_idx_lut < len(
                                        main_app_state['color_list']):
                                    actual_lut_val[cid_val_int_lut] = main_app_state['color_list'][c_idx_lut]

                        v_mask.setImage(main_app_state['id_masks'], axes={'x': 1, 'y': 0, 't': 2})
                        if v_mask.imageItem: v_mask.imageItem.setLookupTable(actual_lut_val)
                        v_mask.setLevels(min=0, max=lut_size_val - 1 if lut_size_val > 0 else 0)
                        valid_mask_idx_val = 0 <= main_app_state['current_frame_index'] < main_app_state['id_masks'].shape[
                            2]
                        v_mask.setCurrentIndex(main_app_state['current_frame_index'] if valid_mask_idx_val else 0)
                        pi_mask = ui_elements.get('pi_mask')
                        if pi_mask: pi_mask.setTitle(f"Frame: {v_mask.currentIndex} (ID Mask)")

                    lineage_tree_widget = ui_elements.get('lineage_tree_widget')
                    if lineage_tree_widget:
                        lineage_tree_widget.set_data(
                            main_app_state['trj'], main_app_state['color_list'], main_app_state['cell_color_idx'],
                            main_app_state['cell_visibility'], main_app_state['track_states'], main_app_state['ancestry']
                        )

                    # Clear all caches when tracking is completed
                    if 'cell_ui_callbacks' in sys.modules:
                        cell_ui_callbacks._clear_all_caches()
                    
                    if ui_callbacks:
                        ui_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)

        if tracking_successful_flag:
            logging.info(f'Cell tracking completed in {time.time() - start_time:.2f}s')
            if not batch_mode:
                QApplication.processEvents()

    except Exception as e_track:
        logging.error(f"Error during cell tracking operations: {e_track}\n{traceback.format_exc()}")
        if not batch_mode:
            QMessageBox.critical(ui_elements.get('win'), "Tracking Error", f"An error occurred during tracking: {e_track}")
        main_app_state['trj'] = pd.DataFrame()
        main_app_state['cell_lineage'] = {}
        main_app_state['ancestry'] = {}
        main_app_state['has_lineage'] = False
        if not batch_mode:
            lineage_tree_widget = ui_elements.get('lineage_tree_widget')
            if lineage_tree_widget:
                lineage_tree_widget.set_data(pd.DataFrame(), [], {}, {}, {}, {})
    finally:
        tracking_logger.handlers = original_handlers
        tracking_logger.setLevel(original_level)
        temp_log_handler.close()
        main_app_state['captured_tracking_log'] = log_capture_string_io.getvalue()
        log_capture_string_io.close()
        logging.info(f"Captured tracking log length: {len(main_app_state['captured_tracking_log'])} characters.")
