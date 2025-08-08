# cell_file_operations.py
# Handles data loading operations, including single folder and batch processing.
# V2 (Gemini - This Update): Fixed "Optimize ILP" button enablement after data loading.
# V3 (Gemini - This Update): Added functionality to load previously analyzed data.

import os
import json
import logging
import traceback
import time
import io  # For batch log capture
import sys
import pandas as pd
import numpy as np
import tifffile
import pyqtgraph as pg  # For BusyCursor
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog, QApplication
from PyQt5.QtCore import Qt

from cell_io import read_img_sequence, save_results
from cell_tracking import track_cells
from cell_drawing import prepare_mask_colors, generate_distinct_colors
import cell_ui_callbacks


def load_previous_results(main_app_state, ui_elements, ui_callbacks, ui_actions):
    """
    Loads previously analyzed data from a saved results folder.
    """
    from cell_undoredo import clear_undo_redo_stacks

    clear_undo_redo_stacks(ui_elements)

    dialog_title = 'Select Previously Analyzed Results Folder'
    start_directory_for_dialog = main_app_state.get('path_in', os.path.expanduser("~"))
    if main_app_state.get('path_in') and os.path.isdir(main_app_state['path_in']):
        start_directory_for_dialog = os.path.dirname(main_app_state['path_in'])

    selected_path_from_dialog = QFileDialog.getExistingDirectory(
        ui_elements.get('win'),
        dialog_title,
        start_directory_for_dialog
    )
    if not selected_path_from_dialog:
        logging.info('Canceled results folder selection')
        return

    # Check if this is a valid results folder
    tracks_csv_path = os.path.join(selected_path_from_dialog, 'tracks.csv')
    lineage_csv_path = os.path.join(selected_path_from_dialog, 'lineage_info.csv')
    experiment_params_path = os.path.join(selected_path_from_dialog, 'experiment_parameters.csv')
    
    if not os.path.exists(tracks_csv_path):
        QMessageBox.critical(ui_elements.get('win'), "Invalid Results Folder", 
                           f"Selected folder does not contain 'tracks.csv'. Please select a valid results folder.\n\nPath: {selected_path_from_dialog}")
        return

    # Try to find the original data folder
    original_data_folder = None
    results_folder_name = os.path.basename(selected_path_from_dialog)
    
    # Look for original data folder in common locations
    parent_dir = os.path.dirname(selected_path_from_dialog)
    possible_original_folders = [
        # Remove the experiment suffix to get original folder name
        os.path.join(parent_dir, results_folder_name.split('_Exp_')[0]),
        # Look for folders with similar names
        os.path.join(parent_dir, results_folder_name.split('_Exp_')[0] + '_mask_all'),
        os.path.join(parent_dir, results_folder_name.split('_Exp_')[0] + '_GT'),
    ]
    
    for folder in possible_original_folders:
        if os.path.exists(folder):
            original_data_folder = folder
            break
    
    if not original_data_folder:
        # Ask user to select original data folder
        original_folder_dialog = QFileDialog.getExistingDirectory(
            ui_elements.get('win'),
            'Select Original Data Folder (for raw images and masks)',
            parent_dir
        )
        if original_folder_dialog:
            original_data_folder = original_folder_dialog
        else:
            QMessageBox.warning(ui_elements.get('win'), "No Original Data", 
                              "Could not find original data folder. The cell editor may not work properly without raw images and masks.")
            original_data_folder = None

    # Show progress dialog for data loading
    progress_dialog = QProgressDialog("Loading previous results...", "Cancel", 0, 100, ui_elements.get('win'))
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setAutoClose(False)
    progress_dialog.show()
    
    with pg.BusyCursor():
        try:
            progress_dialog.setValue(10)
            progress_dialog.setLabelText("Loading trajectory data...")
            QApplication.processEvents()
            
            # Load tracks.csv
            trj = pd.read_csv(tracks_csv_path)
            if trj.empty:
                QMessageBox.critical(ui_elements.get('win'), "Empty Results", 
                                   "The tracks.csv file is empty. Cannot load results.")
                return
            
            # Convert particle and frame columns to numeric
            trj['particle'] = pd.to_numeric(trj['particle'], errors='coerce')
            trj['frame'] = pd.to_numeric(trj['frame'], errors='coerce')
            
            # Load original data if available
            raw_imgs = None
            raw_masks = None
            
            # First try to load from saved PNG files in the results folder
            raw_images_dir = os.path.join(selected_path_from_dialog, 'raw_images')
            if os.path.exists(raw_images_dir):
                progress_dialog.setValue(15)
                progress_dialog.setLabelText("Loading saved raw images...")
                QApplication.processEvents()
                
                try:
                    raw_imgs = read_img_sequence(raw_images_dir, 'png')
                    logging.info(f"Successfully loaded raw images from saved PNG files: {raw_images_dir}")
                except Exception as e:
                    logging.warning(f"Could not load raw images from saved PNG files: {e}")
                    raw_imgs = None
            
            # If no saved PNG files, try to load from original data folder
            if raw_imgs is None and original_data_folder:
                progress_dialog.setValue(15)
                progress_dialog.setLabelText("Loading original raw images...")
                QApplication.processEvents()
                
                try:
                    # Try to load raw images
                    raw_imgs = read_img_sequence(original_data_folder, 
                                                main_app_state['params']['Raw image extension'][0])
                    logging.info(f"Successfully loaded raw images from {original_data_folder}")
                except Exception as e:
                    logging.warning(f"Could not load raw images from {original_data_folder}: {e}")
                    raw_imgs = None
                
                progress_dialog.setValue(20)
                progress_dialog.setLabelText("Loading original masks...")
                QApplication.processEvents()
                
                try:
                    # Try to load masks (look for mask folder)
                    mask_folder_suffix = main_app_state['params']['Mask folder suffix'][0]
                    mask_folder = original_data_folder + mask_folder_suffix
                    
                    if os.path.exists(mask_folder):
                        raw_masks = read_img_sequence(mask_folder, 
                                                     main_app_state['params']['Mask extension'][0])
                        logging.info(f"Successfully loaded masks from {mask_folder}")
                    else:
                        # Try the original folder itself for masks
                        raw_masks = read_img_sequence(original_data_folder, 
                                                     main_app_state['params']['Mask extension'][0])
                        logging.info(f"Successfully loaded masks from {original_data_folder}")
                except Exception as e:
                    logging.warning(f"Could not load masks: {e}")
                    raw_masks = None
            
            progress_dialog.setValue(30)
            progress_dialog.setLabelText("Loading lineage information...")
            QApplication.processEvents()
            
            # Load lineage information
            cell_lineage = {}
            ancestry = {}
            
            # Try to load from lineage_relationships.csv (new format)
            lineage_relationships_path = os.path.join(selected_path_from_dialog, 'lineage_relationships.csv')
            if os.path.exists(lineage_relationships_path):
                try:
                    lineage_df = pd.read_csv(lineage_relationships_path)
                    for _, row in lineage_df.iterrows():
                        parent_id = int(row['parent_id'])
                        daughter_id = int(row['daughter_id'])
                        cell_lineage.setdefault(parent_id, []).append(daughter_id)
                        ancestry.setdefault(daughter_id, []).append(parent_id)
                    logging.info(f"Loaded lineage from {lineage_relationships_path}")
                except Exception as e:
                    logging.warning(f"Could not load lineage from {lineage_relationships_path}: {e}")
            
            # Fallback to lineage_info.csv (old format)
            elif os.path.exists(lineage_csv_path):
                try:
                    lineage_df = pd.read_csv(lineage_csv_path)
                    for _, row in lineage_df.iterrows():
                        parent_id = int(row['parent_id'])
                        daughter_id = int(row['daughter_id'])
                        cell_lineage.setdefault(parent_id, []).append(daughter_id)
                        ancestry.setdefault(daughter_id, []).append(parent_id)
                    logging.info(f"Loaded lineage from {lineage_csv_path}")
                except Exception as e:
                    logging.warning(f"Could not load lineage from {lineage_csv_path}: {e}")
            
            # If no lineage file found, try to extract from trajectory data
            if not cell_lineage and 'parent_particle' in trj.columns:
                logging.info("No lineage file found, extracting lineage from trajectory data...")
                trj_copy = trj.copy()
                trj_copy['particle'] = pd.to_numeric(trj_copy['particle'], errors='coerce')
                trj_copy['parent_particle'] = pd.to_numeric(trj_copy['parent_particle'], errors='coerce')
                trj_copy.dropna(subset=['particle', 'parent_particle'], inplace=True)
                
                for _, row in trj_copy.iterrows():
                    parent_id = int(row['parent_particle'])
                    daughter_id = int(row['particle'])
                    if parent_id > 0:  # Skip if parent is 0 or negative
                        cell_lineage.setdefault(parent_id, []).append(daughter_id)
                        ancestry.setdefault(daughter_id, []).append(parent_id)
                
                # Remove duplicates
                for parent_id in cell_lineage:
                    cell_lineage[parent_id] = list(set(cell_lineage[parent_id]))
                for daughter_id in ancestry:
                    ancestry[daughter_id] = list(set(ancestry[daughter_id]))
                
                logging.info(f"Extracted lineage from trajectory: {len(cell_lineage)} parent-daughter relationships")
            
            progress_dialog.setValue(50)
            progress_dialog.setLabelText("Loading experiment parameters...")
            QApplication.processEvents()
            
            # Load experiment parameters
            pixel_scale = 0.87e-6  # Default
            if os.path.exists(experiment_params_path):
                params_df = pd.read_csv(experiment_params_path)
                pixel_scale_param = params_df[params_df['Parameter'] == 'pixel_scale_microns']
                if not pixel_scale_param.empty:
                    pixel_scale = float(pixel_scale_param.iloc[0]['Value']) * 1e-6  # Convert microns to meters
            
            # Load settings from YAML if available
            settings_yaml_path = os.path.join(selected_path_from_dialog, 'experiment_settings.yaml')
            if os.path.exists(settings_yaml_path):
                try:
                    import yaml
                    with open(settings_yaml_path, 'r') as f:
                        saved_params = yaml.safe_load(f)
                    # Update main_app_state params with saved settings
                    for key, value in saved_params.items():
                        if key in main_app_state['params']:
                            if isinstance(value, list) and len(value) >= 1:
                                main_app_state['params'][key] = (value[0],) + main_app_state['params'][key][1:]
                except Exception as e:
                    logging.warning(f"Could not load settings from YAML: {e}")
            
            progress_dialog.setValue(70)
            progress_dialog.setLabelText("Loading ID masks...")
            QApplication.processEvents()
            
            # Try to load ID masks
            id_masks_dir = os.path.join(selected_path_from_dialog, 'Id_masks_raw_per_frame_original')
            id_masks = None
            if os.path.exists(id_masks_dir):
                try:
                    # Determine file extension
                    files = [f for f in os.listdir(id_masks_dir) if f.startswith('id_masks_raw_original_')]
                    if files:
                        ext = files[0].split('.')[-1]
                        id_masks = read_img_sequence(id_masks_dir, ext)
                except Exception as e:
                    logging.warning(f"Could not load ID masks: {e}")
            
            progress_dialog.setValue(90)
            progress_dialog.setLabelText("Setting up application state...")
            QApplication.processEvents()
            
            # Reset application state
            main_app_state['trj'] = trj
            main_app_state['cell_lineage'] = cell_lineage
            main_app_state['ancestry'] = ancestry
            main_app_state['has_lineage'] = bool(cell_lineage or ancestry)
            main_app_state['id_masks'] = id_masks
            main_app_state['id_masks_initial'] = id_masks.copy() if id_masks is not None else None
            
            # Store the original data path for CTC evaluation
            if original_data_folder:
                main_app_state['path_in'] = original_data_folder
                logging.info(f"Stored original data path for CTC evaluation: {original_data_folder}")
            else:
                # If no original folder found, use the results folder name to construct a path
                results_folder_name = os.path.basename(selected_path_from_dialog)
                if '_Exp_' in results_folder_name:
                    original_folder_name = results_folder_name.split('_Exp_')[0]
                    parent_dir = os.path.dirname(selected_path_from_dialog)
                    constructed_path = os.path.join(parent_dir, original_folder_name)
                    main_app_state['path_in'] = constructed_path
                    logging.info(f"Constructed original data path for CTC evaluation: {constructed_path}")
                else:
                    main_app_state['path_in'] = selected_path_from_dialog
                    logging.warning(f"Could not determine original data path, using results folder: {selected_path_from_dialog}")
            
            # Log lineage information for debugging
            logging.info(f"Loaded lineage data: {len(cell_lineage)} parent-daughter relationships")
            logging.info(f"Loaded ancestry data: {len(ancestry)} daughter-parent relationships")
            if cell_lineage:
                logging.info(f"Sample lineage entries: {dict(list(cell_lineage.items())[:5])}")
            if ancestry:
                logging.info(f"Sample ancestry entries: {dict(list(ancestry.items())[:5])}")
            
            # Set up raw images and masks for the cell editor
            if raw_imgs is not None:
                # Normalize image data for proper display
                if raw_imgs.dtype != np.uint8:
                    # Convert to uint8 if not already
                    if np.max(raw_imgs) > 1.0:  # Likely 16-bit or higher
                        raw_imgs = (raw_imgs / np.max(raw_imgs) * 255).astype(np.uint8)
                    else:  # Likely float 0-1
                        raw_imgs = (raw_imgs * 255).astype(np.uint8)
                
                main_app_state['raw_imgs'] = raw_imgs
                logging.info(f"Set raw images with shape: {raw_imgs.shape}")
                logging.info(f"Raw images data type: {raw_imgs.dtype}")
                logging.info(f"Raw images min/max: {np.min(raw_imgs)}/{np.max(raw_imgs)}")
                logging.info(f"Raw images mean/std: {np.mean(raw_imgs):.2f}/{np.std(raw_imgs):.2f}")
            else:
                # Create placeholder if no raw images available
                if id_masks is not None:
                    main_app_state['raw_imgs'] = np.zeros_like(id_masks, dtype=np.uint8)
                    logging.warning("No raw images available, using placeholder")
                else:
                    main_app_state['raw_imgs'] = None
            
            if raw_masks is not None:
                main_app_state['raw_masks'] = raw_masks
                logging.info(f"Set raw masks with shape: {raw_masks.shape}")
            else:
                # Use ID masks as raw masks if no separate masks available
                if id_masks is not None:
                    main_app_state['raw_masks'] = id_masks
                    logging.info("Using ID masks as raw masks")
                else:
                    main_app_state['raw_masks'] = None
            
            # Extract cell IDs and set up visibility
            cell_ids = sorted(trj['particle'].unique())
            main_app_state['cell_ids'] = cell_ids
            main_app_state['cell_visibility'] = {int(cid): True for cid in cell_ids}
            
            # Extract cell coordinates for overlay display
            cell_x = {}
            cell_y = {}
            for _, row in trj.iterrows():
                cell_id = int(row['particle'])
                frame = int(row['frame'])
                if 'x' in row and 'y' in row:
                    cell_x[(cell_id, frame)] = float(row['x'])
                    cell_y[(cell_id, frame)] = float(row['y'])
            
            main_app_state['cell_x'] = cell_x
            main_app_state['cell_y'] = cell_y
            
            # Extract track states if available
            track_states = {}
            
            # First try to load from separate track_states.csv file
            track_states_file = os.path.join(selected_path_from_dialog, 'track_states.csv')
            if os.path.exists(track_states_file):
                try:
                    track_states_df = pd.read_csv(track_states_file)
                    for _, row in track_states_df.iterrows():
                        track_states[int(row['cell_id'])] = str(row['state'])
                    logging.info(f"Loaded track states from {track_states_file}")
                except Exception as e:
                    logging.warning(f"Could not load track states from {track_states_file}: {e}")
            
            # If no separate file or loading failed, try to extract from state column in tracks.csv
            if not track_states and 'state' in trj.columns:
                for cid in cell_ids:
                    state_values = trj[trj['particle'] == cid]['state'].unique()
                    if len(state_values) > 0 and pd.notna(state_values[0]):
                        track_states[int(cid)] = str(state_values[0])
                logging.info(f"Extracted track states from tracks.csv state column")
            
            # Set default states for any missing cells
            for cid in cell_ids:
                if int(cid) not in track_states:
                    track_states[int(cid)] = 'unknown'
            
            main_app_state['track_states'] = track_states
            logging.info(f"Final track states loaded: {len(track_states)} cells")

            # Load additional state data for complete functionality restoration
            progress_dialog.setValue(85)
            progress_dialog.setLabelText("Loading additional state data...")
            QApplication.processEvents()

            # Load cell visibility
            cell_visibility = {}
            visibility_file = os.path.join(selected_path_from_dialog, 'cell_visibility.csv')
            if os.path.exists(visibility_file):
                try:
                    visibility_df = pd.read_csv(visibility_file)
                    for _, row in visibility_df.iterrows():
                        cell_visibility[int(row['cell_id'])] = bool(row['visible'])
                    logging.info(f"Loaded cell visibility from {visibility_file}")
                except Exception as e:
                    logging.warning(f"Could not load cell visibility from {visibility_file}: {e}")
            
            # If no visibility file, create default visibility
            if not cell_visibility:
                cell_visibility = {int(cid): True for cid in cell_ids}
            
            main_app_state['cell_visibility'] = cell_visibility

            # Load cell frame presence
            cell_frame_presence = {}
            frame_presence_file = os.path.join(selected_path_from_dialog, 'cell_frame_presence.csv')
            if os.path.exists(frame_presence_file):
                try:
                    frame_presence_df = pd.read_csv(frame_presence_file)
                    for _, row in frame_presence_df.iterrows():
                        cell_id = int(row['cell_id'])
                        frame = int(row['frame'])
                        if cell_id not in cell_frame_presence:
                            cell_frame_presence[cell_id] = set()
                        cell_frame_presence[cell_id].add(frame)
                    logging.info(f"Loaded cell frame presence from {frame_presence_file}")
                except Exception as e:
                    logging.warning(f"Could not load cell frame presence from {frame_presence_file}: {e}")
            
            # If no frame presence file, create from trajectory
            if not cell_frame_presence:
                for cid in cell_ids:
                    frames = set(trj[trj['particle'] == cid]['frame'].unique())
                    cell_frame_presence[int(cid)] = frames
            
            main_app_state['cell_frame_presence'] = cell_frame_presence

            # Load cell coordinates
            cell_x = {}
            cell_y = {}
            coord_file = os.path.join(selected_path_from_dialog, 'cell_coordinates.csv')
            if os.path.exists(coord_file):
                try:
                    coord_df = pd.read_csv(coord_file)
                    for _, row in coord_df.iterrows():
                        cell_id = int(row['cell_id'])
                        frame = int(row['frame'])
                        cell_x[(cell_id, frame)] = float(row['x'])
                        cell_y[(cell_id, frame)] = float(row['y'])
                    logging.info(f"Loaded cell coordinates from {coord_file}")
                except Exception as e:
                    logging.warning(f"Could not load cell coordinates from {coord_file}: {e}")
            
            # If no coordinate file, create from trajectory
            if not cell_x:
                for _, row in trj.iterrows():
                    cell_id = int(row['particle'])
                    frame = int(row['frame'])
                    cell_x[(cell_id, frame)] = float(row['x'])
                    cell_y[(cell_id, frame)] = float(row['y'])
                logging.info("Extracted cell coordinates from trajectory")
            
            main_app_state['cell_x'] = cell_x
            main_app_state['cell_y'] = cell_y

            # Load merged masks if available (with progress updates to prevent hanging)
            merged_masks = None
            merged_masks_dir = os.path.join(selected_path_from_dialog, 'merged_masks')
            if os.path.exists(merged_masks_dir):
                try:
                    files = sorted([f for f in os.listdir(merged_masks_dir) if f.startswith('merged_mask_') and f.endswith('.tif')])
                    if files:
                        progress_dialog.setValue(87)
                        progress_dialog.setLabelText("Loading merged masks...")
                        QApplication.processEvents()
                        
                        # Read first file to get dimensions
                        first_mask = tifffile.imread(os.path.join(merged_masks_dir, files[0]))
                        height, width = first_mask.shape
                        num_frames = len(files)
                        
                        merged_masks = np.zeros((height, width, num_frames), dtype=np.uint16)
                        for i, filename in enumerate(files):
                            if i % 5 == 0:  # Update progress every 5 files
                                progress_dialog.setValue(87 + int(10 * i / len(files)))
                                progress_dialog.setLabelText(f"Loading merged masks... ({i+1}/{len(files)})")
                                QApplication.processEvents()
                            
                            mask = tifffile.imread(os.path.join(merged_masks_dir, filename))
                            merged_masks[:, :, i] = mask
                        logging.info(f"Loaded merged masks from {merged_masks_dir}")
                except Exception as e:
                    logging.warning(f"Could not load merged masks from {merged_masks_dir}: {e}")
            
            main_app_state['merged_masks'] = merged_masks

            # Try to load complete cell editor table if available
            cell_editor_file = os.path.join(selected_path_from_dialog, 'cell_editor_table.csv')
            if os.path.exists(cell_editor_file):
                try:
                    progress_dialog.setValue(90)
                    progress_dialog.setLabelText("Loading cell editor data...")
                    QApplication.processEvents()
                    
                    cell_editor_df = pd.read_csv(cell_editor_file)
                    logging.info(f"Loaded complete cell editor table from {cell_editor_file}")
                    
                    # Update main_app_state with cell editor data
                    for _, row in cell_editor_df.iterrows():
                        cell_id = int(row['cell_id'])
                        
                        # Update visibility
                        if 'visible' in row:
                            cell_visibility[cell_id] = bool(row['visible'])
                        
                        # Update state
                        if 'state' in row and pd.notna(row['state']):
                            track_states[cell_id] = str(row['state'])
                    
                    logging.info(f"Updated {len(cell_editor_df)} cells from cell editor table")
                    
                except Exception as e:
                    logging.warning(f"Could not load cell editor table from {cell_editor_file}: {e}")
            
            # Set up color information
            if id_masks is not None:
                # When loading previous results, preserve the original ID masks
                # and just set up the color mapping
                main_app_state['id_masks'] = id_masks
                main_app_state['background_id'] = 0
                
                # Extract unique cell IDs from the trajectory
                valid_particle_ids = pd.to_numeric(trj['particle'], errors='coerce').dropna().astype(int)
                cell_ids_unique = sorted(list(set(valid_particle_ids)))
                cell_ids_list = [0] + cell_ids_unique  # 0 is background
                
                # Generate colors for the cells
                num_actual_cells = len(cell_ids_unique)
                generated_colors = generate_distinct_colors(n_colors=num_actual_cells, n_intensity_levels=3) if num_actual_cells > 0 else []
                color_list = [(20, 20, 20)] + generated_colors  # Background color + cell colors
                
                main_app_state['cell_ids'] = cell_ids_list
                main_app_state['color_list'] = color_list
                main_app_state['cell_color_idx'] = {cell_id: idx for idx, cell_id in enumerate(cell_ids_list)}
            else:
                main_app_state['color_list'] = []
                main_app_state['cell_color_idx'] = {}
                main_app_state['id_masks'] = None
                main_app_state['cell_ids'] = []
                main_app_state['background_id'] = 0
            
            # Set up other required state variables
            main_app_state['next_available_daughter_id'] = max(cell_ids) + 1 if cell_ids else 1
            main_app_state['has_lineage'] = bool(cell_lineage or ancestry)
            main_app_state['current_frame_index'] = 0
            main_app_state['background_id'] = 0
            
            # Clear UI elements
            for item_dict_key, plot_widget_ref_key in [
                ('track_plots_per_cell', 'pi_raw_img'),
                ('cell_ids_raw_img', 'pi_raw_img'),
                ('cell_ids_mask', 'pi_mask')
            ]:
                item_dict = main_app_state.get(item_dict_key, {})
                plot_widget_ref = ui_elements.get(plot_widget_ref_key)
                if plot_widget_ref:
                    for key_to_remove in list(item_dict.keys()):
                        if item_dict[key_to_remove].scene():
                            plot_widget_ref.removeItem(item_dict[key_to_remove])
                    item_dict.clear()
            
            # Initialize overlay elements for raw images and masks
            progress_dialog.setValue(92)
            progress_dialog.setLabelText("Initializing overlay elements...")
            QApplication.processEvents()
            
            try:
                # Initialize overlay elements similar to tracking process
                if 'particle' in trj.columns:
                    unique_particles_for_plot_items = trj['particle'].unique()
                else:
                    unique_particles_for_plot_items = []
                    
                for cell_id_val_pi in unique_particles_for_plot_items:
                    cell_id_int_pi = int(cell_id_val_pi)
                    if cell_id_int_pi == main_app_state.get('background_id', 0): 
                        continue

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
                
                logging.info(f"Initialized overlay elements for {len(unique_particles_for_plot_items)} cells")
                
            except Exception as e:
                logging.error(f"Error initializing overlay elements: {e}")
            
            # Update UI with timeout protection
            try:
                progress_dialog.setValue(95)
                progress_dialog.setLabelText("Updating UI...")
                QApplication.processEvents()
                
                ui_callbacks.handle_populate_cell_table(main_app_state, ui_elements, ui_callbacks)
                
                progress_dialog.setValue(97)
                progress_dialog.setLabelText("Finalizing...")
                QApplication.processEvents()
                
                # Verify that the cell table was populated
                table_widget = ui_elements.get('table_cell_selection')
                if table_widget:
                    row_count = table_widget.rowCount()
                    logging.info(f"Cell table populated with {row_count} rows")
                    if row_count == 0:
                        logging.warning("Cell table is empty after population - this may indicate an issue")
                else:
                    logging.warning("Cell table widget not found in UI elements")
                
            except Exception as e:
                logging.error(f"Error updating UI: {e}")
                # Continue anyway to avoid hanging
            
            # Update image views with loaded data
            v_raw_img = ui_elements.get('v_raw_img')
            v_mask = ui_elements.get('v_mask')
            
            if v_raw_img and main_app_state.get('raw_imgs') is not None:
                v_raw_img.setImage(main_app_state['raw_imgs'], axes={'x': 1, 'y': 0, 't': 2}, autoLevels=True, autoRange=False)
                v_raw_img.setCurrentIndex(main_app_state['current_frame_index'])
                
                # Set proper levels based on actual image data
                img_data = main_app_state['raw_imgs']
                if img_data is not None:
                    img_min, img_max = np.min(img_data), np.max(img_data)
                    if img_max > img_min:  # Avoid division by zero
                        v_raw_img.setLevels(min=img_min, max=img_max)
                        logging.info(f"Set raw image levels to {img_min}-{img_max}")
                    else:
                        logging.warning(f"Raw image has no contrast: min={img_min}, max={img_max}")
                
                # Auto-range to fit the image content
                v_raw_img.autoRange()
                logging.info("Applied auto-range to raw image view")
                
                pi_raw_img = ui_elements.get('pi_raw_img')
                if pi_raw_img:
                    pi_raw_img.setTitle(f"Raw Image - Frame {main_app_state['current_frame_index']}")
            
            # Set up mask view - try ID masks first, then raw masks
            if v_mask:
                if main_app_state.get('id_masks') is not None:
                    # Use ID masks for visualization
                    v_mask.setImage(main_app_state['id_masks'], axes={'x': 1, 'y': 0, 't': 2}, autoLevels=False, autoRange=False)
                    
                    # Set up lookup table for ID masks
                    if main_app_state.get('color_list') and main_app_state.get('cell_color_idx'):
                        all_drawable_ids = [int(i) for i in main_app_state['cell_color_idx'].keys() if isinstance(i, (int, float))]
                        lut_size = max(all_drawable_ids) + 1 if all_drawable_ids else 1
                        actual_lut = np.zeros((lut_size, 3), dtype=np.uint8)
                        
                        bg_col_idx = main_app_state['cell_color_idx'].get(main_app_state['background_id'])
                        bg_color = main_app_state['color_list'][bg_col_idx] if bg_col_idx is not None and 0 <= bg_col_idx < len(main_app_state['color_list']) else (20, 20, 20)
                        
                        if main_app_state['background_id'] >= 0 and main_app_state['background_id'] < lut_size:
                            actual_lut[main_app_state['background_id']] = bg_color
                        
                        for cid, c_idx in main_app_state['cell_color_idx'].items():
                            if isinstance(cid, (int, float)):
                                cid_int = int(cid)
                                if cid_int >= 0 and cid_int < lut_size and c_idx < len(main_app_state['color_list']):
                                    actual_lut[cid_int] = main_app_state['color_list'][c_idx]
                        
                        if v_mask.imageItem:
                            v_mask.imageItem.setLookupTable(actual_lut)
                        v_mask.setLevels(min=0, max=lut_size - 1 if lut_size > 0 else 0)
                    
                    v_mask.setCurrentIndex(main_app_state['current_frame_index'])
                    pi_mask = ui_elements.get('pi_mask')
                    if pi_mask:
                        pi_mask.setTitle(f"Mask - Frame {main_app_state['current_frame_index']} (ID Mask)")
                        
                elif main_app_state.get('raw_masks') is not None:
                    # Fall back to raw masks
                    v_mask.setImage(main_app_state['raw_masks'], axes={'x': 1, 'y': 0, 't': 2}, autoLevels=True, autoRange=False)
                    v_mask.setCurrentIndex(main_app_state['current_frame_index'])
                    
                    # Auto-range to fit the mask content
                    v_mask.autoRange()
                    logging.info("Applied auto-range to mask view")
                    
                    pi_mask = ui_elements.get('pi_mask')
                    if pi_mask:
                        pi_mask.setTitle(f"Mask - Frame {main_app_state['current_frame_index']} (Raw Mask)")
            
            # Clear all caches when new data is loaded
            if 'cell_ui_callbacks' in sys.modules:
                cell_ui_callbacks._clear_all_caches()
            
            # Force update of overlay display
            ui_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)
            
            # Update lineage tree
            lineage_tree_widget = ui_elements.get('lineage_tree_widget')
            if lineage_tree_widget:
                try:
                    lineage_tree_widget.set_data(
                        trj, 
                        main_app_state.get('color_list', []),
                        main_app_state.get('cell_color_idx', {}),
                        main_app_state.get('cell_visibility', {}),
                        main_app_state.get('track_states', {}),
                        ancestry,
                        main_app_state.get('params')
                    )
                    lineage_tree_widget.draw_all_lineage_trees()
                    logging.info("Lineage tree updated successfully")
                except Exception as e:
                    logging.error(f"Error updating lineage tree: {e}")
            
            # Single update of overlay display (debounced)
            ui_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)
            
            # Enable appropriate buttons
            buttons_to_enable = [
                'save_results', 'calculate_stats', 'select_all_visible',
                'select_none_visible', 'select_complete_tracks', 'select_mitosis_tracks',
                'select_fusion_tracks', 'select_singular_tracks', 'relink_visible_tracks'
            ]
            for btn_key in buttons_to_enable:
                btn_widget = ui_elements.get('button_widgets_map', {}).get(btn_key)
                if btn_widget:
                    btn_widget.setEnabled(True)
            
            # Disable tracking button since we already have results
            b_cell_tracking = ui_elements.get('button_widgets_map', {}).get('run_tracking')
            if b_cell_tracking:
                b_cell_tracking.setEnabled(False)
            
            # Update window title
            win = ui_elements.get('win')
            if win:
                win.setWindowTitle(f'Cell Tracking - Loaded Results ({os.path.basename(selected_path_from_dialog)})')
            
            progress_dialog.setValue(100)
            progress_dialog.close()
            
            # Count total lineage relationships
            total_lineage_relationships = len(cell_lineage)
            for parent_id, daughters in cell_lineage.items():
                total_lineage_relationships += len(daughters)
            
            # Check if saved PNG images were used
            raw_images_dir = os.path.join(selected_path_from_dialog, 'raw_images')
            image_source = "saved PNG images" if os.path.exists(raw_images_dir) else "original data folder"
            
            # Log detailed information about what was loaded
            logging.info("=== LOAD PREVIOUS RESULTS SUMMARY ===")
            logging.info(f"Trajectory data: {len(trj)} rows, {len(trj.columns)} columns")
            logging.info(f"Raw images loaded: {main_app_state.get('raw_imgs') is not None}")
            if main_app_state.get('raw_imgs') is not None:
                logging.info(f"Raw images shape: {main_app_state['raw_imgs'].shape}")
            logging.info(f"ID masks loaded: {main_app_state.get('id_masks') is not None}")
            if main_app_state.get('id_masks') is not None:
                logging.info(f"ID masks shape: {main_app_state['id_masks'].shape}")
            logging.info(f"Cell lineage relationships: {len(cell_lineage)}")
            logging.info(f"Cell visibility entries: {len(main_app_state.get('cell_visibility', {}))}")
            logging.info(f"Track states entries: {len(main_app_state.get('track_states', {}))}")
            logging.info(f"Cell coordinates entries: {len(main_app_state.get('cell_x', {}))}")
            logging.info(f"Image source: {image_source}")
            logging.info("=== END LOAD SUMMARY ===")
            
            QMessageBox.information(ui_elements.get('win'), "Load Successful", 
                                  f"Successfully loaded previous results from:\n{selected_path_from_dialog}\n\n"
                                  f"Loaded {len(cell_ids)} tracks with {total_lineage_relationships} lineage relationships.\n"
                                  f"Raw images loaded from: {image_source}")
            
            logging.info(f"Successfully loaded previous results from {selected_path_from_dialog}")
            
        except Exception as e:
            progress_dialog.close()
            QMessageBox.critical(ui_elements.get('win'), "Load Error", 
                               f"Error loading previous results:\n{str(e)}\n\n{traceback.format_exc()}")
            logging.error(f"Error loading previous results: {e}")
            return
        finally:
            # Ensure progress dialog is always closed
            if 'progress_dialog' in locals():
                progress_dialog.close()


def load_data_from_folder(main_app_state, ui_elements, ui_callbacks, ui_actions):
    """
    Handles the logic for opening a folder, reading image sequences, and initializing the application state.
    """
    from cell_undoredo import clear_undo_redo_stacks

    clear_undo_redo_stacks(ui_elements)

    dialog_title = 'Select Raw Image Folder (Standard Mode) or Base Dataset Parent Folder (Advanced Mode)'
    start_directory_for_dialog = main_app_state.get('path_in', os.path.expanduser("~"))
    if main_app_state.get('path_in') and os.path.isdir(main_app_state['path_in']):
        start_directory_for_dialog = main_app_state['path_in']

    raw_folder_param_text = main_app_state['params']['Raw Image Folder Name'][0]

    if raw_folder_param_text and os.path.isabs(raw_folder_param_text):
        if os.path.isdir(raw_folder_param_text):
            if main_app_state['params']['Enable Advanced File Structure'][0]:
                potential_parent = os.path.dirname(raw_folder_param_text)
                start_directory_for_dialog = potential_parent if potential_parent and os.path.isdir(
                    potential_parent) else raw_folder_param_text
            else:
                start_directory_for_dialog = raw_folder_param_text
        else:
            potential_parent_of_param = os.path.dirname(raw_folder_param_text)
            if potential_parent_of_param and os.path.isdir(potential_parent_of_param):
                start_directory_for_dialog = potential_parent_of_param

    selected_path_from_dialog = QFileDialog.getExistingDirectory(
        ui_elements.get('win'),
        dialog_title,
        start_directory_for_dialog
    )
    if not selected_path_from_dialog:
        logging.info('Canceled folder selection')
        return

    main_app_state['path_in'] = selected_path_from_dialog

    # Show progress dialog for data loading
    progress_dialog = QProgressDialog("Loading data...", "Cancel", 0, 100, ui_elements.get('win'))
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setAutoClose(False)
    progress_dialog.show()
    
    try:
        with pg.BusyCursor():
            raw_image_folder_actual, generic_mask_folder_actual = "", ""
            class_specific_mask_folders_paths = {}
            main_app_state['loaded_class_specific_masks'].clear()
            main_app_state['track_states'].clear()
            main_app_state['last_saved_res_path'] = None  # Reset last saved RES path on new data load
            use_advanced_structure = main_app_state['params']['Enable Advanced File Structure'][0]

        logging.info(f"Loading data. Advanced structure: {use_advanced_structure}")
        logging.info(f"Base path selected: {main_app_state['path_in']}")
        
        progress_dialog.setValue(10)
        progress_dialog.setLabelText("Determining folder structure...")
        QApplication.processEvents()

        if use_advanced_structure:
            base_dataset_parent_dir = main_app_state['path_in']
            raw_folder_name_param = main_app_state['params']['Raw Image Folder Name'][0]
            generic_mask_folder_name_param = main_app_state['params']['Generic Mask Folder Name'][0]
            class_mask_name_pattern_param = main_app_state['params']['Class-Specific Mask Folder Name Pattern'][0]
            class_definitions_str_param = main_app_state['params']['Class Definitions (JSON list)'][0]

            raw_image_folder_actual = os.path.join(base_dataset_parent_dir, raw_folder_name_param)
            generic_mask_folder_actual = os.path.join(base_dataset_parent_dir, generic_mask_folder_name_param)

            try:
                class_names = json.loads(class_definitions_str_param)
                if not isinstance(class_names, list) or not all(isinstance(cn, str) for cn in class_names):
                    raise ValueError("Class Definitions must be a JSON list of strings.")
                for class_name in class_names:
                    actual_class_mask_folder_name = class_mask_name_pattern_param.replace('{class_name}', class_name)
                    class_specific_mask_folders_paths[class_name] = os.path.join(base_dataset_parent_dir,
                                                                                 actual_class_mask_folder_name)
            except (json.JSONDecodeError, ValueError) as e_json:
                QMessageBox.critical(ui_elements.get('win'), "Config Error", f"Error in Class Definitions: {e_json}")
                return
        else:  # Standard structure
            raw_image_folder_actual = main_app_state['path_in']
            img_folder_parent_dir = os.path.dirname(raw_image_folder_actual)
            img_folder_basename = os.path.basename(raw_image_folder_actual)
            mask_folder_suffix_param = main_app_state['params']['Mask folder suffix'][0]
            generic_mask_folder_name_std = img_folder_basename + mask_folder_suffix_param
            generic_mask_folder_actual = os.path.join(img_folder_parent_dir, generic_mask_folder_name_std)

        logging.info(f"Actual raw image folder: {raw_image_folder_actual}")
        logging.info(f"Actual generic mask folder: {generic_mask_folder_actual}")

        progress_dialog.setValue(20)
        progress_dialog.setLabelText("Checking folder structure...")
        QApplication.processEvents()

        if not os.path.isdir(raw_image_folder_actual):
            QMessageBox.critical(ui_elements.get('win'), "Error",
                                 f"Raw image folder not found: {raw_image_folder_actual}")
            progress_dialog.close()
            return
        if not os.path.isdir(generic_mask_folder_actual):
            QMessageBox.warning(ui_elements.get('win'), "Warning",  # Changed to warning, can proceed with only raw
                                f"Generic mask folder not found: {generic_mask_folder_actual}. Will load raw images only.")
            new_masks_generic = None  # Indicate no masks loaded
        else:  # Masks folder exists, try to load
            try:
                progress_dialog.setValue(30)
                progress_dialog.setLabelText("Loading generic masks...")
                QApplication.processEvents()
                logging.info("Loading generic masks...")
                new_masks_generic = read_img_sequence(generic_mask_folder_actual,
                                                      main_app_state['params']['Mask extension'][0])
            except Exception as e_mask_read:
                QMessageBox.warning(ui_elements.get('win'), "Mask Read Error",
                                    f"Could not read generic masks from {generic_mask_folder_actual}: {e_mask_read}. Will load raw images only.")
                new_masks_generic = None

        try:  # Always try to load raw images
            progress_dialog.setValue(50)
            progress_dialog.setLabelText("Loading raw images...")
            QApplication.processEvents()
            logging.info("Loading raw images...")
            new_imgs = read_img_sequence(raw_image_folder_actual, main_app_state['params']['Raw image extension'][0])
        except Exception as e_img_read:
            QMessageBox.critical(ui_elements.get('win'), "Image Read Error",
                                 f"Could not read raw images from {raw_image_folder_actual}: {e_img_read}\n{traceback.format_exc()}")
            progress_dialog.close()
            return

        if new_imgs.ndim < 3 or new_imgs.shape[2] == 0:
            QMessageBox.critical(ui_elements.get('win'), 'Data Reading Failed',
                                 "Raw image data invalid or empty.")
            progress_dialog.close()
            return

        if new_masks_generic is not None:  # If masks were loaded, check compatibility
            if new_masks_generic.ndim < 3 or new_imgs.shape[:2] != new_masks_generic.shape[:2] or new_imgs.shape[2] != \
                    new_masks_generic.shape[2]:
                QMessageBox.critical(ui_elements.get('win'), 'Data Reading Failed',
                                     "Image/generic mask data have mismatched dimensions (height, width, or number of frames).")
                progress_dialog.close()
                return

        if use_advanced_structure:
            progress_dialog.setValue(70)
            progress_dialog.setLabelText("Loading class-specific masks...")
            QApplication.processEvents()
            logging.info("Loading class-specific masks...")
            for class_name_key, folder_path_val in class_specific_mask_folders_paths.items():
                if os.path.isdir(folder_path_val):
                    try:
                        logging.info(f"Loading masks for class '{class_name_key}'...")
                        loaded_class_mask = read_img_sequence(folder_path_val,
                                                              main_app_state['params']['Mask extension'][0])
                        # Check compatibility with raw images (or generic masks if they exist)
                        ref_shape_for_class_mask = new_imgs.shape if new_masks_generic is None else new_masks_generic.shape
                        if loaded_class_mask.shape[:3] != ref_shape_for_class_mask[:3]:
                            QMessageBox.warning(ui_elements.get('win'), "Shape Mismatch",
                                                f"Masks for class '{class_name_key}' shape mismatch with reference images/masks.")
                        else:
                            main_app_state['loaded_class_specific_masks'][class_name_key] = loaded_class_mask
                            logging.info(f"Successfully loaded masks for class '{class_name_key}'")
                    except Exception as e_class_mask:
                        QMessageBox.warning(ui_elements.get('win'), "Read Error",
                                            f"Could not read masks for class {class_name_key}: {e_class_mask}. Skipping.")
                else:  # Folder not found
                    QMessageBox.warning(ui_elements.get('win'), "Warning",
                                        f"Class mask folder for '{class_name_key}' not found: {folder_path_val}. Skipping.")

        # Reset application state variables
        main_app_state['trj'] = pd.DataFrame()
        main_app_state['merged_masks'] = None
        main_app_state['id_masks'] = None
        main_app_state['id_masks_initial'] = None
        main_app_state['cell_ids'] = []
        main_app_state['ancestry'].clear()
        main_app_state['cell_lineage'].clear()
        main_app_state['next_available_daughter_id'] = 1
        main_app_state['cell_y'].clear()
        main_app_state['cell_x'].clear()

        pi_raw_img = ui_elements.get('pi_raw_img')
        pi_mask = ui_elements.get('pi_mask')

        for item_dict_key, plot_widget_ref_key in [
            ('track_plots_per_cell', 'pi_raw_img'),
            ('cell_ids_raw_img', 'pi_raw_img'),
            ('cell_ids_mask', 'pi_mask')
        ]:
            item_dict = main_app_state.get(item_dict_key, {})
            plot_widget_ref = ui_elements.get(plot_widget_ref_key)
            if plot_widget_ref:
                for key_to_remove in list(item_dict.keys()):
                    if item_dict[key_to_remove].scene():
                        plot_widget_ref.removeItem(item_dict[key_to_remove])
                item_dict.clear()

        main_app_state['wide_track_cell_id'] = None
        main_app_state['color_list'] = None
        main_app_state['track_data_per_frame'].clear()
        main_app_state['cell_color_idx'].clear()
        main_app_state['cell_visibility'].clear()
        main_app_state['cell_frame_presence'].clear()

        table_cell_selection = ui_elements.get('table_cell_selection')
        if table_cell_selection:
            table_cell_selection.setRowCount(0)

        v_raw_img = ui_elements.get('v_raw_img')
        v_mask = ui_elements.get('v_mask')
        if main_app_state.get('v_raw_img_original_state') and v_raw_img:
            v_raw_img.ui.histogram.gradient.restoreState(main_app_state['v_raw_img_original_state'])
        if main_app_state.get('v_mask_original_state') and v_mask:
            v_mask.ui.histogram.gradient.restoreState(main_app_state['v_mask_original_state'])

        lineage_tree_widget = ui_elements.get('lineage_tree_widget')
        if lineage_tree_widget:
            lineage_tree_widget.set_data(pd.DataFrame(), [], {}, {}, {}, {}, main_app_state.get('params'))

        main_app_state['raw_imgs'] = new_imgs
        main_app_state['raw_masks'] = new_masks_generic if new_masks_generic is not None else np.zeros_like(new_imgs,
                                                                                                            dtype=np.uint16)  # Placeholder if no masks
        main_app_state['current_frame_index'] = 0
        main_app_state['background_id'] = 0

        if v_raw_img:
            # Handle different image dimensions - use axes parameter instead of transposing
            raw_imgs = main_app_state['raw_imgs']
            if raw_imgs.ndim == 4:  # Color image: (height, width, channels, frames)
                # For color images, convert to grayscale by taking the mean across channels
                gray_imgs = np.mean(raw_imgs, axis=2)  # Average across channels
                v_raw_img.setImage(gray_imgs, axes={'x': 1, 'y': 0, 't': 2})
            else:  # Grayscale image: (height, width, frames)
                v_raw_img.setImage(raw_imgs, axes={'x': 1, 'y': 0, 't': 2}, autoLevels=True, autoRange=False)
            v_raw_img.setCurrentIndex(main_app_state['current_frame_index'])
            
            # Auto-range to fit the image content
            v_raw_img.autoRange()
            logging.info("Applied auto-range to raw image view during data loading")
            
        if pi_raw_img:
            pi_raw_img.setTitle(f"Frame: {main_app_state['current_frame_index']}")

        if v_mask:
            # Handle different mask dimensions - use axes parameter instead of transposing
            raw_masks = main_app_state['raw_masks']
            if raw_masks.ndim == 4:  # Color mask: (height, width, channels, frames)
                # For color masks, convert to grayscale by taking the mean across channels
                gray_masks = np.mean(raw_masks, axis=2)  # Average across channels
                v_mask.setImage(gray_masks, axes={'x': 1, 'y': 0, 't': 2}, autoLevels=True, autoRange=False)
            else:  # Grayscale mask: (height, width, frames)
                v_mask.setImage(raw_masks, axes={'x': 1, 'y': 0, 't': 2}, autoLevels=True, autoRange=False)
            v_mask.setCurrentIndex(main_app_state['current_frame_index'])
            
            # Auto-range to fit the mask content
            v_mask.autoRange()
            logging.info("Applied auto-range to mask view during data loading")
            
        if pi_mask:
            pi_mask.setTitle(f"Frame: {main_app_state['current_frame_index']} (Raw Mask - Generic)")

        # --- V2 UPDATE: Enable tracking and optimizer buttons after data load ---
        b_cell_tracking = ui_elements.get('button_widgets_map', {}).get('run_tracking')
        if b_cell_tracking: b_cell_tracking.setEnabled(True)

        b_optimize_ilp = ui_elements.get('button_widgets_map', {}).get('optimize_ilp')
        if b_optimize_ilp:
            # Enable the optimizer if mask data is available, as it's required.
            optimizer_enabled = main_app_state.get('raw_masks') is not None
            b_optimize_ilp.setEnabled(optimizer_enabled)
            logging.info(f"Optimizer button enabled: {optimizer_enabled}")
        # --- END V2 UPDATE ---


        buttons_to_disable_keys = [
            'save_results', 'calculate_stats', 'select_all_visible',
            'select_none_visible', 'select_complete_tracks', 'merge_selected_track_with_parent',
            'evaluate_tracking_ctc'  # Disable evaluate button on new data load
        ]
        for btn_key in buttons_to_disable_keys:
            btn_widget = ui_elements.get('button_widgets_map', {}).get(btn_key)
            if btn_widget:
                btn_widget.setEnabled(False)

        win = ui_elements.get('win')
        if win:
            win.setWindowTitle(f'Cell Tracking ({os.path.basename(raw_image_folder_actual)})')

        progress_dialog.setValue(90)
        progress_dialog.setLabelText("Finalizing data loading...")
        QApplication.processEvents()

        logging.info("Data loading complete.")
        
        # Close progress dialog
        progress_dialog.setValue(100)
        progress_dialog.close()
        
        # Ensure log display scrolls to bottom after data loading
        log_display_widget = ui_elements.get('log_display_widget')
        if log_display_widget:
            # Use a timer to ensure this happens after all UI updates are complete
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, log_display_widget.force_scroll_to_bottom)
            
    except Exception as e:
        # Ensure progress dialog is closed on any error
        if 'progress_dialog' in locals():
            progress_dialog.close()
        QMessageBox.critical(ui_elements.get('win'), "Data Loading Error",
                           f"Error loading data:\n{str(e)}\n\n{traceback.format_exc()}")
        logging.error(f"Error loading data: {e}")
        return
    finally:
        # Ensure progress dialog is always closed
        if 'progress_dialog' in locals():
            progress_dialog.close()


def perform_batch_tracking(main_app_state, ui_elements):
    """
    Handles batch processing of multiple datasets.
    """
    from cell_tracking_orchestrator import perform_cell_state_classification, initiate_cell_tracking

    path_in = main_app_state.get('path_in')
    params = main_app_state.get('params')
    win = ui_elements.get('win')

    parent_dir = QFileDialog.getExistingDirectory(
        win,
        'Select Parent Folder of Datasets',
        os.path.dirname(path_in) if path_in and os.path.exists(os.path.dirname(path_in)) else os.path.expanduser("~")
    )
    if not parent_dir:
        logging.info("Batch tracking cancelled by user.")
        return

    dataset_paths = []
    use_adv_batch = params['Enable Advanced File Structure'][0]

    for item_name in os.listdir(parent_dir):
        potential_path = os.path.join(parent_dir, item_name)
        if os.path.isdir(potential_path):
            if use_adv_batch:
                raw_check = os.path.join(potential_path, params['Raw Image Folder Name'][0])
                mask_check = os.path.join(potential_path, params['Generic Mask Folder Name'][0])
                if os.path.isdir(raw_check) and os.path.isdir(mask_check):
                    dataset_paths.append(potential_path)
            else:
                mask_suffix = params['Mask folder suffix'][0]
                if not item_name.endswith(mask_suffix):
                    mask_folder_to_check = os.path.join(parent_dir,
                                                        item_name + mask_suffix)  # Construct full path to mask folder
                    if os.path.isdir(mask_folder_to_check):  # Check if this mask folder exists
                        dataset_paths.append(potential_path)

    if not dataset_paths:
        QMessageBox.warning(win, 'No Datasets', "No valid dataset structures found in the selected parent folder.")
        return

    processed_count, error_count = 0, 0
    start_time_batch = time.time()
    progress_dialog = QProgressDialog("Processing datasets...", "Cancel", 0, len(dataset_paths), win)
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.show()

    original_loaded_masks_global = main_app_state['loaded_class_specific_masks'].copy()
    original_track_states_global = main_app_state['track_states'].copy()
    original_ancestry_global = main_app_state['ancestry'].copy()
    original_cell_lineage_global = main_app_state['cell_lineage'].copy()

    for idx, current_base_or_raw_path in enumerate(dataset_paths):
        progress_dialog.setValue(idx)
        progress_dialog.setLabelText(f"Processing {os.path.basename(current_base_or_raw_path)}...")
        QApplication.processEvents()
        if progress_dialog.wasCanceled():
            logging.info("Batch tracking cancelled by user during processing.")
            break

        current_dataset_log_capture = io.StringIO()
        batch_item_logger = logging.getLogger(f"batch_item_{idx}")
        for handler in batch_item_logger.handlers[:]:
            batch_item_logger.removeHandler(handler)

        batch_item_log_handler = logging.StreamHandler(current_dataset_log_capture)
        batch_item_log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s'))
        batch_item_log_handler.setLevel(logging.INFO)
        batch_item_logger.addHandler(batch_item_log_handler)
        batch_item_logger.setLevel(logging.INFO)
        batch_item_logger.propagate = False

        batch_tracking_successful_item = False
        try:
            actual_raw_folder_batch, actual_generic_mask_folder_batch = "", ""
            path_for_saving_batch, path_for_track_cells_param_batch = "", ""
            batch_dataset_specific_loaded_masks = {}

            if use_adv_batch:
                actual_raw_folder_batch = os.path.join(current_base_or_raw_path, params['Raw Image Folder Name'][0])
                actual_generic_mask_folder_batch = os.path.join(current_base_or_raw_path,
                                                                params['Generic Mask Folder Name'][0])
                path_for_saving_batch = actual_raw_folder_batch
                path_for_track_cells_param_batch = actual_raw_folder_batch
                try:
                    class_names_batch = json.loads(params['Class Definitions (JSON list)'][0])
                    if isinstance(class_names_batch, list):
                        for cn_b in class_names_batch:
                            if isinstance(cn_b, str):
                                class_mask_folder_name_b = params['Class-Specific Mask Folder Name Pattern'][0].replace(
                                    '{class_name}', cn_b)
                                class_mask_path_b = os.path.join(current_base_or_raw_path, class_mask_folder_name_b)
                                if os.path.isdir(class_mask_path_b):
                                    batch_dataset_specific_loaded_masks[cn_b] = read_img_sequence(class_mask_path_b,
                                                                                                  params[
                                                                                                      'Mask extension'][
                                                                                                      0])
                except Exception as e_cls_load_b:
                    batch_item_logger.warning(
                        f"Batch: Error loading class masks for {actual_raw_folder_batch}: {e_cls_load_b}")
            else:  # Standard mode
                actual_raw_folder_batch = current_base_or_raw_path
                actual_generic_mask_folder_batch = os.path.join(
                    os.path.dirname(actual_raw_folder_batch),
                    os.path.basename(actual_raw_folder_batch) + params['Mask folder suffix'][0]
                )
                path_for_saving_batch = actual_raw_folder_batch
                path_for_track_cells_param_batch = actual_raw_folder_batch

            if not (os.path.isdir(actual_raw_folder_batch) and os.path.isdir(actual_generic_mask_folder_batch)):
                batch_item_logger.error(
                    f"Batch: Invalid folder structure for {current_base_or_raw_path}. Raw: '{actual_raw_folder_batch}', Mask: '{actual_generic_mask_folder_batch}'. Skipping.")
                error_count += 1
                continue

            batch_raw_imgs = read_img_sequence(actual_raw_folder_batch, params['Raw image extension'][0])
            batch_generic_masks = read_img_sequence(actual_generic_mask_folder_batch, params['Mask extension'][0])

            if batch_raw_imgs.ndim < 3 or batch_generic_masks.ndim < 3 or \
                    batch_raw_imgs.shape[2] != batch_generic_masks.shape[2] or batch_raw_imgs.shape[2] == 0:
                batch_item_logger.error(f"Batch: Invalid image/mask data for {current_base_or_raw_path}. Skipping.")
                error_count += 1
                continue

            # FIXED: Updated to support all tracking modes including ILP and Trackastra
            batch_tracking_mode_val = params['Tracking Mode'][0]
            valid_modes_list_batch = ["Forward", "Backward", "Basic", "ILP", "Trackastra"]
            tracking_mode_to_use_batch = "Backward"
            if isinstance(batch_tracking_mode_val, str) and batch_tracking_mode_val in valid_modes_list_batch:
                tracking_mode_to_use_batch = batch_tracking_mode_val
            else:
                batch_item_logger.warning(
                    f"Batch: Invalid tracking mode '{batch_tracking_mode_val}' for {current_base_or_raw_path}. Defaulting to 'Backward'.")

            # FIXED: Create a temporary app state for batch processing
            temp_batch_app_state = {
                'path_in': path_for_track_cells_param_batch,
                'raw_masks': batch_generic_masks.copy(),
                'raw_imgs': batch_raw_imgs,
                'params': params,
                'loaded_class_specific_masks': batch_dataset_specific_loaded_masks,
                'cell_visibility': {},
                'cell_frame_presence': {},
                'track_data_per_frame': {},
                'cell_y': {},
                'cell_x': {},
                'cell_lineage': {},
                'ancestry': {},
                'track_states': {},
                'next_available_daughter_id': 1,
                'last_saved_res_path': None,
                'merged_masks': batch_generic_masks.copy(),
                'captured_tracking_log': ""
            }

            # FIXED: Use the tracking orchestrator instead of direct trackpy calls
            # This ensures ILP and Trackastra modes work properly
            try:
                # Temporarily update the tracking mode in params for this batch item
                original_tracking_mode = params['Tracking Mode'][0]
                params['Tracking Mode'] = (tracking_mode_to_use_batch, 'str')
                
                # Use the orchestrator which handles all tracking modes
                initiate_cell_tracking(temp_batch_app_state, ui_elements=None, ui_callbacks=None, ui_actions=None)
                
                # Restore original tracking mode
                params['Tracking Mode'] = (original_tracking_mode, 'str')
                
                # Extract results from the temporary app state
                batch_trj = temp_batch_app_state.get('trj', pd.DataFrame())
                batch_col_tuple = temp_batch_app_state.get('col_tuple', {})
                batch_col_weights = temp_batch_app_state.get('col_weights', {})
                batch_cell_lineage = temp_batch_app_state.get('cell_lineage', {})
                batch_id_masks = temp_batch_app_state.get('id_masks')
                batch_cell_ids = temp_batch_app_state.get('cell_ids', [])
                batch_color_list = temp_batch_app_state.get('color_list', [])
                batch_background_id = temp_batch_app_state.get('background_id', 0)
                batch_cell_visibility = temp_batch_app_state.get('cell_visibility', {})
                
                batch_tracking_successful_item = True
                
            except Exception as e_orchestrator:
                batch_item_logger.error(f"Batch: Error in tracking orchestrator for {current_base_or_raw_path}: {e_orchestrator}")
                # Fallback to original trackpy method for backward compatibility
                batch_item_logger.info("Batch: Falling back to trackpy method...")
                from cell_tracking import track_cells
                batch_trj, batch_col_tuple, batch_col_weights, batch_cell_lineage = track_cells(
                    path_for_track_cells_param_batch,
                    batch_generic_masks.copy(),
                    tracking_mode=tracking_mode_to_use_batch if tracking_mode_to_use_batch in ["Forward", "Backward", "Basic"] else "Backward",
                    min_cell_id=1,
                    search_range=params['Trackpy search range'][0],
                    memory=params['Trackpy memory'][0],
                    neighbor_strategy=params['Trackpy neighbor strategy'][0],
                    mitosis_max_dist_factor=params['Mitosis Max Distance Factor'][0],
                    mitosis_area_sum_min_factor=params['Mitosis Area Sum Min Factor'][0],
                    mitosis_area_sum_max_factor=params['Mitosis Area Sum Max Factor'][0],
                    mitosis_daughter_area_similarity=params['Mitosis Daughter Area Similarity'][0]
                )
                
                # Prepare masks and visibility for fallback
                batch_id_masks, batch_cell_ids, batch_color_list, batch_background_id = prepare_mask_colors(
                    batch_generic_masks.copy(), batch_trj
                )
                batch_cell_visibility = {
                    cid: True for cid in pd.to_numeric(batch_trj['particle'], errors='coerce').dropna().astype(int).unique()
                    if cid != batch_background_id
                }
                batch_tracking_successful_item = True

            if batch_trj is None or batch_trj.empty:
                batch_item_logger.warning(
                    f"Empty trajectory for {os.path.basename(actual_raw_folder_batch)}. Skipping save for this item.")
                processed_count += 1
                continue

            # Prepare cell color index for visualization
            batch_cell_color_idx = {cid: cidx for cidx, cid in enumerate(batch_cell_ids)}

            min_duration_batch_param = params.get('Min Tracklet Duration', (1,))[0]
            if min_duration_batch_param > 1 and not batch_trj.empty and 'particle' in batch_trj.columns:
                track_lengths_batch_df = batch_trj.groupby('particle')['frame'].nunique()
                short_track_ids_batch = track_lengths_batch_df[track_lengths_batch_df < min_duration_batch_param].index
                short_track_ids_batch_int = pd.to_numeric(short_track_ids_batch, errors='coerce').dropna().astype(int)
                for s_id_b in short_track_ids_batch_int:
                    if s_id_b in batch_cell_visibility:
                        batch_cell_visibility[s_id_b] = False

            batch_item_track_states = {}
            if batch_dataset_specific_loaded_masks and not batch_trj.empty and batch_id_masks is not None:
                temp_batch_app_state = {
                    'trj': batch_trj, 'id_masks': batch_id_masks,
                    'loaded_class_specific_masks': batch_dataset_specific_loaded_masks,
                    'params': main_app_state['params'],
                    'track_states': batch_item_track_states,
                    'cell_visibility': batch_cell_visibility
                }
                perform_cell_state_classification(temp_batch_app_state, ui_elements=None)

                if batch_item_track_states and 'state' not in batch_trj.columns:
                    batch_trj['state'] = batch_trj['particle'].map(batch_item_track_states).fillna("N/A")

            batch_item_counts = {'start': 'N/A', 'end': 'N/A'}
            num_frames_batch = batch_id_masks.shape[2] if batch_id_masks is not None else \
                (int(batch_trj['frame'].max()) + 1 if not batch_trj.empty else 0)
            if num_frames_batch > 0 and not batch_trj.empty and 'particle' in batch_trj.columns and batch_cell_visibility:
                visible_pids_batch = {pid_b for pid_b, vis_b in batch_cell_visibility.items() if vis_b}
                trj_for_count_batch = batch_trj.copy()
                trj_for_count_batch['particle'] = pd.to_numeric(trj_for_count_batch['particle'],
                                                                errors='coerce').dropna().astype(int)
                visible_trj_batch = trj_for_count_batch[trj_for_count_batch['particle'].isin(visible_pids_batch)]
                if not visible_trj_batch.empty:
                    batch_item_counts['start'] = visible_trj_batch[visible_trj_batch['frame'] == 0][
                        'particle'].nunique()
                    batch_item_counts['end'] = visible_trj_batch[visible_trj_batch['frame'] == num_frames_batch - 1][
                        'particle'].nunique()
                else:
                    batch_item_counts['start'] = 0
                    batch_item_counts['end'] = 0

            if batch_tracking_successful_item:
                batch_item_logger.info(
                    f"Batch processing for {os.path.basename(current_base_or_raw_path)} completed successfully.")

            current_dataset_log_content = current_dataset_log_capture.getvalue()

            # For batch mode, we need to construct a temporary main_app_state for save_results
            # to correctly get ancestry and pass ui_elements as None.
            temp_main_app_state_for_save = {
                'ancestry': {child: [parent] for parent, children in batch_cell_lineage.items() for child in children},
                # Reconstruct basic ancestry
                'params': main_app_state['params'],  # Use global params
                # Other minimal state if save_results expects it, but most are passed as args
            }

            save_results(
                path_for_saving_batch, batch_trj, batch_col_tuple, batch_col_weights,
                batch_id_masks, batch_cell_ids, batch_background_id, batch_color_list,
                batch_cell_color_idx, batch_cell_visibility,
                params['Pixel scale'][0], params['Pixel scale'][3],
                params['Show id\'s'][0], True, params['Show tracks'][0],
                params['Mask extension'][0],
                cell_lineage=batch_cell_lineage,
                ancestry_map=temp_main_app_state_for_save['ancestry'],  # Pass reconstructed ancestry
                use_thick_line=True,
                show_mitosis=params['Show Mitosis Labels'][0],
                counts=batch_item_counts,
                lineage_plot_widget=None,
                command_output_log=current_dataset_log_content,
                main_app_state=temp_main_app_state_for_save,  # Pass temp state
                ui_elements=None  # No UI elements to update in batch mode for enabling eval button
            )
            processed_count += 1
        except Exception as e_batch_item_proc:
            batch_item_logger.error(
                f'ERROR processing {current_base_or_raw_path}: {e_batch_item_proc}\n{traceback.format_exc()}')
            error_count += 1
        finally:
            batch_item_logger.removeHandler(batch_item_log_handler)
            batch_item_log_handler.close()
            current_dataset_log_capture.close()
            main_log_message = f"Batch item {os.path.basename(current_base_or_raw_path)} processing finished. "
            if 'current_dataset_log_content' in locals() and current_dataset_log_content:
                main_log_message += f"Log length: {len(current_dataset_log_content)} chars."
            else:
                main_log_message += "No specific log content captured for this item."
            logging.info(main_log_message)

        if progress_dialog.wasCanceled():
            break

    main_app_state['loaded_class_specific_masks'] = original_loaded_masks_global
    main_app_state['track_states'] = original_track_states_global
    main_app_state['ancestry'] = original_ancestry_global
    main_app_state['cell_lineage'] = original_cell_lineage_global

    progress_dialog.setValue(len(dataset_paths))
    summary_msg_batch = (
        f"Batch Processing Complete.\n"
        f"Total Datasets Attempted: {len(dataset_paths)}\n"
        f"Successfully Processed: {processed_count}\n"
        f"Errors: {error_count}\n"
        f"Time Taken: {time.time() - start_time_batch:.2f} seconds"
    )
    logging.info(summary_msg_batch)
    QMessageBox.information(win, 'Batch Processing Complete', summary_msg_batch)
