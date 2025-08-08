# cell_undoredo.py
# Manages undo/redo functionality for the cell tracking application.

import logging
import copy
import pandas as pd
import numpy as np
from collections import defaultdict

# --- Undo/Redo History ---
MAX_UNDO_HISTORY = 20
undo_stack = []
redo_stack = []

def capture_current_state(main_app_state):
    """
    Captures the current relevant state for undo/redo from the main application state.
    Args:
        main_app_state (dict): A dictionary containing all relevant global variables from cell_main.
    Returns:
        dict: A snapshot of the current state.
    """
    state = {
        'trj': main_app_state['trj'].copy(deep=True) if main_app_state.get('trj') is not None else pd.DataFrame(),
        'cell_lineage': copy.deepcopy(main_app_state['cell_lineage']) if main_app_state.get('cell_lineage') is not None else defaultdict(list),
        'ancestry': copy.deepcopy(main_app_state['ancestry']) if main_app_state.get('ancestry') is not None else defaultdict(list),
        'id_masks': main_app_state['id_masks'].copy() if main_app_state.get('id_masks') is not None else None,
        'id_masks_initial': main_app_state['id_masks_initial'].copy() if main_app_state.get('id_masks_initial') is not None else None,
        'cell_visibility': copy.deepcopy(main_app_state['cell_visibility']) if main_app_state.get('cell_visibility') is not None else {},
        'cell_color_idx': copy.deepcopy(main_app_state['cell_color_idx']) if main_app_state.get('cell_color_idx') is not None else {},
        'cell_ids': copy.deepcopy(main_app_state['cell_ids']) if main_app_state.get('cell_ids') is not None else [],
        'cell_frame_presence': copy.deepcopy(main_app_state['cell_frame_presence']) if main_app_state.get('cell_frame_presence') is not None else {},
        'cell_x': copy.deepcopy(main_app_state['cell_x']) if main_app_state.get('cell_x') is not None else defaultdict(dict),
        'cell_y': copy.deepcopy(main_app_state['cell_y']) if main_app_state.get('cell_y') is not None else defaultdict(dict),
        'track_states': copy.deepcopy(main_app_state['track_states']) if main_app_state.get('track_states') is not None else {},
        'next_available_daughter_id': main_app_state['next_available_daughter_id'],
        'has_lineage': main_app_state['has_lineage'],
        'wide_track_cell_id': main_app_state['wide_track_cell_id'],
        'color_list': copy.deepcopy(main_app_state['color_list']) if main_app_state.get('color_list') is not None else None,
        'background_id': main_app_state['background_id'],
        'current_frame_index': main_app_state['current_frame_index']
    }
    return state

def push_state_for_undo(main_app_state, ui_actions):
    """
    Saves the current state to the undo stack.
    Args:
        main_app_state (dict): The current application state.
        ui_actions (dict): Dictionary containing UI actions like 'undo_action', 'redo_action'.
    """
    global undo_stack, redo_stack, MAX_UNDO_HISTORY

    current_state_snapshot = capture_current_state(main_app_state)
    undo_stack.append(current_state_snapshot)

    if len(undo_stack) > MAX_UNDO_HISTORY:
        undo_stack.pop(0)

    redo_stack.clear()
    logging.info(f"UNDO: State pushed to undo stack. Stack size: {len(undo_stack)}")
    update_undo_redo_actions_enabled_state(ui_actions)

def restore_state_from_snapshot(state_snapshot, main_app_state, ui_elements, ui_callbacks):
    """
    Restores the global state from a given snapshot.
    Args:
        state_snapshot (dict): The state snapshot to restore.
        main_app_state (dict): The main application state dictionary to update.
        ui_elements (dict): Dictionary of UI elements.
        ui_callbacks (module): Module containing UI callback functions like populate_cell_table.
    """
    if not state_snapshot:
        logging.warning("UNDO/REDO: Attempted to restore an empty state snapshot.")
        return

    main_app_state['trj'] = state_snapshot['trj']
    main_app_state['cell_lineage'] = state_snapshot['cell_lineage']
    main_app_state['ancestry'] = state_snapshot['ancestry']
    main_app_state['id_masks'] = state_snapshot['id_masks']
    main_app_state['id_masks_initial'] = state_snapshot['id_masks_initial']
    main_app_state['cell_visibility'] = state_snapshot['cell_visibility']
    main_app_state['cell_color_idx'] = state_snapshot['cell_color_idx']
    main_app_state['cell_ids'] = state_snapshot['cell_ids']
    main_app_state['cell_frame_presence'] = state_snapshot['cell_frame_presence']
    main_app_state['cell_x'] = state_snapshot['cell_x']
    main_app_state['cell_y'] = state_snapshot['cell_y']
    main_app_state['track_states'] = state_snapshot['track_states']
    main_app_state['next_available_daughter_id'] = state_snapshot['next_available_daughter_id']
    main_app_state['has_lineage'] = state_snapshot['has_lineage']
    main_app_state['wide_track_cell_id'] = state_snapshot['wide_track_cell_id']
    main_app_state['color_list'] = state_snapshot['color_list']
    main_app_state['background_id'] = state_snapshot['background_id']
    main_app_state['current_frame_index'] = state_snapshot.get('current_frame_index', 0)

    logging.info("UNDO/REDO: State restored from snapshot.")

    # Refresh UI elements based on the restored state
    if ui_elements.get('table_cell_selection'):
        ui_callbacks.handle_populate_cell_table(main_app_state, ui_elements, ui_callbacks)

    v_mask = ui_elements.get('v_mask')
    if v_mask and main_app_state['id_masks'] is not None and main_app_state['color_list'] and main_app_state['cell_color_idx']:
        all_ids_numeric = [int(i) for i in main_app_state['cell_color_idx'].keys() if isinstance(i, (int, float)) and pd.notna(i)]
        lut_size = max(all_ids_numeric) + 1 if all_ids_numeric else 1
        actual_lut = np.zeros((lut_size, 3), dtype=np.uint8)
        
        bg_col_idx_restored = main_app_state['cell_color_idx'].get(main_app_state['background_id'])
        bg_color_restored = main_app_state['color_list'][bg_col_idx_restored] if bg_col_idx_restored is not None and 0 <= bg_col_idx_restored < len(main_app_state['color_list']) else (20, 20, 20)
        
        if main_app_state['background_id'] >= 0 and main_app_state['background_id'] < lut_size:
            actual_lut[main_app_state['background_id']] = bg_color_restored
        
        for cid_val, c_idx in main_app_state['cell_color_idx'].items():
            if isinstance(cid_val, (int, float)) and pd.notna(cid_val):
                cid_val_int = int(cid_val)
                if cid_val_int >= 0 and cid_val_int < lut_size and c_idx < len(main_app_state['color_list']):
                    actual_lut[cid_val_int] = main_app_state['color_list'][c_idx]

        v_mask.setImage(main_app_state['id_masks'], axes={'x': 1, 'y': 0, 't': 2})
        if v_mask.imageItem:
            v_mask.imageItem.setLookupTable(actual_lut)
        v_mask.setLevels(min=0, max=lut_size - 1 if lut_size > 0 else 0)

    v_raw_img = ui_elements.get('v_raw_img')
    if v_raw_img and main_app_state.get('raw_imgs') is not None:
        if 0 <= main_app_state['current_frame_index'] < main_app_state['raw_imgs'].shape[2]:
            v_raw_img.setCurrentIndex(main_app_state['current_frame_index'])
    
    if v_mask and main_app_state.get('id_masks') is not None:
        if 0 <= main_app_state['current_frame_index'] < main_app_state['id_masks'].shape[2]:
            v_mask.setCurrentIndex(main_app_state['current_frame_index'])
        elif main_app_state['id_masks'].shape[2] > 0:
            v_mask.setCurrentIndex(0)

    ui_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)

    lineage_tree_widget = ui_elements.get('lineage_tree_widget')
    if lineage_tree_widget:
        lineage_tree_widget.set_data(
            main_app_state['trj'] if main_app_state.get('trj') is not None else pd.DataFrame(),
            main_app_state['color_list'] if main_app_state.get('color_list') is not None else [],
            main_app_state['cell_color_idx'] if main_app_state.get('cell_color_idx') is not None else {},
            main_app_state['cell_visibility'] if main_app_state.get('cell_visibility') is not None else {},
            main_app_state['track_states'] if main_app_state.get('track_states') is not None else {},
            main_app_state['ancestry'] if main_app_state.get('ancestry') is not None else defaultdict(list)
        )
        if lineage_tree_widget.current_root_id is not None and lineage_tree_widget.current_root_id in lineage_tree_widget.all_roots:
            lineage_tree_widget.draw_lineage_tree_for_single_root(lineage_tree_widget.current_root_id)
        else:
            lineage_tree_widget.draw_all_lineage_trees()
        lineage_tree_widget.set_highlighted_track(main_app_state['wide_track_cell_id'])

    b_merge_track = ui_elements.get('b_merge_track')
    if b_merge_track:
        can_merge_restored = False
        if main_app_state['wide_track_cell_id'] is not None:
            parents_of_selected_restored = main_app_state['ancestry'].get(main_app_state['wide_track_cell_id'], [])
            if parents_of_selected_restored and parents_of_selected_restored[0] != main_app_state['background_id']:
                can_merge_restored = True
        b_merge_track.setEnabled(can_merge_restored)

    logging.info("UNDO/REDO: UI refreshed after state restoration.")


def undo_last_operation(main_app_state, ui_elements, ui_callbacks, ui_actions):
    """
    Performs an undo operation.
    Args:
        main_app_state (dict): The current application state.
        ui_elements (dict): Dictionary of UI elements.
        ui_callbacks (module): Module containing UI callback functions.
        ui_actions (dict): Dictionary containing UI actions like 'undo_action', 'redo_action'.
    """
    global undo_stack, redo_stack
    if not undo_stack:
        logging.info("UNDO: Stack is empty. Nothing to undo.")
        return

    state_to_redo = capture_current_state(main_app_state)
    redo_stack.append(state_to_redo)
    if len(redo_stack) > MAX_UNDO_HISTORY: redo_stack.pop(0)

    last_saved_state = undo_stack.pop()
    restore_state_from_snapshot(last_saved_state, main_app_state, ui_elements, ui_callbacks)
    logging.info(f"UNDO: Operation undone. Undo stack size: {len(undo_stack)}, Redo stack size: {len(redo_stack)}")
    update_undo_redo_actions_enabled_state(ui_actions)

def redo_last_operation(main_app_state, ui_elements, ui_callbacks, ui_actions):
    """
    Performs a redo operation.
    Args:
        main_app_state (dict): The current application state.
        ui_elements (dict): Dictionary of UI elements.
        ui_callbacks (module): Module containing UI callback functions.
        ui_actions (dict): Dictionary containing UI actions like 'undo_action', 'redo_action'.
    """
    global undo_stack, redo_stack
    if not redo_stack:
        logging.info("REDO: Stack is empty. Nothing to redo.")
        return

    state_to_undo = capture_current_state(main_app_state)
    undo_stack.append(state_to_undo)
    if len(undo_stack) > MAX_UNDO_HISTORY: undo_stack.pop(0)

    state_to_restore = redo_stack.pop()
    restore_state_from_snapshot(state_to_restore, main_app_state, ui_elements, ui_callbacks)
    logging.info(f"REDO: Operation redone. Undo stack size: {len(undo_stack)}, Redo stack size: {len(redo_stack)}")
    update_undo_redo_actions_enabled_state(ui_actions)

def clear_undo_redo_stacks(ui_actions):
    """
    Clears undo and redo stacks.
    Args:
        ui_actions (dict): Dictionary containing UI actions like 'undo_action', 'redo_action'.
    """
    global undo_stack, redo_stack
    undo_stack.clear()
    redo_stack.clear()
    update_undo_redo_actions_enabled_state(ui_actions)
    logging.info("UNDO/REDO: Stacks cleared.")

def update_undo_redo_actions_enabled_state(ui_actions):
    """
    Updates the enabled state of undo/redo QActions.
    Args:
        ui_actions (dict): Dictionary containing UI actions like 'undo_action', 'redo_action'.
    """
    global undo_stack, redo_stack
    undo_action = ui_actions.get('undo_action')
    redo_action = ui_actions.get('redo_action')

    if undo_action:
        undo_action.setEnabled(len(undo_stack) > 0)
    if redo_action:
        redo_action.setEnabled(len(redo_stack) > 0)
