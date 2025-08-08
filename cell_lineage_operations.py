# cell_lineage_operations.py
# Handles interactive lineage editing operations.
# V3 (Gemini): Added detailed debug logging for fusion operation.
# V4 (Gemini): Refined parent_particle assignment in fusion, more logging.
# V5 (Gemini): Further refined parent_particle and lineage updates for "Insert & Split" operation.
# V6 (Gemini): Focused refinement on parent/child relationships for "Insert & Split" to fix plotting.

import logging
import traceback
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QLabel, QComboBox, QDialogButtonBox, QApplication
import pyqtgraph as pg
from cell_tracking import initialize_next_daughter_id, get_new_daughter_id
import cell_undoredo


def _is_descendant_of_any(potential_descendant_id, ancestor_ids_list, current_ancestry_map, current_lineage_map):
    """
    Checks if potential_descendant_id is a descendant of ANY of the ancestor_ids_list.
    Uses current_lineage_map for traversal downwards.
    """
    if not ancestor_ids_list: return False

    for ancestor_id_check in ancestor_ids_list:
        queue = [ancestor_id_check]
        visited_descendants = {ancestor_id_check}
        depth = 0
        max_depth = len(current_lineage_map.keys()) + len(
            current_ancestry_map.keys()) + 10 if current_lineage_map else 1000

        while queue and depth < max_depth:
            depth += 1
            current_parent_in_search = queue.pop(0)

            children_of_current_parent = current_lineage_map.get(current_parent_in_search, [])
            for child_node in children_of_current_parent:
                if child_node == potential_descendant_id:
                    return True
                if child_node not in visited_descendants:
                    visited_descendants.add(child_node)
                    queue.append(child_node)
        if depth >= max_depth:
            logging.warning(
                f"Max depth reached in _is_descendant_of_any for ancestor {ancestor_id_check}, descendant {potential_descendant_id}")
    return False


def handle_merge_track_with_parent(main_app_state, ui_elements, ui_callbacks,
                                   ui_actions_module):
    """
    Merges the currently selected track (wide_track_cell_id) with its primary parent.
    ui_actions_module is cell_undoredo.
    """
    wide_track_cell_id = main_app_state.get('wide_track_cell_id')
    ancestry = main_app_state.get('ancestry')
    win = ui_elements.get('win')

    if wide_track_cell_id is None:
        QMessageBox.warning(win, "Selection Error", "Please select a track first.")
        return

    child_to_merge_id = wide_track_cell_id
    parents_of_child = ancestry.get(child_to_merge_id, [])

    if not parents_of_child:
        QMessageBox.warning(win, "Selection Error", f"Track {child_to_merge_id} has no parents to merge with.")
        return

    actual_parent_id = parents_of_child[0]
    background_id = main_app_state.get('background_id', 0)

    if child_to_merge_id == actual_parent_id or actual_parent_id == background_id:
        QMessageBox.critical(win, "Merge Error", "Invalid merge condition (cannot merge with self or background).")
        return

    if _is_descendant_of_any(actual_parent_id, [child_to_merge_id], ancestry, main_app_state.get('cell_lineage')):
        QMessageBox.critical(win, "Merge Error",
                             f"Cannot merge {child_to_merge_id} into {actual_parent_id} as it would create a cycle (parent is a descendant of child).")
        return

    if QMessageBox.question(win, 'Confirm Merge', f"Merge Track {child_to_merge_id} into Parent {actual_parent_id}?",
                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No:
        return

    cell_undoredo.push_state_for_undo(main_app_state, ui_elements)

    trj = main_app_state.get('trj')
    cell_lineage = main_app_state.get('cell_lineage')
    id_masks = main_app_state.get('id_masks')
    id_masks_initial = main_app_state.get('id_masks_initial')

    with pg.BusyCursor():
        try:
            original_grandparents = ancestry.get(actual_parent_id, []).copy()
            daughters_of_child_to_merge = cell_lineage.get(child_to_merge_id, []).copy()

            for daughter_id in daughters_of_child_to_merge:
                if daughter_id == actual_parent_id:
                    QMessageBox.critical(win, "Merge Error", "Cycle detected during daughter reparenting.")
                    if cell_undoredo.undo_stack: cell_undoredo.undo_stack.pop()
                    cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)
                    return

                if daughter_id in ancestry:
                    if child_to_merge_id in ancestry[daughter_id]:
                        ancestry[daughter_id].remove(child_to_merge_id)
                    if actual_parent_id not in ancestry[daughter_id]:
                        ancestry[daughter_id].append(actual_parent_id)
                    ancestry[daughter_id].sort()
                    if not ancestry[daughter_id]: del ancestry[daughter_id]
                else:
                    ancestry[daughter_id] = [actual_parent_id]
                trj.loc[trj['particle'] == daughter_id, 'parent_particle'] = ancestry[daughter_id][0] if ancestry.get(
                    daughter_id) else pd.NA

            primary_grandparent_for_trj = original_grandparents[0] if original_grandparents else pd.NA
            trj.loc[trj['particle'] == child_to_merge_id, 'parent_particle'] = primary_grandparent_for_trj
            trj.loc[trj['particle'] == child_to_merge_id, 'particle'] = actual_parent_id

            if id_masks is not None: id_masks[id_masks == child_to_merge_id] = actual_parent_id
            if id_masks_initial is not None: id_masks_initial[id_masks_initial == child_to_merge_id] = actual_parent_id

            for p_of_child in parents_of_child:
                if p_of_child in cell_lineage and child_to_merge_id in cell_lineage[p_of_child]:
                    cell_lineage[p_of_child].remove(child_to_merge_id)
                    if not cell_lineage[p_of_child]: del cell_lineage[p_of_child]

            cell_lineage.setdefault(actual_parent_id, []).extend(
                d for d in daughters_of_child_to_merge if d not in cell_lineage.get(actual_parent_id, []))
            if cell_lineage.get(actual_parent_id): cell_lineage[actual_parent_id].sort()

            if child_to_merge_id in cell_lineage: del cell_lineage[child_to_merge_id]
            if child_to_merge_id in ancestry: del ancestry[child_to_merge_id]

            if original_grandparents:
                ancestry[actual_parent_id] = sorted(list(set(original_grandparents)))
            elif actual_parent_id in ancestry:
                del ancestry[actual_parent_id]

            for d_to_remove_key in ['cell_ids', 'cell_color_idx', 'cell_visibility', 'cell_frame_presence',
                                    'track_data_per_frame', 'track_states']:
                d_to_remove = main_app_state.get(d_to_remove_key)
                if d_to_remove is not None and child_to_merge_id in d_to_remove:
                    if isinstance(d_to_remove, list):
                        try:
                            d_to_remove.remove(child_to_merge_id)
                        except ValueError:
                            pass
                    else:
                        del d_to_remove[child_to_merge_id]

            cell_x = main_app_state.get('cell_x');
            cell_y = main_app_state.get('cell_y')
            if cell_x:
                for kx in [k for k in cell_x if k[0] == child_to_merge_id]: del cell_x[kx]
            if cell_y:
                for ky in [k for k in cell_y if k[0] == child_to_merge_id]: del cell_y[ky]

            for item_d_key in ['cell_ids_raw_img', 'cell_ids_mask', 'track_plots_per_cell']:
                item_d = main_app_state.get(item_d_key)
                plot_widget_key = 'pi_raw_img' if 'raw_img' in item_d_key or 'plots' in item_d_key else 'pi_mask'
                plot_widget = ui_elements.get(plot_widget_key)
                if item_d and plot_widget and child_to_merge_id in item_d and item_d[child_to_merge_id].scene():
                    plot_widget.removeItem(item_d[child_to_merge_id])
                    del item_d[child_to_merge_id]

            main_app_state['cell_visibility'][actual_parent_id] = True
            main_app_state['has_lineage'] = bool(cell_lineage or ancestry)
            main_app_state['wide_track_cell_id'] = None

            b_merge_track_btn = ui_elements.get('button_widgets_map', {}).get('merge_selected_track_with_parent')
            if b_merge_track_btn: b_merge_track_btn.setEnabled(False)

            initialize_next_daughter_id(trj)

            ui_callbacks.handle_populate_cell_table(main_app_state, ui_elements, ui_callbacks)
            ui_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)
            lineage_tree_widget = ui_elements.get('lineage_tree_widget')
            if lineage_tree_widget:
                lineage_tree_widget.set_data(trj, main_app_state.get('color_list'),
                                             main_app_state.get('cell_color_idx'),
                                             main_app_state.get('cell_visibility'), main_app_state.get('track_states'),
                                             ancestry, main_app_state.get('params'))
                lineage_tree_widget.draw_all_lineage_trees()
                lineage_tree_widget.set_highlighted_track(None)
                # Force immediate repaint for instant visual feedback
                lineage_tree_widget.plot_item.update()
                # Process events to ensure immediate UI update
                QApplication.processEvents()

            QMessageBox.information(win, "Merge Successful",
                                    f"Track {child_to_merge_id} merged into {actual_parent_id}.")

        except Exception as e_merge:
            logging.error(f"Error during merge operation: {e_merge}\n{traceback.format_exc()}")
            QMessageBox.critical(win, "Merge Error", f"An error occurred: {e_merge}")
            if cell_undoredo.undo_stack: cell_undoredo.undo_stack.pop()
            cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)
            return


def process_node_operation_request(dragged_id, target_id, operation_name,
                                   main_app_state, ui_elements, ui_callbacks,
                                   ui_actions_module):
    trj = main_app_state.get('trj')
    win = ui_elements.get('win')

    if trj is None or trj.empty:
        logging.warning("No trajectory data for lineage operation.")
        return

    if operation_name == "Cancel":
        logging.info(f"Lineage operation cancelled by user for {dragged_id} onto {target_id}.")
        return

    cell_undoredo.push_state_for_undo(main_app_state, ui_elements)

    modified = False
    cell_lineage = main_app_state.get('cell_lineage')
    ancestry = main_app_state.get('ancestry')
    id_masks = main_app_state.get('id_masks')
    id_masks_initial = main_app_state.get('id_masks_initial')

    try:
        with pg.BusyCursor():
            def creates_cycle_local(potential_parent_id, potential_child_id):
                return _is_descendant_of_any(potential_parent_id, [potential_child_id], ancestry, cell_lineage)

            if operation_name == "Join Track (Merge Dragged into Target)":
                if creates_cycle_local(target_id, dragged_id) or creates_cycle_local(dragged_id, target_id):
                    QMessageBox.critical(win, "Merge Error", "Cannot merge: Operation would create a cycle.")
                    raise ValueError("Cycle error in Join Track")

                original_parents_of_dragged = ancestry.get(dragged_id, []).copy()
                original_children_of_dragged = cell_lineage.get(dragged_id, []).copy()
                original_parents_of_target = ancestry.get(target_id, []).copy()

                trj.loc[trj['particle'] == dragged_id, 'particle'] = target_id

                new_parents_for_target = set(original_parents_of_dragged)
                new_parents_for_target.update(p for p in original_parents_of_target if p != dragged_id)
                new_parents_for_target.discard(target_id)

                if new_parents_for_target:
                    ancestry[target_id] = sorted(list(new_parents_for_target))
                    trj.loc[trj['particle'] == target_id, 'parent_particle'] = ancestry[target_id][0]
                elif target_id in ancestry:
                    del ancestry[target_id]
                    trj.loc[trj['particle'] == target_id, 'parent_particle'] = pd.NA

                for p_dragged in original_parents_of_dragged:
                    if p_dragged in cell_lineage and dragged_id in cell_lineage[p_dragged]:
                        cell_lineage[p_dragged].remove(dragged_id)
                        if not cell_lineage[p_dragged]: del cell_lineage[p_dragged]

                current_target_children = cell_lineage.get(target_id, [])
                new_children_for_target = set(current_target_children)
                new_children_for_target.update(c for c in original_children_of_dragged if c != target_id)

                if new_children_for_target:
                    cell_lineage[target_id] = sorted(list(new_children_for_target))
                elif target_id in cell_lineage:
                    del cell_lineage[target_id]

                for child_d in original_children_of_dragged:
                    if child_d == target_id: continue
                    current_child_parents = set(ancestry.get(child_d, []))
                    current_child_parents.discard(dragged_id)
                    current_child_parents.add(target_id)
                    current_child_parents.discard(child_d)
                    if current_child_parents:
                        ancestry[child_d] = sorted(list(current_child_parents))
                        trj.loc[trj['particle'] == child_d, 'parent_particle'] = ancestry[child_d][0]
                    elif child_d in ancestry:
                        del ancestry[child_d]
                        trj.loc[trj['particle'] == child_d, 'parent_particle'] = pd.NA

                if dragged_id in cell_lineage: del cell_lineage[dragged_id]
                if dragged_id in ancestry: del ancestry[dragged_id]

                if id_masks is not None: id_masks[id_masks == dragged_id] = target_id
                if id_masks_initial is not None: id_masks_initial[id_masks_initial == dragged_id] = target_id

                for d_key in ['cell_visibility', 'track_states', 'cell_frame_presence', 'cell_color_idx']:
                    state_dict = main_app_state.get(d_key)
                    if state_dict and dragged_id in state_dict: del state_dict[dragged_id]
                if main_app_state.get('cell_ids') and dragged_id in main_app_state['cell_ids']:
                    main_app_state['cell_ids'].remove(dragged_id)

                cell_x = main_app_state.get('cell_x');
                cell_y = main_app_state.get('cell_y')
                if cell_x:
                    for kx in [k for k in cell_x if k[0] == dragged_id]: del cell_x[kx]
                if cell_y:
                    for ky in [k for k in cell_y if k[0] == dragged_id]: del cell_y[ky]

                for item_d_key_op in ['cell_ids_raw_img', 'cell_ids_mask', 'track_plots_per_cell']:
                    item_d_op = main_app_state.get(item_d_key_op)
                    plot_widget_key_op = 'pi_raw_img' if 'raw_img' in item_d_key_op or 'plots' in item_d_key_op else 'pi_mask'
                    plot_widget_op_ref = ui_elements.get(plot_widget_key_op)
                    if item_d_op and plot_widget_op_ref and dragged_id in item_d_op and item_d_op[dragged_id].scene():
                        plot_widget_op_ref.removeItem(item_d_op[dragged_id])
                        del item_d_op[dragged_id]

                main_app_state.get('cell_visibility', {})[target_id] = True
                modified = True

            elif operation_name == "Set Target as Parent of Dragged":
                if creates_cycle_local(target_id, dragged_id):
                    QMessageBox.warning(win, "Invalid Operation",
                                        "Cycle detected: Target is already a descendant of Dragged.")
                    raise ValueError("Cycle error in Set Target as Parent")
                old_parents_of_dragged = ancestry.get(dragged_id, []).copy()
                for old_p in old_parents_of_dragged:
                    if old_p in cell_lineage and dragged_id in cell_lineage[old_p]:
                        cell_lineage[old_p].remove(dragged_id)
                        if not cell_lineage[old_p]: del cell_lineage[old_p]
                ancestry[dragged_id] = [target_id]
                trj.loc[trj['particle'] == dragged_id, 'parent_particle'] = target_id
                cell_lineage.setdefault(target_id, []).append(dragged_id)
                cell_lineage[target_id] = sorted(list(set(cell_lineage[target_id])))
                modified = True

            elif operation_name == "Set Dragged as Parent of Target":
                if creates_cycle_local(dragged_id, target_id):
                    QMessageBox.warning(win, "Invalid Operation",
                                        "Cycle detected: Dragged is already a descendant of Target.")
                    raise ValueError("Cycle error in Set Dragged as Parent")
                old_parents_of_target = ancestry.get(target_id, []).copy()
                for old_p in old_parents_of_target:
                    if old_p in cell_lineage and target_id in cell_lineage[old_p]:
                        cell_lineage[old_p].remove(target_id)
                        if not cell_lineage[old_p]: del cell_lineage[old_p]
                ancestry[target_id] = [dragged_id]
                trj.loc[trj['particle'] == target_id, 'parent_particle'] = dragged_id
                cell_lineage.setdefault(dragged_id, []).append(target_id)
                cell_lineage[dragged_id] = sorted(list(set(cell_lineage[dragged_id])))
                modified = True

            elif operation_name == "Set Dragged as Additional Parent of Target":
                if creates_cycle_local(dragged_id, target_id):
                    QMessageBox.warning(win, "Invalid Operation",
                                        "Cycle detected: Dragged is already a descendant of Target.")
                    raise ValueError("Cycle error in Set Dragged as Additional Parent")
                if target_id in ancestry.get(dragged_id, []):
                    QMessageBox.warning(win, "Invalid Operation",
                                        "Cycle detected: Target is already a parent of Dragged.")
                    raise ValueError("Cycle error: Target is parent of Dragged")

                ancestry.setdefault(target_id, []).append(dragged_id)
                ancestry[target_id] = sorted(list(set(ancestry[target_id])))
                trj.loc[trj['particle'] == target_id, 'parent_particle'] = ancestry[target_id][0]
                cell_lineage.setdefault(dragged_id, []).append(target_id)
                cell_lineage[dragged_id] = sorted(list(set(cell_lineage[dragged_id])))
                modified = True

    except ValueError as ve:
        logging.warning(f"Lineage operation '{operation_name}' aborted due to: {ve}")
        if cell_undoredo.undo_stack: cell_undoredo.undo_stack.pop()
        cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)
        return
    except Exception as e_op_node_proc:
        logging.error(f"Error during node operation '{operation_name}': {e_op_node_proc}\n{traceback.format_exc()}")
        QMessageBox.critical(win, "Operation Error", f"An error occurred: {e_op_node_proc}")
        if cell_undoredo.undo_stack: cell_undoredo.undo_stack.pop()
        cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)
        return

    if modified:
        main_app_state['has_lineage'] = bool(cell_lineage or ancestry)
        initialize_next_daughter_id(trj)

        ui_callbacks.handle_populate_cell_table(main_app_state, ui_elements, ui_callbacks)
        ui_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)
        lineage_tree_widget = ui_elements.get('lineage_tree_widget')
        if lineage_tree_widget:
            lineage_tree_widget.set_data(trj, main_app_state.get('color_list'), main_app_state.get('cell_color_idx'),
                                         main_app_state.get('cell_visibility'), main_app_state.get('track_states'),
                                         ancestry, main_app_state.get('params'))
            lineage_tree_widget.draw_all_lineage_trees()

        if operation_name == "Join Track (Merge Dragged into Target)" and id_masks is not None and \
                ui_elements.get('v_mask') and ui_elements.get('v_mask').imageItem and \
                main_app_state.get('color_list') and main_app_state.get('cell_color_idx'):
            ui_callbacks.handle_update_cell_visibility(target_id, True, main_app_state, ui_elements)
    else:
        if cell_undoredo.undo_stack: cell_undoredo.undo_stack.pop()
        cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)
        logging.info(f"UNDO: Popped state as operation '{operation_name}' resulted in no change or was cancelled.")


def process_node_break_request(clicked_track_id, break_frame_index, main_app_state, ui_elements, ui_callbacks,
                               ui_actions_module):
    logging.info(f"LINEAGE_EDIT (Break): Track {clicked_track_id} at frame {break_frame_index}.")
    win = ui_elements.get('win')
    trj = main_app_state.get('trj')
    if trj is None or trj.empty:
        QMessageBox.warning(win, "Break Error", "Trajectory data is empty.")
        return

    original_track_frames_series = trj[trj['particle'] == clicked_track_id]['frame']
    if original_track_frames_series.empty:
        QMessageBox.warning(win, "Break Error", f"Track ID {clicked_track_id} not found.")
        return

    min_frame_orig_track = original_track_frames_series.min()
    max_frame_orig_track = original_track_frames_series.max()

    if not (min_frame_orig_track <= break_frame_index <= max_frame_orig_track + 1):
        QMessageBox.warning(win, "Break Error",
                            "Break frame is outside the track's current range (or one past its end).")
        return

    frames_in_new_segment = original_track_frames_series[original_track_frames_series >= break_frame_index]
    frames_in_old_segment = original_track_frames_series[original_track_frames_series < break_frame_index]

    if frames_in_new_segment.empty and break_frame_index <= max_frame_orig_track:
        QMessageBox.warning(win, "Break Error", "Break operation would result in an empty new segment from this track.")
        return
    if frames_in_old_segment.empty and frames_in_new_segment.empty:
        QMessageBox.warning(win, "Break Error", "Break operation results in no valid segments.")
        return

    confirm_msg = f"Are you sure you want to break track {clicked_track_id} at frame {break_frame_index}?\n"
    if frames_in_old_segment.empty:
        confirm_msg += f"The entire track {clicked_track_id} will become a new root track with a new ID."
    elif frames_in_new_segment.empty:
        confirm_msg += f"Track {clicked_track_id} will end at frame {break_frame_index - 1}. No new segment created from this track."
    else:
        confirm_msg += f"The segment from frame {break_frame_index} onwards will become a new track."

    if QMessageBox.question(win, 'Confirm Break Lineage', confirm_msg, QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.No) == QMessageBox.No:
        return

    cell_undoredo.push_state_for_undo(main_app_state, ui_elements)

    cell_lineage = main_app_state.get('cell_lineage')
    ancestry = main_app_state.get('ancestry')
    id_masks = main_app_state.get('id_masks')
    id_masks_initial = main_app_state.get('id_masks_initial')

    with pg.BusyCursor():
        try:
            if main_app_state.get('next_available_daughter_id') is None or \
                    main_app_state.get('next_available_daughter_id') <= (
            trj['particle'].max() if not trj.empty and trj['particle'].notna().any() else 0):
                initialize_next_daughter_id(trj)

            new_segment_id = -1

            if not frames_in_new_segment.empty:
                new_segment_id = get_new_daughter_id()
                logging.info(
                    f"LINEAGE_EDIT (Break): New segment ID: {new_segment_id} for tail of {clicked_track_id} from frame {break_frame_index}.")
                new_segment_trj_mask = (trj['particle'] == clicked_track_id) & (trj['frame'] >= break_frame_index)
                trj.loc[new_segment_trj_mask, 'particle'] = new_segment_id
                
                # Set the new segment's parent to the original parent of the clicked track
                original_parent_id = None
                if clicked_track_id in ancestry:
                    original_parent_id = ancestry[clicked_track_id][0] if ancestry[clicked_track_id] else None
                
                if original_parent_id is not None:
                    trj.loc[new_segment_trj_mask, 'parent_particle'] = original_parent_id
                else:
                    # If no original parent, new segment becomes a root track
                    trj.loc[new_segment_trj_mask, 'parent_particle'] = pd.NA

                if id_masks is not None and id_masks_initial is not None:
                    for frame_idx_mask_val in frames_in_new_segment.unique():
                        if 0 <= frame_idx_mask_val < id_masks.shape[2]:
                            original_pixels_this_frame = (
                                        id_masks_initial[:, :, frame_idx_mask_val] == clicked_track_id)
                            id_masks[:, :, frame_idx_mask_val][original_pixels_this_frame] = new_segment_id
                            id_masks_initial[:, :, frame_idx_mask_val][original_pixels_this_frame] = new_segment_id

            original_children_of_clicked_track = cell_lineage.get(clicked_track_id, []).copy()
            children_for_new_segment = []
            children_for_old_segment = []

            for child_id_val in original_children_of_clicked_track:
                child_start_frame_series = trj[trj['particle'] == child_id_val]['frame']
                if child_start_frame_series.empty: continue
                child_start_frame = child_start_frame_series.min()
                if pd.isna(child_start_frame): continue

                if new_segment_id != -1 and child_start_frame >= break_frame_index:
                    # Children that start after break frame go to the new segment
                    children_for_new_segment.append(child_id_val)
                    ancestry[child_id_val] = [new_segment_id]
                    trj.loc[trj['particle'] == child_id_val, 'parent_particle'] = new_segment_id
                else:
                    # Children that start before break frame - they should not be daughters of the breaking trail
                    # They should become root tracks or be reassigned based on their actual parent
                    children_for_old_segment.append(child_id_val)
                    # Clear their parent relationship since the breaking trail should have no daughters
                    if child_id_val in ancestry: 
                        del ancestry[child_id_val]
                    trj.loc[trj['particle'] == child_id_val, 'parent_particle'] = pd.NA

            # Update lineage relationships for the new segment
            if new_segment_id != -1:
                # Get the original parent of the clicked track
                original_parent_id = None
                if clicked_track_id in ancestry:
                    original_parent_id = ancestry[clicked_track_id][0] if ancestry[clicked_track_id] else None
                
                # Set the new segment's parent to the original parent (not the clicked track)
                if original_parent_id is not None:
                    ancestry[new_segment_id] = [original_parent_id]
                    # Add new segment as daughter of the original parent
                    if original_parent_id not in cell_lineage:
                        cell_lineage[original_parent_id] = []
                    cell_lineage[original_parent_id].append(new_segment_id)
                    cell_lineage[original_parent_id] = sorted(cell_lineage[original_parent_id])
                else:
                    # If no original parent, new segment becomes a root track
                    ancestry[new_segment_id] = []
                
                # Add any children that belong to the new segment
                if children_for_new_segment:
                    cell_lineage[new_segment_id] = sorted(children_for_new_segment)

            # Update lineage for the original track (breaking trail)
            if frames_in_old_segment.empty:
                # If old segment is empty, remove it from lineage
                if clicked_track_id in cell_lineage: 
                    del cell_lineage[clicked_track_id]
                if clicked_track_id in ancestry:
                    del ancestry[clicked_track_id]
            else:
                # Clear all daughter IDs from the original track (breaking trail)
                if clicked_track_id in cell_lineage:
                    del cell_lineage[clicked_track_id]
                # Keep the original track in ancestry if it has a parent
                if clicked_track_id in ancestry and not ancestry[clicked_track_id]:
                    del ancestry[clicked_track_id]

            # Note: We don't remove new_segment_id from ancestry as it should have its parent set correctly above
            if frames_in_old_segment.empty and clicked_track_id in ancestry:
                del ancestry[clicked_track_id]

            if new_segment_id != -1:
                cell_ids_list_ref = main_app_state.get('cell_ids', [])
                if isinstance(cell_ids_list_ref, list) and new_segment_id not in cell_ids_list_ref:
                    cell_ids_list_ref.append(new_segment_id)
                    logging.info(f"LINEAGE_EDIT (Break): Added new segment {new_segment_id} to cell_ids list. Total cells: {len(cell_ids_list_ref)}")
                elif isinstance(cell_ids_list_ref, set):
                    cell_ids_list_ref.add(new_segment_id)
                    logging.info(f"LINEAGE_EDIT (Break): Added new segment {new_segment_id} to cell_ids set. Total cells: {len(cell_ids_list_ref)}")

                cell_color_idx = main_app_state.get('cell_color_idx');
                color_list = main_app_state.get('color_list')
                original_color_idx_val = cell_color_idx.get(clicked_track_id)
                cell_color_idx[new_segment_id] = original_color_idx_val if original_color_idx_val is not None else (
                    len(cell_color_idx) % len(color_list) if color_list and len(color_list) > 0 else 0)

                # Ensure the new segment is visible and has proper state
                main_app_state.get('cell_visibility')[new_segment_id] = True  # Always make new segments visible
                main_app_state.get('track_states')[new_segment_id] = main_app_state.get('track_states').get(
                    clicked_track_id, "N/A")
                logging.info(f"LINEAGE_EDIT (Break): Set new segment {new_segment_id} visibility to True")

                cell_frame_presence = main_app_state.get('cell_frame_presence')
                all_original_frames_set = cell_frame_presence.get(clicked_track_id, set()).copy()
                cell_frame_presence[new_segment_id] = {f for f in all_original_frames_set if f >= break_frame_index}

                cell_x_break = main_app_state.get('cell_x');
                cell_y_break = main_app_state.get('cell_y')
                for frame_val_tr in cell_frame_presence.get(new_segment_id, set()):
                    if (clicked_track_id, frame_val_tr) in cell_x_break: 
                        cell_x_break[new_segment_id, frame_val_tr] = cell_x_break.pop((clicked_track_id, frame_val_tr))
                    if (clicked_track_id, frame_val_tr) in cell_y_break: 
                        cell_y_break[new_segment_id, frame_val_tr] = cell_y_break.pop((clicked_track_id, frame_val_tr))
                
                logging.info(f"LINEAGE_EDIT (Break): Transferred coordinates for new segment {new_segment_id} for frames: {list(cell_frame_presence.get(new_segment_id, set()))}")

            cell_frame_presence_old = main_app_state.get('cell_frame_presence')
            all_original_frames_set_for_old = cell_frame_presence_old.get(clicked_track_id, set()).copy()
            if frames_in_old_segment.empty:
                logging.info(f"LINEAGE_EDIT (Break): Old segment of {clicked_track_id} is empty. Cleaning up.")
                if clicked_track_id in main_app_state.get('cell_visibility'): main_app_state.get('cell_visibility')[
                    clicked_track_id] = False

                cell_ids_list_ref_old = main_app_state.get('cell_ids', [])
                if isinstance(cell_ids_list_ref_old, list) and clicked_track_id in cell_ids_list_ref_old:
                    try:
                        cell_ids_list_ref_old.remove(clicked_track_id)
                    except ValueError:
                        pass
                elif isinstance(cell_ids_list_ref_old, set):
                    cell_ids_list_ref_old.discard(clicked_track_id)

                for d_cleanup_key in ['cell_color_idx', 'track_states', 'cell_frame_presence', 'ancestry',
                                      'cell_lineage']:
                    d_cleanup_val = main_app_state.get(d_cleanup_key)
                    if d_cleanup_val is not None and clicked_track_id in d_cleanup_val: del d_cleanup_val[
                        clicked_track_id]

                cell_x_cleanup = main_app_state.get('cell_x');
                cell_y_cleanup = main_app_state.get('cell_y')
                keys_to_del_x_old_val = [k for k in cell_x_cleanup if k[0] == clicked_track_id];
                for k_del_x_val in keys_to_del_x_old_val: del cell_x_cleanup[k_del_x_val]
                keys_to_del_y_old_val = [k for k in cell_y_cleanup if k[0] == clicked_track_id];
                for k_del_y_val in keys_to_del_y_old_val: del cell_y_cleanup[k_del_y_val]

                for plot_item_dict_key_break in ['cell_ids_raw_img', 'cell_ids_mask', 'track_plots_per_cell']:
                    plot_item_dict_break = main_app_state.get(plot_item_dict_key_break)
                    plot_widget_key_break = 'pi_raw_img' if 'raw_img' in plot_item_dict_key_break or 'plots' in plot_item_dict_key_break else 'pi_mask'
                    plot_widget_break = ui_elements.get(plot_widget_key_break)
                    if plot_item_dict_break and plot_widget_break and clicked_track_id in plot_item_dict_break:
                        if plot_item_dict_break[clicked_track_id].scene(): plot_widget_break.removeItem(
                            plot_item_dict_break[clicked_track_id])
                        del plot_item_dict_break[clicked_track_id]
            else:
                cell_frame_presence_old[clicked_track_id] = {f for f in all_original_frames_set_for_old if
                                                             f < break_frame_index}
                if not cell_frame_presence_old[clicked_track_id] and clicked_track_id in cell_frame_presence_old:
                    del cell_frame_presence_old[clicked_track_id]

            main_app_state['has_lineage'] = bool(cell_lineage or ancestry)
            logging.info(
                f"LINEAGE_EDIT (Break): Track {clicked_track_id} broken. New tail ID (if any): {new_segment_id}.")
            
            # Log the updated lineage relationships
            if new_segment_id != -1:
                # Safely get the original parent ID
                ancestry_list = ancestry.get(new_segment_id, [])
                original_parent_id = ancestry_list[0] if ancestry_list else None
                logging.info(f"LINEAGE_EDIT (Break): Updated lineage - Original track {clicked_track_id} (breaking trail) has no daughters")
                logging.info(f"LINEAGE_EDIT (Break): New segment {new_segment_id} has parent: {original_parent_id}")
                if original_parent_id:
                    logging.info(f"LINEAGE_EDIT (Break): Original parent {original_parent_id} now has daughters: {cell_lineage.get(original_parent_id, [])}")

        except Exception as e_break_lineage:
            logging.error(f"LINEAGE_EDIT (Break): Error: {e_break_lineage}\n{traceback.format_exc()}")
            QMessageBox.critical(win, "Break Error", f"An error occurred: {e_break_lineage}")
            if cell_undoredo.undo_stack: cell_undoredo.undo_stack.pop()
            cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)
            return

        # Debug logging for track breaking
        logging.info(f"LINEAGE_EDIT (Break): After break operation - Track {clicked_track_id} status:")
        if new_segment_id != -1:
            logging.info(f"  - New segment ID: {new_segment_id}")
            logging.info(f"  - New segment frames: {trj[trj['particle'] == new_segment_id]['frame'].tolist()}")
            # Safely get parent from trajectory
            new_segment_trj = trj[trj['particle'] == new_segment_id]
            new_segment_parent = new_segment_trj['parent_particle'].iloc[0] if not new_segment_trj.empty else 'N/A'
            logging.info(f"  - New segment parent in trj: {new_segment_parent}")
        logging.info(f"  - Original track frames: {trj[trj['particle'] == clicked_track_id]['frame'].tolist()}")
        # Safely get parent from trajectory
        original_track_trj = trj[trj['particle'] == clicked_track_id]
        original_track_parent = original_track_trj['parent_particle'].iloc[0] if not original_track_trj.empty else 'N/A'
        logging.info(f"  - Original track parent in trj: {original_track_parent}")
        logging.info(f"  - Updated ancestry: {ancestry}")
        logging.info(f"  - Updated cell_lineage: {cell_lineage}")
        
        # Check if the new segment is properly in the main_app_state
        if new_segment_id != -1:
            logging.info(f"  - New segment in cell_ids: {new_segment_id in main_app_state.get('cell_ids', [])}")
            logging.info(f"  - New segment in cell_visibility: {new_segment_id in main_app_state.get('cell_visibility', {})}")
            logging.info(f"  - New segment in track_states: {new_segment_id in main_app_state.get('track_states', {})}")
        
        ui_callbacks.handle_populate_cell_table(main_app_state, ui_elements, ui_callbacks)
        ui_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)
        
        # Force update the lineage tree with complete refresh
        lineage_tree_widget = ui_elements.get('lineage_tree_widget')
        if lineage_tree_widget:
            logging.info(f"LINEAGE_EDIT (Break): Updating lineage tree widget with new data")
            try:
                # Update the lineage tree with new data
                lineage_tree_widget.set_data(trj, main_app_state.get('color_list'), main_app_state.get('cell_color_idx'),
                                             main_app_state.get('cell_visibility'), main_app_state.get('track_states'),
                                             ancestry, main_app_state.get('params'))
                lineage_tree_widget.draw_all_lineage_trees()
                # Force immediate repaint for instant visual feedback
                lineage_tree_widget.plot_item.update()
                # Process events to ensure immediate UI update
                QApplication.processEvents()
                logging.info(f"LINEAGE_EDIT (Break): Lineage tree widget update completed")
            except Exception as e:
                logging.error(f"LINEAGE_EDIT (Break): Error updating lineage tree: {e}")
        else:
            logging.warning(f"LINEAGE_EDIT (Break): No lineage tree widget found in UI elements")
            
        QMessageBox.information(win, "Lineage Broken",
                                f"Track {clicked_track_id} split. New segment (if created): {new_segment_id if new_segment_id != -1 else 'None'}.")


def process_split_and_fuse_head_request(original_track_id, break_frame_index, target_fusion_id, main_app_state,
                                        ui_elements, ui_callbacks, ui_actions_module):
    logging.info(
        f"LINEAGE_EDIT (Split&Fuse Head): Original {original_track_id}, Break {break_frame_index}, Target {target_fusion_id}")
    QMessageBox.information(ui_elements.get('win'), "Split & Fuse Head",
                            "This specific operation needs review/implementation based on desired workflow.")


def process_insert_dragged_and_split_target(dragged_id, target_id, insert_frame_on_target,
                                            main_app_state, ui_elements, ui_callbacks,
                                            ui_actions_module):
    win = ui_elements.get('win')
    trj = main_app_state.get('trj')
    logging.info(
        f"LINEAGE_EDIT (Insert&Split): Dragged {dragged_id} onto Target {target_id} at frame {insert_frame_on_target}.")

    if trj is None or trj.empty:
        QMessageBox.warning(win, "Operation Error", "Trajectory data is empty.")
        return
    if dragged_id == target_id:
        QMessageBox.warning(win, "Operation Error", "Cannot drag a track onto itself for this operation.")
        return

    target_track_frames = trj[trj['particle'] == target_id]['frame']
    if target_track_frames.empty:
        QMessageBox.warning(win, "Operation Error", f"Target track {target_id} not found in trajectory.")
        return
    min_target_frame = target_track_frames.min()
    max_target_frame = target_track_frames.max()

    if not (min_target_frame <= insert_frame_on_target <= max_target_frame):
        QMessageBox.warning(win, "Operation Error",
                            f"Insert frame {insert_frame_on_target} is outside target track {target_id}'s range [{min_target_frame}-{max_target_frame}].")
        return

    if insert_frame_on_target == min_target_frame:
        QMessageBox.warning(win, "Operation Error",
                            f"Cannot insert and split at the very first frame ({insert_frame_on_target}) of target track {target_id}. "
                            f"This would leave no 'head' segment for the target. "
                            f"Consider using 'Set Dragged as Parent of Target' or breaking the target track first.")
        return

    if _is_descendant_of_any(target_id, [dragged_id], main_app_state['ancestry'], main_app_state['cell_lineage']):
        QMessageBox.warning(win, "Cycle Error",
                            f"Cannot perform operation: Target track {target_id} is a descendant of dragged track {dragged_id}.")
        return
    if _is_descendant_of_any(dragged_id, [target_id], main_app_state['ancestry'], main_app_state['cell_lineage']):
        QMessageBox.warning(win, "Cycle Error",
                            f"Cannot perform operation: Dragged track {dragged_id} is a descendant of target track {target_id}.")
        return

    confirm_msg = (f"Insert dragged track {dragged_id} as a child of target track {target_id} "
                   f"(up to frame {insert_frame_on_target - 1}), and split target track {target_id} at frame {insert_frame_on_target}?\n"
                   f"The tail of target {target_id} (from frame {insert_frame_on_target}) will become a new track, "
                   f"parented by both the original target {target_id} (head part) and dragged track {dragged_id}.")
    if QMessageBox.question(win, 'Confirm Insert & Split', confirm_msg, QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.No) == QMessageBox.No:
        return

    cell_undoredo.push_state_for_undo(main_app_state, ui_elements)

    cell_lineage = main_app_state.get('cell_lineage')
    ancestry = main_app_state.get('ancestry')
    id_masks = main_app_state.get('id_masks')
    id_masks_initial = main_app_state.get('id_masks_initial')

    with pg.BusyCursor():
        try:
            if main_app_state.get('next_available_daughter_id') is None or \
                    main_app_state.get('next_available_daughter_id') <= (
            trj['particle'].max() if not trj.empty and trj['particle'].notna().any() else 0):
                initialize_next_daughter_id(trj)

            new_tail_id = get_new_daughter_id()
            logging.info(f"LINEAGE_EDIT (Insert&Split): New tail ID for target {target_id} will be {new_tail_id}.")

            # --- Store original relationships before modification ---
            original_target_children = cell_lineage.get(target_id, []).copy()
            original_dragged_children = cell_lineage.get(dragged_id, []).copy()
            original_parents_of_dragged = ancestry.get(dragged_id, []).copy()

            # --- 1. Update Trajectory (trj) ---
            target_tail_mask_trj = (trj['particle'] == target_id) & (trj['frame'] >= insert_frame_on_target)
            trj.loc[target_tail_mask_trj, 'particle'] = new_tail_id
            trj.loc[trj['particle'] == new_tail_id, 'parent_particle'] = target_id  # Primary parent for trj

            # Clean dragged_id's old parent_particle entries in trj before setting new one
            trj.loc[trj['particle'] == dragged_id, 'parent_particle'] = pd.NA  # Clear first
            trj.loc[trj['particle'] == dragged_id, 'parent_particle'] = target_id  # Set new parent

            # --- 2. Update Lineage Maps (cell_lineage, ancestry) ---

            # A. Handle dragged_id:
            #   A1. Remove dragged_id from the lineage of ALL its original parents.
            for old_p_of_dragged in original_parents_of_dragged:
                if old_p_of_dragged in cell_lineage and dragged_id in cell_lineage[old_p_of_dragged]:
                    cell_lineage[old_p_of_dragged].remove(dragged_id)
                    if not cell_lineage[old_p_of_dragged]: del cell_lineage[old_p_of_dragged]
            #   A2. Set ancestry for dragged_id: its SOLE parent is now target_id (head part).
            ancestry[dragged_id] = [target_id]
            #   A3. Update cell_lineage for dragged_id: its children are its original children (do NOT add new_tail_id).
            dragged_children_set = set(original_dragged_children)
            # dragged_children_set.add(new_tail_id)  # REMOVE this line to avoid making new tail a daughter of dragged
            cell_lineage[dragged_id] = sorted(list(dragged_children_set))

            # B. Handle target_id (head part):
            children_to_move_to_new_tail = []
            children_to_keep_with_target_head = []
            for child_ot_id in original_target_children:
                if child_ot_id == dragged_id or child_ot_id == new_tail_id: continue
                child_trj_frames = trj[trj['particle'] == child_ot_id]
                if child_trj_frames.empty: continue
                child_start_frame_ot = child_trj_frames['frame'].min()
                if child_start_frame_ot >= insert_frame_on_target:
                    children_to_move_to_new_tail.append(child_ot_id)
                else:
                    children_to_keep_with_target_head.append(child_ot_id)

            target_head_children_set = set(children_to_keep_with_target_head)
            target_head_children_set.add(dragged_id)
            target_head_children_set.add(new_tail_id)
            cell_lineage[target_id] = sorted(list(target_head_children_set))

            # C. Handle new_tail_id:
            if children_to_move_to_new_tail:
                cell_lineage[new_tail_id] = sorted(children_to_move_to_new_tail)
            elif new_tail_id in cell_lineage:
                del cell_lineage[new_tail_id]

            # ancestry[new_tail_id] = sorted(list(set([target_id, dragged_id])))  # REMOVE dragged_id as parent
            ancestry[new_tail_id] = [target_id]  # Only target_id is parent

            for child_moved_id in children_to_move_to_new_tail:
                ancestry[child_moved_id] = [new_tail_id]
                trj.loc[trj['particle'] == child_moved_id, 'parent_particle'] = new_tail_id

            # --- 3. Update ID Masks ---
            if id_masks is not None and id_masks_initial is not None:
                for frame_idx_m in trj.loc[trj['particle'] == new_tail_id, 'frame'].unique():
                    if 0 <= frame_idx_m < id_masks.shape[2]:
                        original_target_pixels_this_frame = (id_masks_initial[:, :, frame_idx_m] == target_id)
                        id_masks[:, :, frame_idx_m][original_target_pixels_this_frame] = new_tail_id
                        id_masks_initial[:, :, frame_idx_m][original_target_pixels_this_frame] = new_tail_id

            # --- 4. Update State Dictionaries for new_tail_id ---
            # (This part seems mostly correct from v5)
            cell_ids_list_ref_split = main_app_state.get('cell_ids', [])
            if isinstance(cell_ids_list_ref_split, list) and new_tail_id not in cell_ids_list_ref_split:
                cell_ids_list_ref_split.append(new_tail_id)
            elif isinstance(cell_ids_list_ref_split, set):
                cell_ids_list_ref_split.add(new_tail_id)

            cell_color_idx_split = main_app_state.get('cell_color_idx')
            color_list_split = main_app_state.get('color_list')
            original_target_color_idx = cell_color_idx_split.get(target_id)
            cell_color_idx_split[new_tail_id] = original_target_color_idx if original_target_color_idx is not None else \
                (len(cell_color_idx_split) % len(color_list_split) if color_list_split and len(
                    color_list_split) > 0 else 0)

            main_app_state.get('cell_visibility')[new_tail_id] = main_app_state.get('cell_visibility').get(target_id,
                                                                                                           True)
            main_app_state.get('track_states')[new_tail_id] = main_app_state.get('track_states').get(target_id, "N/A")

            cell_frame_presence_split = main_app_state.get('cell_frame_presence')
            original_target_frames_cfp = cell_frame_presence_split.get(target_id,
                                                                       set()).copy()  # Use a different var name
            cell_frame_presence_split[new_tail_id] = {f for f in original_target_frames_cfp if
                                                      f >= insert_frame_on_target}
            cell_frame_presence_split[target_id] = {f for f in original_target_frames_cfp if f < insert_frame_on_target}
            if not cell_frame_presence_split[target_id]:
                if target_id in cell_frame_presence_split: del cell_frame_presence_split[target_id]

            cell_x_split = main_app_state.get('cell_x');
            cell_y_split = main_app_state.get('cell_y')
            for frame_val_splt in cell_frame_presence_split.get(new_tail_id, set()):
                if (target_id, frame_val_splt) in cell_x_split: cell_x_split[
                    new_tail_id, frame_val_splt] = cell_x_split.pop((target_id, frame_val_splt))
                if (target_id, frame_val_splt) in cell_y_split: cell_y_split[
                    new_tail_id, frame_val_splt] = cell_y_split.pop((target_id, frame_val_splt))

            main_app_state.get('cell_visibility')[dragged_id] = True

            main_app_state['has_lineage'] = bool(cell_lineage or ancestry)

        except Exception as e_ins_split:
            logging.error(f"LINEAGE_EDIT (Insert&Split): Error: {e_ins_split}\n{traceback.format_exc()}")
            QMessageBox.critical(win, "Insert & Split Error", f"An error occurred: {e_ins_split}")
            if cell_undoredo.undo_stack: cell_undoredo.undo_stack.pop()
            cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)
            return

        ui_callbacks.handle_populate_cell_table(main_app_state, ui_elements)
        ui_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)
        lineage_tree_widget = ui_elements.get('lineage_tree_widget')
        if lineage_tree_widget:
            # Optimize the update for better responsiveness
            lineage_tree_widget.set_data(trj, main_app_state.get('color_list'), main_app_state.get('cell_color_idx'),
                                         main_app_state.get('cell_visibility'), main_app_state.get('track_states'),
                                         ancestry)
            # Use a more efficient update approach
            lineage_tree_widget.draw_all_lineage_trees()
            # Force immediate repaint for instant visual feedback
            lineage_tree_widget.plot_item.update()
            # Process events to ensure immediate UI update
            QApplication.processEvents()

        QMessageBox.information(win, "Insert & Split Successful",
                                f"Track {dragged_id} inserted. Target {target_id} split, new tail is {new_tail_id}.")


def handle_relink_visible_tracks(main_app_state, ui_elements, ui_callbacks, ui_actions_module):
    win = ui_elements.get('win')
    trj = main_app_state.get('trj')
    cell_visibility = main_app_state.get('cell_visibility')
    cell_lineage = main_app_state.get('cell_lineage')
    ancestry = main_app_state.get('ancestry')

    if trj is None or trj.empty or not cell_visibility or not cell_lineage or not ancestry:
        QMessageBox.warning(win, "Relink Error", "Not enough data to perform relinking.")
        return
    if QMessageBox.question(win, 'Confirm Relink', "This will re-parent daughters of hidden tracks. Proceed?",
                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.No:
        return

    cell_undoredo.push_state_for_undo(main_app_state, ui_elements)
    modified_lineage = False

    with pg.BusyCursor():
        try:
            hidden_track_ids = [track_id for track_id, is_visible in cell_visibility.items() if not is_visible]
            for hidden_id in hidden_track_ids:
                original_daughters_of_hidden = list(cell_lineage.get(hidden_id, []))
                parents_of_hidden = list(ancestry.get(hidden_id, []))
                if not original_daughters_of_hidden:
                    if hidden_id in ancestry: del ancestry[hidden_id]; modified_lineage = True
                    if 'parent_particle' in trj.columns and not trj[trj['particle'] == hidden_id].empty:
                        trj.loc[trj['particle'] == hidden_id, 'parent_particle'] = pd.NA;
                        modified_lineage = True
                    for p_of_h in parents_of_hidden:
                        if p_of_h in cell_lineage and hidden_id in cell_lineage[p_of_h]:
                            cell_lineage[p_of_h].remove(hidden_id)
                            if not cell_lineage[p_of_h]: del cell_lineage[p_of_h]
                            modified_lineage = True
                    continue

                grandparent_id = parents_of_hidden[0] if parents_of_hidden else None
                logging.info(
                    f"Relinking: Hidden {hidden_id}. Grandparent: {grandparent_id}. Daughters: {original_daughters_of_hidden}")

                for daughter_id in original_daughters_of_hidden:
                    if daughter_id in ancestry:
                        if hidden_id in ancestry[daughter_id]: ancestry[daughter_id].remove(hidden_id)
                        if grandparent_id is not None and grandparent_id not in ancestry[daughter_id]:
                            ancestry[daughter_id].append(grandparent_id)
                        if not ancestry[daughter_id]:
                            del ancestry[daughter_id]
                            trj.loc[trj['particle'] == daughter_id, 'parent_particle'] = pd.NA
                        else:
                            ancestry[daughter_id].sort()
                            trj.loc[trj['particle'] == daughter_id, 'parent_particle'] = ancestry[daughter_id][0]
                    elif grandparent_id is not None:
                        ancestry[daughter_id] = [grandparent_id]
                        ancestry[daughter_id].sort()
                        trj.loc[trj['particle'] == daughter_id, 'parent_particle'] = grandparent_id
                    else:
                        trj.loc[trj['particle'] == daughter_id, 'parent_particle'] = pd.NA

                    if grandparent_id is not None:
                        cell_lineage.setdefault(grandparent_id, [])
                        if daughter_id not in cell_lineage[grandparent_id]:
                            cell_lineage[grandparent_id].append(daughter_id)
                            cell_lineage[grandparent_id].sort()
                    modified_lineage = True

                if hidden_id in cell_lineage: del cell_lineage[hidden_id]
                if grandparent_id and grandparent_id in cell_lineage and hidden_id in cell_lineage[grandparent_id]:
                    cell_lineage[grandparent_id].remove(hidden_id)
                    if not cell_lineage[grandparent_id]: del cell_lineage[grandparent_id]
                if hidden_id in ancestry: del ancestry[hidden_id]
                if 'parent_particle' in trj.columns and not trj[trj['particle'] == hidden_id].empty:
                    trj.loc[trj['particle'] == hidden_id, 'parent_particle'] = pd.NA

            if modified_lineage:
                main_app_state['has_lineage'] = bool(cell_lineage or ancestry)
                initialize_next_daughter_id(trj)
                ui_callbacks.handle_populate_cell_table(main_app_state, ui_elements)
                ui_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)
                lineage_tree_widget = ui_elements.get('lineage_tree_widget')
                if lineage_tree_widget:
                    lineage_tree_widget.set_data(trj, main_app_state.get('color_list'),
                                                 main_app_state.get('cell_color_idx'),
                                                 main_app_state.get('cell_visibility'),
                                                 main_app_state.get('track_states'), ancestry)
                    lineage_tree_widget.draw_all_lineage_trees()
                    # Force immediate repaint for instant visual feedback
                    lineage_tree_widget.plot_item.update()
                    # Process events to ensure immediate UI update
                    QApplication.processEvents()
                QMessageBox.information(win, "Relink Successful", "Daughters of hidden tracks relinked.")
            else:
                QMessageBox.information(win, "Relink Info", "No modifications made during relink.")
                if cell_undoredo.undo_stack: cell_undoredo.undo_stack.pop()
                cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)
        except Exception as e_relink:
            logging.error(f"Error during relink operation: {e_relink}\n{traceback.format_exc()}")
            QMessageBox.critical(win, "Relink Error", f"An error occurred: {e_relink}")
            if cell_undoredo.undo_stack: cell_undoredo.undo_stack.pop()
            cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)
            return
