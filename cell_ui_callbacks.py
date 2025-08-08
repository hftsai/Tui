# cell_ui_callbacks.py
# Handles UI interactions and callbacks, keeping the main script cleaner.
# V29 (Gemini): Implemented the full, detailed statistics calculation logic as provided by the user.

import time
import logging
import traceback
import pandas as pd
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QTableWidgetItem, QCheckBox, QMessageBox, QApplication, QWidget, QHBoxLayout
from PyQt5.QtCore import Qt, QPointF
import json
import sys
import cell_undoredo
from cell_evaluation import run_ctc_evaluation_api
import os
from ctc_metrics.metrics import op_clb

# Performance optimization: Add debouncing for update calls
_last_update_time = 0
_update_debounce_ms = 100  # Increased debounce time to 100ms for smoother scrolling

# Add caching for expensive operations
_frame_cache = {}
_cache_max_size = 50  # Maximum number of cached frames

# Add track data cache
_track_data_cache = {}
_track_cache_max_size = 100  # Maximum number of cached track data entries


def _debounced_update(func):
    """Decorator to debounce update calls for better performance."""
    def wrapper(*args, **kwargs):
        global _last_update_time
        current_time = time.time() * 1000
        if current_time - _last_update_time > _update_debounce_ms:
            _last_update_time = current_time
            return func(*args, **kwargs)
    return wrapper


def _clear_frame_cache():
    """Clear the frame cache to free memory."""
    global _frame_cache
    _frame_cache.clear()


def _clear_track_cache():
    """Clear the track data cache to free memory."""
    global _track_data_cache
    _track_data_cache.clear()


def _clear_all_caches():
    """Clear all caches to free memory."""
    _clear_frame_cache()
    _clear_track_cache()


def _get_cached_frame_data(current_frame, trj, cell_visibility, background_id):
    """Get cached frame data or compute and cache it."""
    global _frame_cache
    
    cache_key = (current_frame, id(trj), id(cell_visibility), background_id)
    
    if cache_key in _frame_cache:
        return _frame_cache[cache_key]
    
    # Compute frame data
    trj_current_frame = trj[trj['frame'] == current_frame]
    trj_current_frame_particles = pd.to_numeric(trj_current_frame['particle'], errors='coerce').dropna().astype(int)
    
    # Only get visible cells that are in the current frame
    visible_cells_in_frame = []
    for cell_id in trj_current_frame_particles.values:
        if cell_id != background_id and cell_visibility.get(cell_id, True):
            visible_cells_in_frame.append(cell_id)
    
    # Cache the result
    if len(_frame_cache) >= _cache_max_size:
        # Remove oldest entry (simple FIFO)
        oldest_key = next(iter(_frame_cache))
        del _frame_cache[oldest_key]
    
    frame_data = {
        'trj_current_frame': trj_current_frame,
        'trj_current_frame_particles': trj_current_frame_particles,
        'visible_cells_in_frame': visible_cells_in_frame
    }
    _frame_cache[cache_key] = frame_data
    
    return frame_data


def _get_cached_track_data(cell_id, trj, current_frame):
    """Get cached track data or compute and cache it."""
    global _track_data_cache
    
    cache_key = (cell_id, id(trj), current_frame)
    
    if cache_key in _track_data_cache:
        return _track_data_cache[cache_key]
    
    # Compute track data
    track_data = trj[pd.to_numeric(trj['particle'], errors='coerce') == cell_id]
    track_data_past = track_data[track_data['frame'] <= current_frame]
    
    # Cache the result
    if len(_track_data_cache) >= _track_cache_max_size:
        # Remove oldest entry (simple FIFO)
        oldest_key = next(iter(_track_data_cache))
        del _track_data_cache[oldest_key]
    
    _track_data_cache[cache_key] = track_data_past
    
    return track_data_past


def _set_post_tracking_button_states(ui_elements, enabled, main_app_state=None):
    """Enable or disable all buttons that require tracking data, using stable keys."""
    stable_button_keys = [
        'select_all_visible', 'select_none_visible', 'select_complete_tracks',
        'select_mitosis_tracks', 'select_fusion_tracks', 'select_singular_tracks',
        'relink_visible_tracks', 'calculate_stats', 'save_results', 'evaluate_tracking_ctc',
        'merge_selected_track_with_parent'
    ]
    
    # Check if there's lineage data for mitosis/fusion buttons
    has_lineage_data = False
    if main_app_state:
        cell_lineage = main_app_state.get('cell_lineage', {})
        ancestry = main_app_state.get('ancestry', {})
        # Check if there are any mitosis events (parents with multiple daughters)
        has_mitosis = any(len(daughters) >= 2 for daughters in cell_lineage.values())
        # Check if there are any fusion events (children with multiple parents)
        has_fusion = any(len(parents) >= 2 for parents in ancestry.values())
        has_lineage_data = has_mitosis or has_fusion
        
        if enabled and not has_lineage_data:
            logging.info("Mitosis/Fusion track selection buttons disabled: No mitosis or fusion events detected in tracking data")
        elif enabled and has_lineage_data:
            mitosis_count = sum(1 for daughters in cell_lineage.values() if len(daughters) >= 2)
            fusion_count = sum(1 for parents in ancestry.values() if len(parents) >= 2)
            logging.info(f"Mitosis/Fusion track selection buttons enabled: {mitosis_count} mitosis events, {fusion_count} fusion events detected")
    
    for key in stable_button_keys:
        button = ui_elements.get('button_widgets_map', {}).get(key)
        if button:
            if key != 'merge_selected_track_with_parent':
                # Special handling for mitosis/fusion buttons - only enable if there's lineage data
                if key in ['select_mitosis_tracks', 'select_fusion_tracks', 'select_singular_tracks']:
                    button.setEnabled(enabled and has_lineage_data)
                else:
                    button.setEnabled(enabled)


def _update_ui_after_visibility_change(main_app_state, ui_elements, self_ref_for_callbacks):
    """Centralized UI update after cell visibility changes."""
    # Clear caches when visibility changes
    _clear_frame_cache()
    _clear_track_cache()
    
    self_ref_for_callbacks.handle_populate_cell_table(main_app_state, ui_elements, self_ref_for_callbacks)
    self_ref_for_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)
    lineage_tree_widget = ui_elements.get('lineage_tree_widget')
    if lineage_tree_widget:
        lineage_tree_widget.set_data(
            main_app_state.get('trj'),
            main_app_state.get('color_list'),
            main_app_state.get('cell_color_idx'),
            main_app_state.get('cell_visibility'),
            main_app_state.get('track_states'),
            main_app_state.get('ancestry'),
            main_app_state.get('params')
        )
        if lineage_tree_widget.current_root_id is not None:
            lineage_tree_widget.draw_lineage_tree_for_single_root(lineage_tree_widget.current_root_id)
        else:
            lineage_tree_widget.draw_all_lineage_trees()


def _update_ui_after_state_change(main_app_state, ui_elements, self_ref_for_callbacks):
    """Lightweight UI update after cell state changes only."""
    # Only update the view (no table repopulation or lineage tree redraw needed for state changes)
    self_ref_for_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)


def force_refresh_lineage_tree(main_app_state, ui_elements):
    """Force refresh the lineage tree to reflect current data."""
    lineage_tree_widget = ui_elements.get('lineage_tree_widget')
    if lineage_tree_widget:
        try:
            # Clear caches first
            _clear_all_caches()
            
            # Update lineage tree with current data
            lineage_tree_widget.set_data(
                main_app_state.get('trj'),
                main_app_state.get('color_list'),
                main_app_state.get('cell_color_idx'),
                main_app_state.get('cell_visibility'),
                main_app_state.get('track_states'),
                main_app_state.get('ancestry'),
                main_app_state.get('params')
            )
            
            # Redraw based on current view state
            if lineage_tree_widget.current_root_id is not None:
                lineage_tree_widget.draw_lineage_tree_for_single_root(lineage_tree_widget.current_root_id)
            else:
                lineage_tree_widget.draw_all_lineage_trees()
            
            # Force immediate repaint
            lineage_tree_widget.plot_item.update()
            
            # Safely call QApplication.processEvents() with proper import handling
            try:
                from PyQt5.QtWidgets import QApplication
                QApplication.processEvents()
            except ImportError:
                # Fallback: try to get QApplication instance
                try:
                    app = QApplication.instance()
                    if app:
                        app.processEvents()
                except:
                    pass  # Ignore if QApplication is not available
            
            logging.info("Lineage tree force refreshed successfully")
        except Exception as e:
            logging.error(f"Error force refreshing lineage tree: {e}")


def _connect_table_checkbox_signals(main_app_state, ui_elements, self_ref_for_callbacks):
    """Connects checkbox signals in the table to their handlers."""
    table = ui_elements.get('table_cell_selection')
    if table is None:
        return
    
    # Get the cell_undoredo module from the main app state or ui_elements
    cell_undoredo_module = None
    if 'cell_undoredo' in sys.modules:
        cell_undoredo_module = sys.modules['cell_undoredo']
    
    for row in range(table.rowCount()):
        cell_widget = table.cellWidget(row, 1)  # Column 1 is the checkbox column
        if cell_widget:
            # Find the checkbox in the widget
            for child in cell_widget.findChildren(QCheckBox):
                # Disconnect any existing connections to avoid duplicates
                try:
                    child.stateChanged.disconnect()
                except:
                    pass  # No connections to disconnect
                
                # Connect the signal
                child.stateChanged.connect(
                    lambda state, r=row, c=1: handle_cell_table_checkbox_state_changed(
                        state, r, c, main_app_state, ui_elements, self_ref_for_callbacks, cell_undoredo_module
                    )
                )


def handle_time_changed_raw_img(index_val, time_val, main_app_state, ui_elements, self_ref_for_callbacks):
    """Handles time change in the raw image view."""
    pi_raw_img = ui_elements.get('pi_raw_img')
    v_mask = ui_elements.get('v_mask')
    pi_mask = ui_elements.get('pi_mask')

    if pi_raw_img: pi_raw_img.setTitle(f"Frame: {int(round(index_val))}")
    main_app_state['current_frame_index'] = int(round(index_val))
    if v_mask and main_app_state.get('id_masks') is not None and \
            0 <= main_app_state['current_frame_index'] < main_app_state['id_masks'].shape[2]:
        v_mask.setCurrentIndex(main_app_state['current_frame_index'])
        if pi_mask: pi_mask.setTitle(f"Frame: {main_app_state['current_frame_index']} (ID Mask)")
    elif v_mask and main_app_state.get('raw_masks') is not None and \
            0 <= main_app_state['current_frame_index'] < main_app_state['raw_masks'].shape[2]:
        v_mask.setCurrentIndex(main_app_state['current_frame_index'])
        if pi_mask: pi_mask.setTitle(f"Frame: {main_app_state['current_frame_index']} (Raw Mask - Generic)")
    self_ref_for_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)


def handle_time_changed_mask(index_val, time_val, main_app_state, ui_elements, self_ref_for_callbacks):
    """Handles time change in the mask view."""
    pi_mask = ui_elements.get('pi_mask')
    v_raw_img = ui_elements.get('v_raw_img')
    pi_raw_img = ui_elements.get('pi_raw_img')

    if pi_mask:
        title_suffix = "(ID Mask)" if main_app_state.get('id_masks') is not None else "(Raw Mask - Generic)"
        pi_mask.setTitle(f"Frame: {int(round(index_val))} {title_suffix}")
    main_app_state['current_frame_index'] = int(round(index_val))
    if v_raw_img and main_app_state.get('raw_imgs') is not None and \
            0 <= main_app_state['current_frame_index'] < main_app_state['raw_imgs'].shape[2]:
        v_raw_img.setCurrentIndex(main_app_state['current_frame_index'])
        if pi_raw_img: pi_raw_img.setTitle(f"Frame: {main_app_state['current_frame_index']}")
    self_ref_for_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)


@_debounced_update
def handle_update_view_for_current_frame(main_app_state, ui_elements):
    """Updates visual elements based on the current frame."""
    current_frame = main_app_state.get('current_frame_index', 0)
    trj = main_app_state.get('trj')
    cell_visibility = main_app_state.get('cell_visibility', {})
    cell_color_idx = main_app_state.get('cell_color_idx', {})
    color_list = main_app_state.get('color_list')
    background_id = main_app_state.get('background_id', 0)
    cell_x = main_app_state.get('cell_x', {})
    cell_y = main_app_state.get('cell_y', {})
    show_ids = main_app_state.get('show_ids_current', True)
    show_tracks = main_app_state.get('show_tracks_current', True)
    cell_lineage = main_app_state.get('cell_lineage', {})
    show_mitosis_labels = main_app_state['params']['Show Mitosis Labels'][0]
    wide_track_cell_id = main_app_state.get('wide_track_cell_id')

    cell_ids_raw_img_items = main_app_state.get('cell_ids_raw_img', {})
    cell_ids_mask_items = main_app_state.get('cell_ids_mask', {})
    track_plots_per_cell_items = main_app_state.get('track_plots_per_cell', {})

    # Update lineage tree indicator (lightweight operation)
    lineage_tree_widget = ui_elements.get('lineage_tree_widget')
    if lineage_tree_widget:
        lineage_tree_widget.update_current_frame_indicator(current_frame)

    if trj is None or trj.empty or not color_list:
        # Hide all items efficiently
        for item_dict_vu in [cell_ids_raw_img_items, cell_ids_mask_items, track_plots_per_cell_items]:
            if isinstance(item_dict_vu, dict):
                for item_vu in item_dict_vu.values():
                    if hasattr(item_vu, 'hide'): item_vu.hide()
        return

    # Get cached frame data
    frame_data = _get_cached_frame_data(current_frame, trj, cell_visibility, background_id)
    trj_current_frame = frame_data['trj_current_frame']
    trj_current_frame_particles = frame_data['trj_current_frame_particles']
    visible_cells_in_frame = frame_data['visible_cells_in_frame']

    # Pre-compute coordinates for all visible cells in current frame
    cell_coords = {}
    for cell_id_val in visible_cells_in_frame:
        x_pos, y_pos = cell_x.get((cell_id_val, current_frame)), cell_y.get((cell_id_val, current_frame))
        if x_pos is not None and y_pos is not None:
            cell_coords[cell_id_val] = (x_pos, y_pos)

    # Process each visible cell
    for cell_id_val in visible_cells_in_frame:
        coords = cell_coords.get(cell_id_val)
        if coords is None:
            # Hide all items for this cell if no coordinates
            if cell_id_val in cell_ids_raw_img_items:
                cell_ids_raw_img_items[cell_id_val].hide()
            if cell_id_val in cell_ids_mask_items:
                cell_ids_mask_items[cell_id_val].hide()
            if cell_id_val in track_plots_per_cell_items:
                track_plots_per_cell_items[cell_id_val].hide()
            continue

        x_pos, y_pos = coords

        # Handle raw image labels
        if cell_id_val in cell_ids_raw_img_items:
            label_item_raw = cell_ids_raw_img_items[cell_id_val]
            if show_ids:
                label_item_raw.setPos(x_pos, y_pos)
                id_text = str(cell_id_val)
                if show_mitosis_labels:
                    # Optimize mitosis label generation - use cached cell data
                    cell_data = trj_current_frame[trj_current_frame_particles == cell_id_val]
                    parent_info_val = pd.NA
                    if not cell_data.empty and 'parent_particle' in cell_data.columns:
                        parent_info_val = cell_data['parent_particle'].iloc[0]
                    daughters_info = cell_lineage.get(cell_id_val, [])
                    if pd.notna(parent_info_val) and int(parent_info_val) != -1:
                        id_text = f"D{cell_id_val}[P{int(parent_info_val)}]"
                    elif daughters_info:
                        id_text = f"P{cell_id_val}({','.join(map(str, sorted(daughters_info)))})"
                label_item_raw.setText(id_text, color=color_list[cell_color_idx.get(cell_id_val, 0)])
                label_item_raw.show()
            else:
                label_item_raw.hide()

        # Handle mask labels
        if cell_id_val in cell_ids_mask_items:
            label_item_mask = cell_ids_mask_items[cell_id_val]
            if show_ids and main_app_state.get('id_masks') is not None:
                label_item_mask.setPos(x_pos, y_pos)
                label_item_mask.setText(str(cell_id_val), color=(255, 255, 255))
                label_item_mask.show()
            else:
                label_item_mask.hide()

        # Handle track plots
        if cell_id_val in track_plots_per_cell_items:
            track_plot_item = track_plots_per_cell_items[cell_id_val]
            if show_tracks:
                # Use cached track data
                track_data_past = _get_cached_track_data(cell_id_val, trj, current_frame)
                if not track_data_past.empty:
                    pen_width = 4 if cell_id_val == wide_track_cell_id else 2
                    track_plot_item.setData(x=track_data_past['x'].values, y=track_data_past['y'].values,
                                            pen=pg.mkPen(color=color_list[cell_color_idx.get(cell_id_val, 0)],
                                                         width=pen_width))
                    track_plot_item.show()
                else:
                    track_plot_item.hide()
            else:
                track_plot_item.hide()

    # Hide items for cells not in current frame (batch operation)
    all_cell_ids = set(cell_ids_raw_img_items.keys()) | set(cell_ids_mask_items.keys()) | set(track_plots_per_cell_items.keys())
    cells_to_hide = all_cell_ids - set(visible_cells_in_frame)
    
    for cell_id_val in cells_to_hide:
        if cell_id_val in cell_ids_raw_img_items:
            cell_ids_raw_img_items[cell_id_val].hide()
        if cell_id_val in cell_ids_mask_items:
            cell_ids_mask_items[cell_id_val].hide()
        if cell_id_val in track_plots_per_cell_items:
            track_plots_per_cell_items[cell_id_val].hide()


def handle_populate_cell_table(main_app_state, ui_elements, self_ref_for_callbacks=None):
    """
    Populates the cell information table.
    """
    table = ui_elements.get('table_cell_selection')
    if table is None: return

    trj = main_app_state.get('trj')
    has_data = trj is not None and not trj.empty
    _set_post_tracking_button_states(ui_elements, enabled=has_data, main_app_state=main_app_state)

    cell_visibility = main_app_state.get('cell_visibility', {})
    ancestry = main_app_state.get('ancestry', {})
    cell_lineage = main_app_state.get('cell_lineage', {})
    track_states = main_app_state.get('track_states', {})
    background_id = main_app_state.get('background_id', 0)
    
    # Preserve scroll position before repopulating
    scrollbar = table.verticalScrollBar()
    saved_scroll_position = scrollbar.value() if scrollbar else 0

    table.blockSignals(True)
    try:
        sort_col = table.horizontalHeader().sortIndicatorSection()
        sort_order = table.horizontalHeader().sortIndicatorOrder()
        table.setSortingEnabled(False)
        table.setRowCount(0)

        if not has_data:
            return

        # Optimize data processing - avoid unnecessary copying
        trj_copy = trj.copy()
        trj_copy['particle'] = pd.to_numeric(trj_copy['particle'], errors='coerce')
        trj_copy.dropna(subset=['particle'], inplace=True)
        trj_copy['particle'] = trj_copy['particle'].astype(int)

        unique_cell_ids = sorted([cid for cid in trj_copy['particle'].unique() if cid != background_id])
        logging.debug(f"Cell table population: Found {len(unique_cell_ids)} unique cell IDs")

        for cell_id_val in unique_cell_ids:
            row_position = table.rowCount()
            table.insertRow(row_position)

            id_item = QTableWidgetItem(str(cell_id_val))
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row_position, 0, id_item)

            chk_box_item_widget = QWidget()
            chk_box_layout = QHBoxLayout(chk_box_item_widget)
            chk_box = QCheckBox()
            chk_box.setChecked(cell_visibility.get(cell_id_val, True))
            chk_box.setProperty("cell_id", cell_id_val)
            chk_box_layout.addWidget(chk_box)
            chk_box_layout.setAlignment(Qt.AlignCenter)
            chk_box_layout.setContentsMargins(0, 0, 0, 0)
            table.setCellWidget(row_position, 1, chk_box_item_widget)

            parents_list = ancestry.get(cell_id_val, [])
            parents_str = ", ".join(map(str, sorted(list(set(parents_list)))))
            parent_item = QTableWidgetItem(parents_str)
            table.setItem(row_position, 2, parent_item)
            
            daughters_list = cell_lineage.get(cell_id_val, [])
            daughters_str = ", ".join(map(str, sorted(list(set(daughters_list)))))
            daughter_item = QTableWidgetItem(daughters_str)
            table.setItem(row_position, 3, daughter_item)

            state_str = track_states.get(cell_id_val, "N/A")
            state_item = QTableWidgetItem(state_str)
            # Make state column editable for manual annotation
            state_item.setToolTip("Click to edit cell state. Common states: 'cell', 'mitosis', 'fusion', 'unknown'")
            table.setItem(row_position, 4, state_item)

            original_label_val_str = "N/A"
            if 'original_mask_label' in trj_copy.columns:
                track_specific_data = trj_copy[trj_copy['particle'] == cell_id_val]
                if not track_specific_data.empty:
                    unique_orig_labels = pd.to_numeric(track_specific_data['original_mask_label'],
                                                       errors='coerce').dropna().unique()
                    if len(unique_orig_labels) > 0:
                        original_label_val_str = ", ".join(map(str, sorted(unique_orig_labels.astype(int))))
            original_label_item = QTableWidgetItem(original_label_val_str)
            original_label_item.setFlags(original_label_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row_position, 5, original_label_item)

        table.setSortingEnabled(True)
        if sort_col >= 0:
            table.sortByColumn(sort_col, sort_order)
    finally:
        table.blockSignals(False)
        
    # Restore scroll position after repopulating
    if scrollbar:
        scrollbar.setValue(saved_scroll_position)
        
    # Connect checkbox signals after table is populated
    if self_ref_for_callbacks:
        _connect_table_checkbox_signals(main_app_state, ui_elements, self_ref_for_callbacks)


def _parse_id_string(id_str):
    """Parses a comma-separated string of IDs into a list of unique integers."""
    if not id_str.strip(): return []
    try:
        parsed_ids = [int(x.strip()) for x in id_str.split(',') if x.strip().isdigit()]
        return sorted(list(set(p for p in parsed_ids if p > 0)))
    except ValueError:
        return None


def handle_cell_table_item_changed(item, main_app_state, ui_elements, self_ref_for_callbacks, cell_undoredo_module):
    """Handle edits to the cell table, specifically for lineage."""
    table = ui_elements.get('table_cell_selection')
    if table is None or item is None: return

    row, column = item.row(), item.column()
    if column not in [2, 3, 4]: return

    cell_id_item = table.item(row, 0)
    if not cell_id_item: return

    try:
        current_cell_id = int(cell_id_item.text())
        new_id_str = item.text()
        new_ids = _parse_id_string(new_id_str)

        if new_ids is None:
            raise ValueError("Invalid ID format.")

        # Only push undo state for lineage changes (not state changes)
        if column in [2, 3]:
            cell_undoredo_module.push_state_for_undo(main_app_state, ui_elements)

        if column == 2:  # Parent ID changed
            ancestry = main_app_state['ancestry']
            cell_lineage = main_app_state['cell_lineage']
            old_parents = set(ancestry.get(current_cell_id, []))
            new_parents = set(new_ids)
            for p in old_parents - new_parents:
                if p in cell_lineage and current_cell_id in cell_lineage[p]:
                    cell_lineage[p].remove(current_cell_id)
            for p in new_parents - old_parents:
                cell_lineage.setdefault(p, []).append(current_cell_id)
            ancestry[current_cell_id] = new_ids
            # Full UI update needed for lineage changes
            _update_ui_after_visibility_change(main_app_state, ui_elements, self_ref_for_callbacks)
        elif column == 3:  # Daughter ID changed
            cell_lineage = main_app_state['cell_lineage']
            ancestry = main_app_state['ancestry']
            old_daughters = set(cell_lineage.get(current_cell_id, []))
            new_daughters = set(new_ids)
            for d in old_daughters - new_daughters:
                if d in ancestry and current_cell_id in ancestry[d]:
                    ancestry[d].remove(current_cell_id)
            for d in new_daughters - old_daughters:
                ancestry.setdefault(d, []).append(current_cell_id)
            cell_lineage[current_cell_id] = new_ids
            # Full UI update needed for lineage changes
            _update_ui_after_visibility_change(main_app_state, ui_elements, self_ref_for_callbacks)
            
        elif column == 4:  # State changed - lightweight update
            new_state = item.text().strip()
            if new_state == "":
                new_state = "unknown"
            
            # Update track_states in main_app_state
            track_states = main_app_state.get('track_states', {})
            track_states[current_cell_id] = new_state
            main_app_state['track_states'] = track_states
            
            # Update trajectory data if it exists (optimized)
            trj = main_app_state.get('trj')
            if trj is not None and not trj.empty:
                if 'state' not in trj.columns:
                    # Create state column if it doesn't exist
                    trj['state'] = trj['particle'].map(track_states).fillna("unknown")
                    main_app_state['trj'] = trj
                    logging.debug(f"Created state column and updated cell {current_cell_id} to '{new_state}'")
                else:
                    # Update only the specific rows for this cell_id (more efficient)
                    mask = trj['particle'] == current_cell_id
                    if mask.any():
                        trj.loc[mask, 'state'] = new_state
                        main_app_state['trj'] = trj
                        logging.debug(f"Updated state for cell {current_cell_id} to '{new_state}' in trajectory data")
            
            # Update lineage tree when state changes to reflect new classifications
            _update_ui_after_visibility_change(main_app_state, ui_elements, self_ref_for_callbacks)
            # No undo state needed for simple state changes
            return  # Early return to avoid the heavy update below
        
        # Update undo/redo state for lineage changes
        cell_undoredo_module.update_undo_redo_actions_enabled_state(ui_elements)

    except Exception as e:
        logging.error(f"Error updating lineage from table: {e}")
        self_ref_for_callbacks.handle_populate_cell_table(main_app_state, ui_elements, self_ref_for_callbacks)


def handle_cell_table_checkbox_state_changed(state, row, col, main_app_state, ui_elements, self_ref_for_callbacks,
                                             cell_undoredo_module):
    """Handles state changes for checkboxes in the table."""
    table = ui_elements.get('table_cell_selection')
    if table is None or col != 1: return

    cell_id_item = table.item(row, 0)
    if not cell_id_item: return

    cell_id = int(cell_id_item.text())
    is_checked = (state == Qt.Checked)

    if main_app_state['cell_visibility'].get(cell_id) != is_checked:
        cell_undoredo_module.push_state_for_undo(main_app_state, ui_elements)
        main_app_state['cell_visibility'][cell_id] = is_checked
        _update_ui_after_visibility_change(main_app_state, ui_elements, self_ref_for_callbacks)
        cell_undoredo_module.update_undo_redo_actions_enabled_state(ui_elements)


def handle_cell_table_cell_clicked(row, column, main_app_state, ui_elements, self_ref_for_callbacks):
    """Handles clicks on cells in the table."""
    table = ui_elements.get('table_cell_selection')
    if table is None: return

    cell_id_item = table.item(row, 0)
    if cell_id_item:
        clicked_cell_id = int(cell_id_item.text())
        main_app_state['wide_track_cell_id'] = clicked_cell_id

        b_merge_track = ui_elements.get('button_widgets_map', {}).get('merge_selected_track_with_parent')
        if b_merge_track:
            ancestry = main_app_state.get('ancestry', {})
            can_merge = clicked_cell_id in ancestry and ancestry[clicked_cell_id]
            b_merge_track.setEnabled(bool(can_merge))

        lineage_tree_widget = ui_elements.get('lineage_tree_widget')
        if lineage_tree_widget:
            lineage_tree_widget.set_highlighted_track(clicked_cell_id)

        self_ref_for_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)


def handle_lineage_node_clicked_in_main(track_id, frame_index, main_app_state, ui_elements, self_ref_for_callbacks):
    """Handles clicks on nodes in the lineage tree view."""
    main_app_state['wide_track_cell_id'] = track_id
    main_app_state['current_frame_index'] = frame_index

    v_raw_img = ui_elements.get('v_raw_img')
    if v_raw_img: v_raw_img.setCurrentIndex(frame_index)

    table = ui_elements.get('table_cell_selection')
    if table:
        for r in range(table.rowCount()):
            id_item = table.item(r, 0)
            if id_item and int(id_item.text()) == track_id:
                table.selectRow(r)
                break

    self_ref_for_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)


def setup_hover_detection(plot_item_raw, plot_item_mask, main_app_state, ui_elements):
    """Sets up hover detection to display information on image views."""
    main_app_state['hover_label_raw'] = pg.TextItem(anchor=(0, 1))
    plot_item_raw.addItem(main_app_state['hover_label_raw'], ignoreBounds=True)
    main_app_state['hover_label_raw'].hide()

    main_app_state['hover_label_mask'] = pg.TextItem(anchor=(0, 1))
    plot_item_mask.addItem(main_app_state['hover_label_mask'], ignoreBounds=True)
    main_app_state['hover_label_mask'].hide()

    def mouse_moved(plot_widget_item, hover_label_item, event_pos):
        if not plot_widget_item.sceneBoundingRect().contains(event_pos):
            hover_label_item.hide()
            return

        mouse_point = plot_widget_item.vb.mapSceneToView(event_pos)
        mx, my = mouse_point.x(), mouse_point.y()
        my_int, mx_int = int(round(my)), int(round(mx))

        current_frame = main_app_state.get('current_frame_index', 0)
        raw_imgs_data = main_app_state.get('raw_imgs')
        id_masks_data = main_app_state.get('id_masks')
        raw_masks_data = main_app_state.get('raw_masks')

        raw_img_value_str, initial_seg_label_str, tracked_id_mask_value_str = "N/A", "N/A", "N/A"

        if raw_imgs_data is not None and 0 <= my_int < raw_imgs_data.shape[0] and 0 <= mx_int < raw_imgs_data.shape[
            1] and 0 <= current_frame < raw_imgs_data.shape[2]:
            val = raw_imgs_data[my_int, mx_int, current_frame]
            raw_img_value_str = f"({','.join(map(str, val))})" if isinstance(val, np.ndarray) else str(val)
        if raw_masks_data is not None and 0 <= my_int < raw_masks_data.shape[0] and 0 <= mx_int < raw_masks_data.shape[
            1] and 0 <= current_frame < raw_masks_data.shape[2]:
            initial_seg_label_str = str(raw_masks_data[my_int, mx_int, current_frame])
        if id_masks_data is not None and 0 <= my_int < id_masks_data.shape[0] and 0 <= mx_int < id_masks_data.shape[
            1] and 0 <= current_frame < id_masks_data.shape[2]:
            tracked_id_mask_value_str = str(id_masks_data[my_int, mx_int, current_frame])

        hover_text_val = f"XY: ({mx:.1f}, {my:.1f})\nRaw Img: {raw_img_value_str}\nInitial Seg: {initial_seg_label_str}\nTracked ID: {tracked_id_mask_value_str}"
        hover_label_item.setHtml(
            f"<div style='background-color:rgba(0,0,0,0.7); color:white; padding:3px; border-radius:3px;'>{hover_text_val.replace(chr(10), '<br/>')}</div>")
        hover_label_item.setPos(mx, my)
        hover_label_item.show()

    plot_item_raw.scene().sigMouseMoved.connect(
        lambda pos: mouse_moved(plot_item_raw, main_app_state['hover_label_raw'], pos))
    plot_item_mask.scene().sigMouseMoved.connect(
        lambda pos: mouse_moved(plot_item_mask, main_app_state['hover_label_mask'], pos))


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

    # Calculate observed doubling time for cells with complete lifecycle (mitosis to mitosis, fusion, or disappearance)
    observed_lifecycle_durations_hours = []
    debug_info = []  # For debugging
    if cell_lineage:
        logging.info(f"Debug: cell_lineage has {len(cell_lineage)} entries")
        for parent_id_str, daughter_ids_list in cell_lineage.items():
            try:
                parent_id = int(parent_id_str)
                if len(daughter_ids_list) >= 2:  # Parent has at least 2 daughters (mitosis event)
                    logging.info(f"Debug: Parent {parent_id} has {len(daughter_ids_list)} daughters: {daughter_ids_list}")
                    for daughter_id in daughter_ids_list:
                        try:
                            daughter_id_int = int(daughter_id)
                            daughter_key_str = str(daughter_id_int)
                            daughter_key_int = daughter_id_int
                            in_cell_lineage_str = daughter_key_str in cell_lineage
                            in_cell_lineage_int = daughter_key_int in cell_lineage
                            # 1. Divides (has 2 daughters)
                            divides = (in_cell_lineage_str and len(cell_lineage[daughter_key_str]) >= 2) or \
                                      (in_cell_lineage_int and len(cell_lineage[daughter_key_int]) >= 2)
                            # 2. Fuses (is a child in ancestry with >=2 parents)
                            fuses = False
                            if daughter_id_int in ancestry and len(ancestry[daughter_id_int]) >= 2:
                                fuses = True
                            elif daughter_key_str in ancestry and len(ancestry[daughter_key_str]) >= 2:
                                fuses = True
                            # 3. Disappears (track ends before last frame)
                            daughter_track_data = trj_stats[trj_stats['particle'] == daughter_id_int]
                            disappears = False
                            if not daughter_track_data.empty and total_frames_sequence > 0:
                                last_frame = daughter_track_data['frame'].max()
                                if last_frame < (total_frames_sequence - 1):
                                    disappears = True
                            # If any of the above, count as complete lifecycle
                            if divides or fuses or disappears:
                                parent_track_data = trj_stats[trj_stats['particle'] == parent_id]
                                if not parent_track_data.empty and not daughter_track_data.empty:
                                    parent_division_frame = parent_track_data['frame'].max()
                                    daughter_end_frame = daughter_track_data['frame'].max()
                                    lifecycle_duration_frames = daughter_end_frame - parent_division_frame
                                    if lifecycle_duration_frames > 0:
                                        lifecycle_duration_hours = lifecycle_duration_frames / frames_per_hour
                                        observed_lifecycle_durations_hours.append(lifecycle_duration_hours)
                                        debug_info.append(f"Complete lifecycle: Parent {parent_id} -> Daughter {daughter_id_int}, Duration: {lifecycle_duration_hours:.2f} hours, Reason: {'divides' if divides else ('fuses' if fuses else 'disappears')}")
                                    else:
                                        logging.warning(f"Debug: Zero or negative lifecycle duration for Parent {parent_id} -> Daughter {daughter_id_int}")
                                else:
                                    logging.warning(f"Debug: Missing track data for Parent {parent_id} or Daughter {daughter_id_int}")
                            else:
                                logging.info(f"Debug: Daughter {daughter_id_int} not complete (no division, fusion, or disappearance)")
                        except ValueError as e:
                            logging.warning(f"Debug: ValueError converting daughter_id {daughter_id}: {e}")
                            continue
            except ValueError as e:
                logging.warning(f"Debug: ValueError converting parent_id {parent_id_str}: {e}")
                continue
    logging.info(f"Debug: Found {len(observed_lifecycle_durations_hours)} complete lifecycle observations (mitosis, fusion, or disappearance)")
    for info in debug_info:
        logging.info(f"Debug: {info}")
    if observed_lifecycle_durations_hours:
        stats["observed_doubling_time_avg_hours"] = np.mean(observed_lifecycle_durations_hours)
        stats["observed_doubling_time_median_hours"] = np.median(observed_lifecycle_durations_hours)
        stats["observed_doubling_time_std_hours"] = np.std(observed_lifecycle_durations_hours)
        stats["complete_lifecycle_observations"] = len(observed_lifecycle_durations_hours)
    else:
        stats["observed_doubling_time_avg_hours"] = "N/A"
        stats["observed_doubling_time_median_hours"] = "N/A"
        stats["observed_doubling_time_std_hours"] = "N/A"
        stats["complete_lifecycle_observations"] = 0

    stats["fusion_events (2 parents)"] = sum(1 for p in ancestry.values() if len(p) == 2)
    stats["fusion_events (>2 parents)"] = sum(1 for p in ancestry.values() if len(p) > 2)

    if total_frames_sequence > 0:
        stats["cell_count_at_first_frame"] = trj_stats[trj_stats['frame'] == 0]['particle'].nunique()
        stats["cell_count_at_last_frame"] = trj_stats[trj_stats['frame'] == (total_frames_sequence - 1)][
            'particle'].nunique()

    main_app_state['calculated_stats'] = stats
    return stats


def handle_calculate_stats_clicked(main_app_state, ui_elements):
    """Handles the 'Calculate Stats' button click."""
    win = ui_elements.get('win')
    stats = _calculate_comprehensive_stats(main_app_state)

    if "error" in stats:
        QMessageBox.information(win, "Statistics", stats["error"])
        return

    display_message = "Tracking Statistics:\n--------------------\n"
    for key, value in stats.items():
        # A simple formatter to make keys more readable
        formatted_key = key.replace('_', ' ').title()
        if isinstance(value, float):
            display_message += f"{formatted_key}: {value:.2f}\n"
        else:
            display_message += f"{formatted_key}: {value}\n"

    QMessageBox.information(win, "Comprehensive Track Statistics", display_message)
    logging.info("Comprehensive statistics calculated and displayed.")


def handle_select_all_clicked(main_app_state, ui_elements, self_ref_for_callbacks):
    cell_visibility = main_app_state.get('cell_visibility')
    if not cell_visibility: return
    cell_undoredo.push_state_for_undo(main_app_state, ui_elements)
    for k in cell_visibility.keys(): cell_visibility[k] = True
    _update_ui_after_visibility_change(main_app_state, ui_elements, self_ref_for_callbacks)
    cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)


def handle_select_none_clicked(main_app_state, ui_elements, self_ref_for_callbacks):
    cell_visibility = main_app_state.get('cell_visibility')
    if not cell_visibility: return
    cell_undoredo.push_state_for_undo(main_app_state, ui_elements)
    for k in cell_visibility.keys(): cell_visibility[k] = False
    _update_ui_after_visibility_change(main_app_state, ui_elements, self_ref_for_callbacks)
    cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)


def handle_select_complete_clicked(main_app_state, ui_elements, self_ref_for_callbacks):
    trj, raw_imgs, cell_visibility = main_app_state.get('trj'), main_app_state.get('raw_imgs'), main_app_state.get(
        'cell_visibility')
    if trj is None or trj.empty or raw_imgs is None or not cell_visibility: return
    cell_undoredo.push_state_for_undo(main_app_state, ui_elements)
    num_frames_total = raw_imgs.shape[2]
    track_lengths = trj.groupby('particle')['frame'].nunique()
    complete_track_ids = set(track_lengths[track_lengths == num_frames_total].index)
    for cid_key in list(cell_visibility.keys()):
        cell_visibility[cid_key] = (int(cid_key) in complete_track_ids)
    _update_ui_after_visibility_change(main_app_state, ui_elements, self_ref_for_callbacks)
    cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)


def handle_select_mitosis_tracks_clicked(main_app_state, ui_elements, self_ref_for_callbacks):
    trj, cell_visibility, cell_lineage = main_app_state.get('trj'), main_app_state.get(
        'cell_visibility'), main_app_state.get('cell_lineage', {})
    if trj is None or trj.empty or not cell_visibility: return
    cell_undoredo.push_state_for_undo(main_app_state, ui_elements)
    mitosis_related_ids = set()
    for parent_id, daughter_ids in cell_lineage.items():
        if len(daughter_ids) >= 2:
            mitosis_related_ids.add(int(parent_id))
            mitosis_related_ids.update([int(d_id) for d_id in daughter_ids])
    for cid_key in list(cell_visibility.keys()):
        cell_visibility[cid_key] = (int(cid_key) in mitosis_related_ids)
    _update_ui_after_visibility_change(main_app_state, ui_elements, self_ref_for_callbacks)
    cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)


def handle_select_fusion_tracks_clicked(main_app_state, ui_elements, self_ref_for_callbacks):
    trj, cell_visibility, ancestry = main_app_state.get('trj'), main_app_state.get(
        'cell_visibility'), main_app_state.get('ancestry', {})
    if trj is None or trj.empty or not cell_visibility: return
    cell_undoredo.push_state_for_undo(main_app_state, ui_elements)
    fusion_related_ids = set()
    for child_id, parent_ids in ancestry.items():
        if len(parent_ids) >= 2:
            fusion_related_ids.add(int(child_id))
            fusion_related_ids.update([int(p_id) for p_id in parent_ids])
    for cid_key in list(cell_visibility.keys()):
        cell_visibility[cid_key] = (int(cid_key) in fusion_related_ids)
    _update_ui_after_visibility_change(main_app_state, ui_elements, self_ref_for_callbacks)
    cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)


def handle_select_singular_tracks_clicked(main_app_state, ui_elements, self_ref_for_callbacks):
    trj, cell_visibility, cell_lineage, ancestry = main_app_state.get('trj'), main_app_state.get(
        'cell_visibility'), main_app_state.get('cell_lineage', {}), main_app_state.get('ancestry', {})
    if trj is None or trj.empty or not cell_visibility: return
    cell_undoredo.push_state_for_undo(main_app_state, ui_elements)
    mitosis_parents = {int(p_id) for p_id, d_ids in cell_lineage.items() if len(d_ids) >= 2}
    mitosis_daughters = {int(d_id) for d_ids in cell_lineage.values() if len(d_ids) >= 2 for d_id in d_ids}
    fusion_children = {int(c_id) for c_id, p_ids in ancestry.items() if len(p_ids) >= 2}
    fusion_parents = {int(p_id) for p_ids in ancestry.values() if len(p_ids) >= 2 for p_id in p_ids}
    event_related_ids = mitosis_parents.union(mitosis_daughters).union(fusion_children).union(fusion_parents)
    for cid_key in list(cell_visibility.keys()):
        cell_visibility[cid_key] = (int(cid_key) not in event_related_ids)
    _update_ui_after_visibility_change(main_app_state, ui_elements, self_ref_for_callbacks)
    cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)
