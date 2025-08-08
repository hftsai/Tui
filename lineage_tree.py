# lineage_tree.py
# Contains logic adapted from napari-arboretum for tree building and layout,
# and the PyQtGraphLineageWidget for displaying the lineage tree.
# V15 (Gemini & User Request): Added current frame indicator line.
# V16 (Gemini): Fixed double dialog issue on drag-drop by emitting chosen operation.
# V17 (Gemini): Added "Insert Dragged as Child & Split Target's Tail" to dialog and new signal.
# V18 (User Request): Shifted text label position to avoid overlapping lines.
# V19 (Gemini): Added diagnostic logging for future enhancements.
# V20 (Gemini & User Request): Enabled multi-root selection and display.

import itertools
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Optional
import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from pyqtgraph import exporters
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QListWidget, QLabel, QHBoxLayout, QLineEdit,
                             QRadioButton, QButtonGroup, QDialog, QDialogButtonBox, QComboBox, QInputDialog,
                             QFormLayout, QMessageBox, QAbstractItemView, QFileDialog, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QEvent
from PyQt5.QtGui import QMouseEvent, QFont
import pandas as pd
import logging
import os
import traceback


# --- Data classes ---
@dataclass
class TreeNode:
    ID: int
    t: np.ndarray  # Time points for this node
    generation: int
    children: List[int] = field(default_factory=list)
    parent_id: Optional[int] = None
    points: Optional[np.ndarray] = None  # Raw data points (frame, y, x) associated with this node's time points
    state: Optional[str] = None  # Cell state/class type

    @property
    def is_root(self) -> bool:
        """Is this node a root of a tree?"""
        return self.generation == 1

    @property
    def is_leaf(self) -> bool:
        """Is this node a leaf (no children)?"""
        return not self.children


ColorType = npt.ArrayLike  # Type hint for colors
CONNECTOR_EDGE_COLOR = (150, 150, 150)  # Color for lines connecting parent to child segments
NODE_VIEW_MITOSIS_LINE_COLOR = (255, 100, 100)  # Specific color for mitosis lines in Node View
DRAG_LINE_COLOR = (220, 220, 50)  # Color for the temporary line when dragging nodes
DEFAULT_STATE_COLOR = (100, 100, 100)  # Default color for states if not mapped
CURRENT_FRAME_INDICATOR_COLOR = (255, 255, 0, 150)  # Yellow, semi-transparent for the frame line


@dataclass
class Annotation:
    """Represents a text annotation on the plot."""
    x: float
    y: float
    label: str
    color: Tuple[int, int, int, int] = (255, 255, 255, 255)  # Default white
    html: Optional[str] = None  # For rich text


@dataclass
class Edge:
    """Represents a line segment in the 'Track Segments' view."""
    x: Tuple[float, float]  # (start_time, end_time)
    y: Tuple[float, float]  # (y_pos_start, y_pos_end) - usually same for track segments
    color: Tuple[int, int, int] = (200, 200, 200)
    track_id: Optional[int] = None
    node: Optional[TreeNode] = None  # Reference to the TreeNode this edge represents
    is_connector: bool = False  # True if this edge connects a parent to a child
    connected_child_id: Optional[int] = None  # If connector, the ID of the child it connects to


@dataclass
class NodePlotData:
    """Data for plotting nodes in 'Node View' (ScatterPlotItem points)."""
    points: np.ndarray  # Array of (time, y_layout_position)
    color: Tuple[int, int, int]
    track_id: int
    size: int = 5  # Default node size
    pg_scatter_item: Optional[pg.ScatterPlotItem] = None  # Reference to the actual plot item


@dataclass
class LinePlotData:
    """Data for plotting lines in 'Node View' (PlotDataItem lines)."""
    x_coords: List[float]
    y_coords: List[float]
    color: Tuple[int, int, int]
    is_mitosis_connector: bool = False  # True if this line connects parent to daughter nodes
    track_id: Optional[int] = None  # If it represents a track segment within a node


# --- Custom PlotItem for Interactive Mouse Events ---
class InteractivePlotItem(pg.PlotItem):
    def __init__(self, lineage_widget_ref, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lineage_widget_ref = lineage_widget_ref  # Reference to parent LineageTreeWidget

    def mousePressEvent(self, ev: QMouseEvent):
        self.lineage_widget_ref.customMousePressEvent(ev)
        if not ev.isAccepted():  # Allow base class handling if not accepted by custom logic
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent):
        self.lineage_widget_ref.customMouseMoveEvent(ev)
        if not ev.isAccepted():
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent):
        self.lineage_widget_ref.customMouseReleaseEvent(ev)
        if not ev.isAccepted():
            super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev: QMouseEvent):
        self.lineage_widget_ref.customMouseDoubleClickEvent(ev)
        if not ev.isAccepted():
            super().mouseDoubleClickEvent(ev)


# --- Dialog for Lineage Operations (used in LineageTreeWidget) ---
class LineageOperationDialog(QDialog):
    """Dialog for choosing lineage operation upon drag-and-drop."""

    def __init__(self, dragged_id, target_id, insert_frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Lineage Operation")
        layout = QVBoxLayout(self)
        self.label = QLabel(
            f"Dragged Node {dragged_id} onto Node {target_id} (near frame {insert_frame}).\nChoose operation:")
        layout.addWidget(self.label)
        self.operation_combo = QComboBox()
        # Define available operations
        self.operations = [
            "Cancel",
            "Join Track (Merge Dragged into Target)",
            "Set Target as Parent of Dragged",
            "Set Dragged as Parent of Target",
            "Set Dragged as Additional Parent of Target",
            "Insert Dragged as Child & Split Target's Tail"  # New fusion operation
        ]
        self.operation_combo.addItems(self.operations)
        layout.addWidget(self.operation_combo)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_selected_operation(self):
        return self.operation_combo.currentText()


# --- Dialog for Split/Fuse Operation (used for double-click) ---
class SplitFuseOperationDialog(QDialog):  # This dialog is for double-click, not drag-drop
    def __init__(self, track_id, frame_index, parent=None):
        super().__init__(parent)
        self.track_id = track_id
        self.frame_index = frame_index
        self.setWindowTitle(f"Lineage Edit Options (Track {track_id} at Frame {frame_index})")

        self.layout = QVBoxLayout(self)

        self.operation_label = QLabel("Choose operation:")
        self.layout.addWidget(self.operation_label)

        self.operation_combo = QComboBox()
        self.operation_combo.addItems([
            "Cancel",
            "Break into two tracks",
            # "Split: Fuse Head into Target, New Tail as Child" # This was an older concept
        ])
        self.layout.addWidget(self.operation_combo)
        # Set default selection to "Break into two tracks" (index 1)
        self.operation_combo.setCurrentIndex(1)
        # self.operation_combo.currentIndexChanged.connect(self._on_operation_changed) # Not needed if only "Break"

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

        self.selected_operation = "Break into two tracks"
        # self.target_fusion_id = None # Not needed for simple break

    def get_selected_operation_and_target(self):  # Simplified for break operation
        self.selected_operation = self.operation_combo.currentText()
        return self.selected_operation, None  # No target ID for simple break


# --- Graph processing functions ---
def build_reverse_graph(graph: Dict[int, List[int]]) -> Tuple[List[int], Dict[int, List[int]]]:
    """Converts an ancestry graph (child -> [parents]) to a lineage graph (parent -> [children]) and finds roots."""
    reverse_graph = defaultdict(list)
    all_nodes_in_input_graph = set(graph.keys())
    all_parent_references = set()

    for child, parents in graph.items():
        for parent in parents:
            reverse_graph[parent].append(child)
            all_parent_references.add(parent)
            all_nodes_in_input_graph.add(parent)

    potential_roots = all_parent_references - set(graph.keys())
    for child_node, parent_list in graph.items():
        if not parent_list or (len(parent_list) == 1 and parent_list[0] == child_node):
            potential_roots.add(child_node)
    roots = sorted(list(potential_roots))
    return roots, dict(reverse_graph)


def get_root_id(
        mock_layer_graph: Dict[int, List[int]],
        all_track_ids_from_data: List[int],
        search_node: int
) -> int:
    """Finds the ultimate root ancestor of a given search_node."""
    if search_node not in all_track_ids_from_data:
        logging.warning(f"[get_root_id] Search node {search_node} not in all_track_ids. Returning as is.")
        return search_node
    current_node = search_node
    visited_to_find_root = {current_node}
    max_depth = len(all_track_ids_from_data) + 5
    depth = 0
    while depth < max_depth:
        parents = mock_layer_graph.get(current_node)
        if not parents: return current_node
        parent = sorted(parents)[0]
        if parent == current_node:
            logging.warning(f"[get_root_id] Self-parenting detected for node {current_node}. Treating as root.")
            return current_node
        if parent in visited_to_find_root:
            logging.warning(
                f"[get_root_id] Cycle detected while finding root for {search_node}. Path: {visited_to_find_root}, current parent: {parent}. Returning current parent.")
            return parent
        current_node = parent
        visited_to_find_root.add(current_node)
        depth += 1
    logging.warning(
        f"[get_root_id] Max depth ({max_depth}) reached while finding root for {search_node}. Returning current node {current_node} as potential root.")
    return current_node


def build_subgraph(
        mock_layer_data: np.ndarray,
        ancestry_for_build: Dict[int, List[int]],
        lineage_for_build: Dict[int, List[int]],
        search_node_id: int,
        track_states_map: Optional[Dict[int, str]] = None
) -> List[TreeNode]:
    """Builds a list of TreeNode objects representing a single lineage tree starting from search_node_id."""
    all_data_track_ids = list(np.unique(mock_layer_data[:, 0]).astype(int))
    
    if search_node_id not in all_data_track_ids:
        logging.warning(f"[build_subgraph] Search node ID {search_node_id} not in data. Returning empty list.")
        return []
    
    root_id = get_root_id(ancestry_for_build, all_data_track_ids, search_node_id)
    
    nodes: List[TreeNode] = []
    queue: List[Tuple[int, int, Optional[int]]] = [(root_id, 1, None)]
    marked_ids_for_subgraph_nodes = set()
    processed_count = 0
    max_processed_nodes = len(all_data_track_ids) * 2
    while queue and processed_count < max_processed_nodes:
        current_id, generation, p_id_for_current_tree_structure = queue.pop(0)
        processed_count += 1
        if current_id in marked_ids_for_subgraph_nodes: continue
        marked_ids_for_subgraph_nodes.add(current_id)
        track_data_points_for_current_id = mock_layer_data[mock_layer_data[:, 0] == current_id]
        if track_data_points_for_current_id.shape[0] == 0:
            continue
        time_points_for_node = np.sort(np.unique(track_data_points_for_current_id[:, 1])).astype(float)
        node_points_data_raw = track_data_points_for_current_id[:, [1, 2, 3]]
        children_ids_of_current = lineage_for_build.get(current_id, [])
        valid_children = [child_id for child_id in children_ids_of_current if child_id != current_id]
        node_state_str = track_states_map.get(current_id, "Unclassified") if track_states_map else "Unclassified"
        node = TreeNode(ID=current_id, t=time_points_for_node, generation=generation,
                        children=valid_children, parent_id=p_id_for_current_tree_structure,
                        points=node_points_data_raw, state=node_state_str)
        nodes.append(node)
        for child_id_val in valid_children:
            if child_id_val not in marked_ids_for_subgraph_nodes:
                queue.append((child_id_val, generation + 1, current_id))
    if processed_count >= max_processed_nodes:
        logging.warning(
            f"[build_subgraph] Max processed nodes ({max_processed_nodes}) reached for root {root_id}. Subgraph might be incomplete.")
    nodes.sort(key=lambda n: (n.generation, n.ID))
    
    return nodes


def calculate_subtree_width(node_id: int, node_dict: Dict[int, TreeNode], memo: Dict[int, float] = None,
                            depth: int = 0) -> float:
    """Recursively calculates the 'width' needed to display a subtree for y-axis layout."""
    if memo is None: memo = {}
    if node_id in memo: return memo[node_id]
    MAX_RECURSION_DEPTH_LAYOUT = 1000
    if depth > MAX_RECURSION_DEPTH_LAYOUT:
        logging.error(
            f"Max recursion depth ({MAX_RECURSION_DEPTH_LAYOUT}) exceeded in calculate_subtree_width for node {node_id}.")
        return 1.0
    leaf_width = 5.0
    node = node_dict.get(node_id)
    if not node or not node.children:
        memo[node_id] = leaf_width
        return leaf_width
    children_widths = [calculate_subtree_width(child_id, node_dict, memo, depth + 1) for child_id in node.children]
    min_spacing_between_children_branches = 3.0
    total_width = sum(children_widths) + max(0, (len(children_widths) - 1)) * min_spacing_between_children_branches
    total_width = max(total_width, 2.0 * leaf_width)
    memo[node_id] = total_width
    return total_width


def layout_tree_base(nodes: List[TreeNode]) -> Tuple[Dict[int, float], float, float]:
    """Calculates y-positions for each node in a list of trees for plot layout."""
    if not nodes: return {}, 0.0, 0.0
    node_dict = {n.ID: n for n in nodes}
    child_to_structural_parent_local = {n.ID: n.parent_id for n in nodes if n.parent_id is not None}
    current_tree_roots = [n for n in nodes if n.parent_id is None or n.parent_id not in node_dict]
    if not current_tree_roots and nodes:
        current_tree_roots = [min(nodes, key=lambda n: n.generation)]
    subtree_widths_memo = {}
    for node in nodes: calculate_subtree_width(node.ID, node_dict, subtree_widths_memo)
    y_positions: Dict[int, float] = {}
    generations = defaultdict(list)
    for node in nodes: generations[node.generation].append(node)
    for gen_level_nodes in generations.values(): gen_level_nodes.sort(key=lambda n: n.ID)
    current_y_offset_for_roots_layout = 0.0
    root_layout_spacing = 10.0
    processed_roots_layout = set()
    
    # First pass: position root nodes
    for root_node_obj in current_tree_roots:
        if root_node_obj.ID in processed_roots_layout or root_node_obj.ID in y_positions: continue
        processed_roots_layout.add(root_node_obj.ID)
        root_subtree_w = subtree_widths_memo.get(root_node_obj.ID, 1.0)
        y_positions[root_node_obj.ID] = current_y_offset_for_roots_layout + root_subtree_w / 2.0
        current_y_offset_for_roots_layout += root_subtree_w + root_layout_spacing
    
    sibling_spacing_unit = 3.0
    
    # Second pass: position children, handling fusion events specially
    for gen_level in sorted(generations.keys()):
        for node_obj in generations[gen_level]:
            if node_obj.ID in y_positions: continue
            parent_id_of_node = child_to_structural_parent_local.get(node_obj.ID)
            if parent_id_of_node is None or parent_id_of_node not in y_positions:
                if node_obj.ID not in y_positions:
                    orphan_width = subtree_widths_memo.get(node_obj.ID, 1.0)
                    y_positions[node_obj.ID] = current_y_offset_for_roots_layout + orphan_width / 2.0
                    current_y_offset_for_roots_layout += orphan_width + root_layout_spacing
                continue
            
            parent_node_obj_ref = node_dict[parent_id_of_node]
            parent_y_center = y_positions[parent_id_of_node]
            
            # Check if this is a fusion event (node has multiple parents)
            # We need to find all nodes that have this node as a child
            potential_parents = []
            for potential_parent_id, potential_parent_node in node_dict.items():
                if node_obj.ID in potential_parent_node.children:
                    potential_parents.append(potential_parent_id)
            
            if len(potential_parents) > 1:
                # This is a fusion event - position the fused cell at the average of parent positions
                parent_positions = []
                for parent_id in potential_parents:
                    if parent_id in y_positions:
                        parent_positions.append(y_positions[parent_id])
                
                if parent_positions:
                    # Position fused cell at average of parent positions
                    y_positions[node_obj.ID] = sum(parent_positions) / len(parent_positions)
                    continue
            
            # Regular child positioning (non-fusion)
            siblings = sorted([node_dict[sid] for sid in parent_node_obj_ref.children if sid in node_dict],
                              key=lambda s: s.ID)
            if not siblings: continue
            try:
                current_node_index_in_siblings = [s.ID for s in siblings].index(node_obj.ID)
            except ValueError:
                continue
            total_siblings_width_sum = sum(subtree_widths_memo.get(s.ID, 1.0) for s in siblings)
            total_space_needed_for_siblings = total_siblings_width_sum + max(0,
                                                                             len(siblings) - 1) * sibling_spacing_unit
            current_child_y_start_offset = parent_y_center - (total_space_needed_for_siblings / 2.0)
            for i in range(current_node_index_in_siblings):
                current_child_y_start_offset += subtree_widths_memo.get(siblings[i].ID, 1.0) + sibling_spacing_unit
            sibling_subtree_w_val = subtree_widths_memo.get(node_obj.ID, 1.0)
            y_positions[node_obj.ID] = current_child_y_start_offset + sibling_subtree_w_val / 2.0
    min_y_coord, max_y_coord = 0.0, current_y_offset_for_roots_layout
    if y_positions:
        all_y_extents_calculated = [y_calc + pm * subtree_widths_memo.get(nid_calc, 1.0) / 2.0
                                    for nid_calc, y_calc in y_positions.items() for pm in [-1, 1]]
        if all_y_extents_calculated:
            min_y_coord = min(all_y_extents_calculated)
            max_y_coord = max(all_y_extents_calculated)
    last_y_fallback = max_y_coord
    for node_fb in nodes:
        if node_fb.ID not in y_positions:
            fb_width = subtree_widths_memo.get(node_fb.ID, 1.0)
            y_positions[node_fb.ID] = last_y_fallback + fb_width / 2.0 + root_layout_spacing
            last_y_fallback = y_positions[node_fb.ID] + fb_width / 2.0
            max_y_coord = max(max_y_coord, last_y_fallback)
    return y_positions, min_y_coord, max_y_coord


def _create_styled_label_html(text: str, color_tuple: Tuple[int, int, int], font_size: str = "11pt") -> str:
    """Helper to create HTML for styled labels."""
    hex_color = "#{:02x}{:02x}{:02x}".format(*color_tuple)
    return f'<span style="font-family: Arial; font-size: {font_size}; color: {hex_color};">{text}</span>'


def layout_track_segment_view(nodes: List[TreeNode], y_positions: Dict[int, float],
                              color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
                              cell_visibility: Optional[Dict[int, bool]] = None,
                              current_view_type: str = "Track Segments",
                              state_color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
                              ancestry_for_view: Optional[Dict[int, List[int]]] = None
                              ) -> Tuple[List[Edge], List[Annotation]]:
    edges = []
    annotations = []
    
    for current_node_obj_layout in nodes:
        if current_node_obj_layout.points is None or current_node_obj_layout.points.size == 0:
            continue
            
        y_pos_layout = y_positions.get(current_node_obj_layout.ID)
        if y_pos_layout is None:
            continue
            
        # Create track segment edge
        time_points_layout = current_node_obj_layout.points[:, 0]
        start_time, end_time = time_points_layout[0], time_points_layout[-1]
        
        # Determine color based on view type
        if current_view_type == "Class Type" and state_color_map and current_node_obj_layout.state:
            edge_color = state_color_map.get(current_node_obj_layout.state, (200, 200, 200))
        elif color_map:
            edge_color = color_map.get(current_node_obj_layout.ID, (200, 200, 200))
        else:
            edge_color = (200, 200, 200)
            
        edge = Edge(
            x=(start_time, end_time),
            y=(y_pos_layout, y_pos_layout),
            color=edge_color,
            track_id=current_node_obj_layout.ID,
            node=current_node_obj_layout
        )
        edges.append(edge)
        
        # Create annotation
        # For Track Segments view, only show ID, not state
        if current_view_type == "Track Segments":
            state_info = ""
        else:
            state_info = f" ({current_node_obj_layout.state})" if current_node_obj_layout.state else ""
        
        parent_info = ""
        if ancestry_for_view and current_node_obj_layout.ID in ancestry_for_view:
            parents = ancestry_for_view[current_node_obj_layout.ID]
            if parents:
                parent_info = f" [P:{','.join(map(str, parents))}]"
        
        label_text = f"{current_node_obj_layout.ID}{state_info}{parent_info}"
        annotation = Annotation(
            x=start_time,
            y=y_pos_layout,
            label=label_text,
            color=edge_color + (255,)
        )
        annotations.append(annotation)
        
        # Add connector edges to parents
        if ancestry_for_view and current_node_obj_layout.ID in ancestry_for_view:
            actual_parents_of_current = ancestry_for_view[current_node_obj_layout.ID]
            for parent_id_val_layout in actual_parents_of_current:
                if not cell_visibility or not cell_visibility.get(parent_id_val_layout, True):
                    continue
                    
                parent_y_pos = y_positions.get(parent_id_val_layout)
                if parent_y_pos is None:
                    continue
                    
                # Find parent's end time
                parent_node = next((n for n in nodes if n.ID == parent_id_val_layout), None)
                if parent_node and parent_node.points is not None and parent_node.points.size > 0:
                    parent_end_time = parent_node.points[-1, 0]
                    
                    connector_edge = Edge(
                        x=(parent_end_time, start_time),
                        y=(parent_y_pos, y_pos_layout),
                        color=CONNECTOR_EDGE_COLOR,
                        is_connector=True,
                        connected_child_id=current_node_obj_layout.ID
                    )
                    edges.append(connector_edge)
    
    return edges, annotations


def layout_node_based_view(nodes: List[TreeNode], y_positions: Dict[int, float],
                           color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
                           cell_visibility: Optional[Dict[int, bool]] = None,
                           current_view_type: str = "Node View",
                           state_color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
                           ancestry_for_view: Optional[Dict[int, List[int]]] = None
                           ) -> Tuple[List[NodePlotData], List[LinePlotData], List[Annotation]]:
    """Generates NodePlotData, LinePlotData, and Annotations for the 'Node View'."""
    if not nodes: return [], [], []
    cell_visibility = cell_visibility if cell_visibility is not None else {}
    color_map = color_map if color_map is not None else {}
    state_color_map = state_color_map if state_color_map is not None else {}
    ancestry_for_view = ancestry_for_view if ancestry_for_view is not None else {}
    visible_nodes_in_layout = [n for n in nodes if cell_visibility.get(n.ID, True)]
    if not visible_nodes_in_layout: return [], [], []
    effective_y_positions = {nid: y_val for nid, y_val in y_positions.items() if
                             nid in [vn.ID for vn in visible_nodes_in_layout]}
    node_plot_items_list: List[NodePlotData] = []
    line_plot_items_list: List[LinePlotData] = []
    annotations_list: List[Annotation] = []
    node_dict_visible = {n.ID: n for n in visible_nodes_in_layout}
    for current_node_obj_nv in visible_nodes_in_layout:
        track_y_layout_pos = effective_y_positions.get(current_node_obj_nv.ID)
        if track_y_layout_pos is None or current_node_obj_nv.points is None or current_node_obj_nv.points.size == 0:
            continue
        node_display_color_nv = color_map.get(current_node_obj_nv.ID, (128, 128, 128))
        if current_view_type == "Class Type" and current_node_obj_nv.state:
            node_display_color_nv = state_color_map.get(current_node_obj_nv.state, DEFAULT_STATE_COLOR)
        scatter_data_for_node = np.array(
            [[time_pt, track_y_layout_pos] for time_pt in current_node_obj_nv.points[:, 0]])
        node_plot_items_list.append(
            NodePlotData(points=scatter_data_for_node, color=node_display_color_nv, track_id=current_node_obj_nv.ID))
        if len(current_node_obj_nv.points) > 1:
            sorted_time_points_for_line = sorted(current_node_obj_nv.points[:, 0])
            # Create straight lines by using constant y-coordinate
            y_coords_for_line = [track_y_layout_pos] * len(sorted_time_points_for_line)
            
            line_plot_items_list.append(LinePlotData(x_coords=sorted_time_points_for_line, y_coords=y_coords_for_line,
                                                     color=node_display_color_nv, track_id=current_node_obj_nv.ID))
        label_text_content_nv = str(current_node_obj_nv.ID)
        if current_view_type == "Class Type" and current_node_obj_nv.state:
            label_text_content_nv += f" ({current_node_obj_nv.state})"
        actual_parents_nv = ancestry_for_view.get(current_node_obj_nv.ID, [])
        if actual_parents_nv:
            parent_str_nv = ",".join(map(str, sorted(actual_parents_nv)))
            label_text_content_nv += f" [P:{parent_str_nv}]"
        label_html_content_nv = _create_styled_label_html(label_text_content_nv, node_display_color_nv)
        annotations_list.append(Annotation(x=float(current_node_obj_nv.points[0, 0]), y=track_y_layout_pos,
                                           label=label_text_content_nv, html=label_html_content_nv,
                                           color=(*node_display_color_nv, 255)))
        for parent_id_val_nv in actual_parents_nv:
            parent_node_obj_nv = node_dict_visible.get(parent_id_val_nv)
            if not parent_node_obj_nv or parent_node_obj_nv.points is None or parent_node_obj_nv.points.size == 0: continue
            parent_y_layout_pos_nv = effective_y_positions.get(parent_id_val_nv)
            if parent_y_layout_pos_nv is None: continue
            mitosis_line_x_coords = [parent_node_obj_nv.points[-1, 0], current_node_obj_nv.points[0, 0]]
            mitosis_line_y_coords = [parent_y_layout_pos_nv, track_y_layout_pos]
            line_plot_items_list.append(LinePlotData(x_coords=mitosis_line_x_coords, y_coords=mitosis_line_y_coords,
                                                     color=NODE_VIEW_MITOSIS_LINE_COLOR, is_mitosis_connector=True))
    return node_plot_items_list, line_plot_items_list, annotations_list


# --- PyQtGraph Widget ---
class LineageTreeWidget(QWidget):
    Y_PADDING_BETWEEN_TREES = 20.0
    HIGHLIGHT_PEN_WIDTH = 4
    DEFAULT_PEN_WIDTH = 2
    CONNECTOR_PEN_WIDTH = 1.5
    NODE_SIZE = 10
    HIGHLIGHT_NODE_SIZE = 14
    NODE_LINE_WIDTH = 2.0
    HIGHLIGHT_NODE_LINE_WIDTH = 3.0
    NODE_CONNECTOR_LINE_WIDTH = 1.5
    NODE_CLICK_Y_TOLERANCE_VIEW_UNITS = 12.0  # Increased for easier clicking and hovering
    DRAG_THRESHOLD_DISTANCE = 5.0  # Minimum distance to start drag operation
    TOOLTIP_HOVER_TOLERANCE = 15.0  # Tolerance for tooltip hover detection

    # Signals
    nodeOperationRequested = pyqtSignal(int, int, str)  # dragged_id, target_id, operation_name
    nodeClickedInView = pyqtSignal(int, int)  # track_id, frame_index
    nodeBreakRequested = pyqtSignal(int, int)  # track_id, frame_index
    splitAndFuseHeadRequested = pyqtSignal(int, int, int)  # original_track_id, break_frame, target_fusion_id
    insertAndSplitRequested = pyqtSignal(int, int, int)  # dragged_id, target_id, insert_frame

    def __init__(self, parent=None):
        super().__init__(parent)
        self.trj_data: Optional[pd.DataFrame] = None
        self.color_list_main: Optional[List[Tuple[int, int, int]]] = None
        self.cell_color_idx_main: Optional[Dict[int, int]] = None
        self.cell_visibility_main: Optional[Dict[int, bool]] = None
        self.track_states_main: Optional[Dict[int, str]] = None
        self.state_color_map_internal: Dict[str, Tuple[int, int, int]] = {}
        self.ancestry_data_main: Optional[Dict[int, List[int]]] = None
        self.lineage_data_main: Optional[Dict[int, List[int]]] = None
        self.current_root_id: Optional[int] = None
        self.selected_root_ids: List[int] = []  # Store multiple selected roots
        self.all_roots: List[int] = []
        self.highlighted_track_id: Optional[int] = None
        self.current_view_type: str = 'Track Segments'
        self.node_view_layout_data: Dict[int, Tuple[np.ndarray, float]] = {}
        # Add storage for track segment and class type layout data
        self.track_segment_layout_data: Dict[int, Tuple[np.ndarray, float]] = {}
        self.dragged_node_id: Optional[int] = None
        self.drag_start_scene_pos: Optional[QPointF] = None
        self.temp_drag_line_item: Optional[pg.PlotDataItem] = None
        # Enhanced visual feedback variables
        self.hovered_node_id: Optional[int] = None
        self.hover_highlight_item: Optional[pg.PlotDataItem] = None
        self.drag_preview_text: Optional[pg.TextItem] = None
        self.drop_zone_highlight: Optional[pg.PlotDataItem] = None
        self.tooltip_text_item: Optional[pg.TextItem] = None
        # Drag detection variables
        self.potential_drag_node_id: Optional[int] = None
        self.drag_started: bool = False
        self.plot_item = InteractivePlotItem(lineage_widget_ref=self)
        self.plot_widget = pg.PlotWidget(plotItem=self.plot_item)
        self.plot_item.setLabel('bottom', 'Time (frames)')
        self.plot_item.setLabel('left', 'Lineage Branch')
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)
        self.plot_item.invertY(True)
        self.plot_item.vb.sigRangeChanged.connect(self._on_view_changed)
        self.current_frame_indicator_line = pg.InfiniteLine(angle=90, movable=False,
                                                            pen=pg.mkPen(CURRENT_FRAME_INDICATOR_COLOR, width=1.5,
                                                                         style=Qt.DashLine))
        self.current_frame_indicator_line.setZValue(100)
        self.current_frame_indicator_line.setVisible(False)
        self.plot_item.addItem(self.current_frame_indicator_line)
        self.view_type_label = QLabel("View Type:")
        self.view_type_group = QButtonGroup(self)
        self.rb_track_segments = QRadioButton("Track Segments");
        self.rb_track_segments.setChecked(True)
        self.rb_node_view = QRadioButton("Node View")
        self.rb_class_type = QRadioButton("Class Type")
        self.view_type_group.addButton(self.rb_track_segments);
        self.view_type_group.addButton(self.rb_node_view);
        self.view_type_group.addButton(self.rb_class_type)
        self.rb_track_segments.toggled.connect(
            lambda checked: self.set_view_type("Track Segments") if checked else None)
        self.rb_node_view.toggled.connect(lambda checked: self.set_view_type("Node View") if checked else None)
        self.rb_class_type.toggled.connect(lambda checked: self.set_view_type("Class Type") if checked else None)
        view_type_hbox = QHBoxLayout();
        view_type_hbox.addWidget(self.view_type_label);
        view_type_hbox.addWidget(self.rb_track_segments);
        view_type_hbox.addWidget(self.rb_node_view);
        view_type_hbox.addWidget(self.rb_class_type);
        view_type_hbox.addStretch()
        self.root_selection_label = QLabel("Select Root Track ID:")
        self.root_list_widget = QListWidget();
        self.root_list_widget.setMaximumHeight(100);
        # Enable multi-selection
        self.root_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.root_list_widget.itemSelectionChanged.connect(self._on_root_selection_changed)
        self.root_list_widget.itemDoubleClicked.connect(self._on_root_double_clicked)
        self.filter_label = QLabel("Filter Roots:")
        self.filter_input = QLineEdit();
        self.filter_input.setPlaceholderText("Enter ID to find...");
        self.filter_input.textChanged.connect(self._filter_roots)
        self.show_all_button = QPushButton("Show All Lineages");
        self.show_all_button.clicked.connect(self.draw_all_lineage_trees)
        
        # Add save button for the lineage plot
        self.save_plot_button = QPushButton("Save Plot");
        self.save_plot_button.clicked.connect(self._save_current_plot)
        self.save_plot_button.setToolTip("Save the current lineage plot as an image file")
        
        # Add refresh button
        self.refresh_button = QPushButton("Refresh");
        self.refresh_button.clicked.connect(self._refresh_lineage_tree)
        self.refresh_button.setToolTip("Refresh lineage tree to reflect current data (F5)")
        
        controls_hbox = QHBoxLayout();
        controls_hbox.addWidget(self.root_selection_label);
        controls_hbox.addWidget(self.root_list_widget);
        controls_hbox.addWidget(self.filter_label);
        controls_hbox.addWidget(self.filter_input);
        controls_hbox.addWidget(self.show_all_button);
        controls_hbox.addWidget(self.save_plot_button);
        controls_hbox.addWidget(self.refresh_button)
        main_layout = QVBoxLayout();
        main_layout.addLayout(view_type_hbox);
        main_layout.addLayout(controls_hbox);
        main_layout.addWidget(self.plot_widget);
        self.setLayout(main_layout)
        
        # Add event filter for debugging mouse events
        self.plot_widget.installEventFilter(self)

    def _on_view_changed(self, viewbox, range_rect):
        """LOGGING HOOK: Called when the plot view is panned or zoomed."""
        x_range = range_rect[0]
        visible_time_width = x_range[1] - x_range[0]
        logging.debug(f"[ViewChange] View range changed. Visible time axis width: {visible_time_width:.2f} frames. "
                     f"This event can be used to dynamically adjust labels.")

    def eventFilter(self, obj, event):
        """Event filter to debug mouse events reaching the plot widget."""
        # Removed excessive debug logging to improve performance
        return super().eventFilter(obj, event)

    def update_current_frame_indicator(self, frame_index: int):
        if self.current_frame_indicator_line:
            self.current_frame_indicator_line.setPos(frame_index)

    def export_current_plot_as_image(self, file_path: str, high_resolution: bool = True):
        """
        Export the current lineage plot as an image file.
        
        Args:
            file_path: Path where to save the image
            high_resolution: If True, creates a high-resolution image optimized for large screens
        """
        try:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory): os.makedirs(directory, exist_ok=True)
            
            # Hide the current frame indicator line before saving
            was_visible = self.current_frame_indicator_line.isVisible()
            self.current_frame_indicator_line.setVisible(False)
            
            if high_resolution:
                # Create high-resolution export optimized for large screens
                self._export_high_resolution_image(file_path)
            else:
                # Use standard PyQtGraph exporter
                exporter = exporters.ImageExporter(self.plot_item)
                exporter.export(file_path)
            
            # Restore the current frame indicator line visibility
            self.current_frame_indicator_line.setVisible(was_visible)
            
            logging.info(f"Lineage plot successfully exported to: {file_path}")
        except Exception as e:
            # Make sure to restore visibility even if export fails
            self.current_frame_indicator_line.setVisible(True)
            logging.error(f"Failed to export lineage plot to {file_path}: {e}\n{traceback.format_exc()}")

    def _export_high_resolution_image(self, file_path: str):
        """
        Export a high-resolution image optimized for large screens.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.colors import to_rgb
            import numpy as np
            
            # Get current plot data
            if not self.trj_data or self.trj_data.empty:
                logging.warning("No trajectory data available for high-resolution export")
                return
            
            # Determine figure size for large screens (e.g., 4K display)
            # 16:9 aspect ratio, 300 DPI for print quality
            fig_width = 16.0  # inches
            fig_height = 9.0   # inches
            dpi = 300
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
            
            # Get current view data
            tree_nodes, y_positions = self._prepare_common_data_for_drawing()
            
            if not tree_nodes:
                logging.warning("No tree nodes available for export")
                return
            
            # Determine plot ranges
            all_frames = []
            all_y_positions = []
            
            for node in tree_nodes:
                if len(node.t) > 0:
                    all_frames.extend(node.t)
                    all_y_positions.append(y_positions.get(node.ID, 0))
            
            if not all_frames:
                logging.warning("No frame data available for export")
                return
            
            x_min, x_max = min(all_frames), max(all_frames)
            y_min, y_max = min(all_y_positions) - 10, max(all_y_positions) + 10
            
            # Set up the plot
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Set labels and title
            ax.set_xlabel('Time (frames)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Lineage Branch', fontsize=14, fontweight='bold')
            
            # Create title based on current view type and selection
            title_parts = [f"Lineage Tree - {self.current_view_type}"]
            if self.selected_root_ids:
                title_parts.append(f"Roots: {', '.join(map(str, self.selected_root_ids))}")
            elif self.current_root_id is not None:
                title_parts.append(f"Root: {self.current_root_id}")
            else:
                title_parts.append("All Lineages")
            
            ax.set_title(' - '.join(title_parts), fontsize=16, fontweight='bold', pad=20)
            
            # Plot track segments
            if self.current_view_type == "Track Segments":
                self._plot_track_segments_matplotlib(ax, tree_nodes, y_positions)
            elif self.current_view_type == "Class Type":
                self._plot_class_type_matplotlib(ax, tree_nodes, y_positions)
            elif self.current_view_type == "Node View":
                self._plot_node_view_matplotlib(ax, tree_nodes, y_positions)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(file_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none',
                       format=file_path.split('.')[-1].lower())
            
            plt.close(fig)
            
        except ImportError:
            logging.warning("Matplotlib not available, falling back to standard export")
            exporter = exporters.ImageExporter(self.plot_item)
            exporter.export(file_path)
        except Exception as e:
            logging.error(f"Error in high-resolution export: {e}")
            # Fallback to standard export
            exporter = exporters.ImageExporter(self.plot_item)
            exporter.export(file_path)

    def _plot_track_segments_matplotlib(self, ax, tree_nodes, y_positions):
        """Plot track segments view using matplotlib."""
        import matplotlib.patches as patches
        
        # Plot horizontal track segments
        for node in tree_nodes:
            if len(node.t) > 0:
                y_pos = y_positions.get(node.ID, 0)
                start_frame = min(node.t)
                end_frame = max(node.t)
                
                # Determine color
                if hasattr(self, 'state_color_map_internal') and node.state in self.state_color_map_internal:
                    color = tuple(c/255 for c in self.state_color_map_internal[node.state])
                else:
                    color = (0.2, 0.2, 0.2)  # Default dark gray
                
                # Plot track segment
                ax.plot([start_frame, end_frame], [y_pos, y_pos], 
                       color=color, linewidth=3, solid_capstyle='round')
                
                # Add track ID label
                ax.text(start_frame - 2, y_pos, str(node.ID), 
                       fontsize=10, ha='right', va='center', fontweight='bold')
        
        # Plot connection lines between parents and children
        if hasattr(self, 'ancestry_data_main') and self.ancestry_data_main:
            for child_id, parents in self.ancestry_data_main.items():
                for parent_id in parents:
                    if parent_id in y_positions and child_id in y_positions:
                        parent_y = y_positions[parent_id]
                        child_y = y_positions[child_id]
                        
                        # Find connection point (end of parent, start of child)
                        parent_node = next((n for n in tree_nodes if n.ID == parent_id), None)
                        child_node = next((n for n in tree_nodes if n.ID == child_id), None)
                        
                        if parent_node and child_node and len(parent_node.t) > 0 and len(child_node.t) > 0:
                            connect_x = max(parent_node.t)
                            ax.plot([connect_x, connect_x], [parent_y, child_y], 
                                   color='red', linewidth=2, linestyle='--', alpha=0.7)

    def _plot_class_type_matplotlib(self, ax, tree_nodes, y_positions):
        """Plot class type view using matplotlib."""
        # Similar to track segments but with state-based coloring
        self._plot_track_segments_matplotlib(ax, tree_nodes, y_positions)

    def _plot_node_view_matplotlib(self, ax, tree_nodes, y_positions):
        """Plot node view using matplotlib."""
        import matplotlib.patches as patches
        
        # Plot nodes as circles
        for node in tree_nodes:
            if len(node.t) > 0:
                y_pos = y_positions.get(node.ID, 0)
                
                # Determine color
                if hasattr(self, 'state_color_map_internal') and node.state in self.state_color_map_internal:
                    color = tuple(c/255 for c in self.state_color_map_internal[node.state])
                else:
                    color = (0.2, 0.2, 0.2)
                
                # Plot node at each time point
                for t in node.t:
                    circle = patches.Circle((t, y_pos), radius=0.5, 
                                          facecolor=color, edgecolor='black', linewidth=1)
                    ax.add_patch(circle)
                
                # Add track ID label
                if len(node.t) > 0:
                    ax.text(min(node.t) - 2, y_pos, str(node.ID), 
                           fontsize=10, ha='right', va='center', fontweight='bold')
        
        # Plot connection lines
        if hasattr(self, 'ancestry_data_main') and self.ancestry_data_main:
            for child_id, parents in self.ancestry_data_main.items():
                for parent_id in parents:
                    if parent_id in y_positions and child_id in y_positions:
                        parent_y = y_positions[parent_id]
                        child_y = y_positions[child_id]
                        
                        parent_node = next((n for n in tree_nodes if n.ID == parent_id), None)
                        child_node = next((n for n in tree_nodes if n.ID == child_id), None)
                        
                        if parent_node and child_node and len(parent_node.t) > 0 and len(child_node.t) > 0:
                            connect_x = max(parent_node.t)
                            ax.plot([connect_x, connect_x], [parent_y, child_y], 
                                   color='red', linewidth=2, linestyle='--', alpha=0.7)

    def _save_current_plot(self):
        """Save the current lineage plot as an image file."""
        # Generate default filename based on current view type and selection
        default_filename = f"lineage_plot_{self.current_view_type.lower().replace(' ', '_')}"
        if self.selected_root_ids:
            default_filename += f"_roots_{'_'.join(map(str, self.selected_root_ids))}"
        elif self.current_root_id is not None:
            default_filename += f"_root_{self.current_root_id}"
        else:
            default_filename += "_all_lineages"
        default_filename += ".png"
        
        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Lineage Plot",
            default_filename,
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        
        if file_path:
            try:
                # Use high-resolution export for better large screen display
                self.export_current_plot_as_image(file_path, high_resolution=True)
                # Show success message
                QMessageBox.information(
                    self,
                    "Save Successful",
                    f"High-resolution lineage plot saved successfully to:\n{file_path}"
                )
            except Exception as e:
                # Show error message
                QMessageBox.critical(
                    self,
                    "Save Failed",
                    f"Failed to save lineage plot:\n{str(e)}"
                )

    def _refresh_lineage_tree(self):
        """Refresh the lineage tree to reflect current data."""
        try:
            # Re-draw based on current selection state
            if self.selected_root_ids:
                self.draw_lineage_trees_for_selection(self.selected_root_ids)
            elif self.current_root_id is not None:
                self.draw_lineage_tree_for_single_root(self.current_root_id)
            else:
                self.draw_all_lineage_trees()
            
            # Force immediate repaint
            self.plot_item.update()
            QApplication.processEvents()
            
            logging.info("Lineage tree refreshed successfully")
        except Exception as e:
            logging.error(f"Error refreshing lineage tree: {e}")
            QMessageBox.critical(self, "Refresh Error", f"Error refreshing lineage tree: {e}")

    def _restore_view_range(self, saved_range):
        """Restore a previously saved view range."""
        if saved_range and 'x_range' in saved_range and 'y_range' in saved_range:
            try:
                self.plot_item.setXRange(*saved_range['x_range'], padding=0)
                self.plot_item.setYRange(*saved_range['y_range'], padding=0)
                return True
            except:
                pass
        return False

    def _generate_state_colors(self):
        self.state_color_map_internal.clear()
        if not self.track_states_main: return
        unique_states = sorted(
            list(set(s for s in self.track_states_main.values() if s and s not in ["N/A", "Unclassified", "Hidden"])))
        predefined_colors = {"Unclassified": (100, 100, 100), "N/A": (80, 80, 80), "Hidden": (50, 50, 50)}
        self.state_color_map_internal.update(predefined_colors)
        for state_name in unique_states:
            if state_name not in self.state_color_map_internal:
                r, g, b = random.randint(50, 250), random.randint(50, 250), random.randint(50, 250)
                if abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
                    r = (r + 100) % 256;
                    g = (g + 150) % 256
                self.state_color_map_internal[state_name] = (r, g, b)
        logging.info(f"Generated state colors: {self.state_color_map_internal}")

    def set_view_type(self, view_type: str):
        if view_type == self.current_view_type:
            return
        self.current_view_type = view_type
        # Clear layout data when switching views
        self.node_view_layout_data.clear()
        self.track_segment_layout_data.clear()
        # Re-draw based on current selection state
        if self.current_root_id is not None and self.current_root_id in self.all_roots:
            self.draw_lineage_tree_for_single_root(self.current_root_id)
        elif self.selected_root_ids:
            self.draw_lineage_trees_for_selection(self.selected_root_ids)
        else:
            self.draw_all_lineage_trees()

    def _filter_roots(self, text: str):
        self.root_list_widget.clear()
        if not self.all_roots: return
        items_to_add = [str(r) for r in self.all_roots if not text or text.lower() in str(r).lower()]
        self.root_list_widget.addItems(items_to_add)

    def _on_root_double_clicked(self, item):
        """On double-click, show only the selected root's tree."""
        try:
            selected_root_id = int(item.text())
            # Temporarily block signals to prevent selection changed from firing
            self.root_list_widget.blockSignals(True)
            self.root_list_widget.clearSelection()
            item.setSelected(True)
            self.root_list_widget.blockSignals(False)
            self.selected_root_ids = [selected_root_id]
            self.draw_lineage_tree_for_single_root(selected_root_id)
        except (ValueError, AttributeError) as e:
            logging.warning(f"Invalid root ID double-clicked from list: {item.text() if item else 'None'}. Error: {e}")

    def _on_root_selection_changed(self):
        """On any change in selection, draw the trees for all selected roots."""
        selected_items = self.root_list_widget.selectedItems()
        if not selected_items:
            self.selected_root_ids = []
            self.current_root_id = None
            self.draw_all_lineage_trees()
            return

        try:
            self.selected_root_ids = sorted([int(item.text()) for item in selected_items])
            logging.info(f"Root selection changed. Selected IDs: {self.selected_root_ids}")
            self.current_root_id = None  # Multi-selection overrides single root view
            self.draw_lineage_trees_for_selection(self.selected_root_ids)
        except ValueError as e:
            logging.error(f"Error processing root selection: {e}")

    def set_highlighted_track(self, track_id: Optional[int]):
        if self.highlighted_track_id != track_id:
            self.highlighted_track_id = track_id
            if self.trj_data is not None and not self.trj_data.empty:
                # Re-draw based on current selection state
                if self.selected_root_ids:
                    self.draw_lineage_trees_for_selection(self.selected_root_ids)
                elif self.current_root_id is not None:
                    self.draw_lineage_tree_for_single_root(self.current_root_id)
                else:
                    self.draw_all_lineage_trees()
            elif self.highlighted_track_id is not None:
                self.highlighted_track_id = None

    def set_data(self, trj_df: pd.DataFrame, color_list: List, cell_color_idx: Dict,
                 cell_visibility: Dict, track_states: Optional[Dict[int, str]] = None,
                 ancestry_map: Optional[Dict[int, List[int]]] = None,
                 timing_params: Optional[Dict] = None):
        self.trj_data = trj_df.copy() if trj_df is not None else pd.DataFrame()
        self.color_list_main = color_list if color_list is not None else []
        self.cell_color_idx_main = cell_color_idx
        self.cell_visibility_main = cell_visibility.copy() if cell_visibility is not None else {}
        self.track_states_main = track_states.copy() if track_states is not None else {}
        self.ancestry_data_main = ancestry_map.copy() if ancestry_map is not None else {}
        # Clear layout data when setting new data
        self.node_view_layout_data.clear()
        self.track_segment_layout_data.clear()
        # Extract lineage data from ancestry (parent -> children mapping)
        self.lineage_data_main = {}
        if self.ancestry_data_main and isinstance(self.ancestry_data_main, dict):
            for child_id, parent_list in self.ancestry_data_main.items():
                for parent_id in parent_list:
                    if parent_id not in self.lineage_data_main:
                        self.lineage_data_main[parent_id] = []
                    self.lineage_data_main[parent_id].append(child_id)
        
        # Store timing parameters for time calculations
        self.timing_params = timing_params or {}
        self.pixel_scale_m = self.timing_params.get('Pixel scale', (0.87e-6,))[0] if isinstance(self.timing_params.get('Pixel scale'), tuple) else 0.87e-6
        self.frames_per_hour = self.timing_params.get('Frames per hour', (12,))[0] if isinstance(self.timing_params.get('Frames per hour'), tuple) else 12
        if self.track_states_main: self._generate_state_colors()
        self.current_root_id = None
        self.selected_root_ids = []
        self.all_roots = []
        self.root_list_widget.clear()
        
        # Clean up all visual feedback
        self._remove_hover_highlight()
        self._remove_drop_zone_highlight()
        self._remove_drag_preview_text()
        self._remove_tooltip()
        if self.temp_drag_line_item:
            self.plot_item.removeItem(self.temp_drag_line_item)
            self.temp_drag_line_item = None
        self.dragged_node_id = None
        self.drag_start_scene_pos = None
        self.hovered_node_id = None
        self.potential_drag_node_id = None
        self.drag_started = False
        
        self.plot_item.clear();
        self.plot_item.addItem(self.current_frame_indicator_line)
        self.highlighted_track_id = None
        if self.trj_data.empty or not all(c in self.trj_data.columns for c in ['particle', 'frame']):
            self.plot_item.setTitle(f"No/Invalid track data ({self.current_view_type})")
            self.show_all_button.setEnabled(False);
            self.current_frame_indicator_line.setVisible(False)
            return
        self.trj_data[['particle', 'frame']] = self.trj_data[['particle', 'frame']].astype(float).astype(int)
        if 'y' not in self.trj_data.columns: self.trj_data['y'] = 0.0
        if 'x' not in self.trj_data.columns: self.trj_data['x'] = 0.0
        current_ancestry_graph = self.ancestry_data_main if isinstance(self.ancestry_data_main, dict) else {}
        _, current_lineage_graph = build_reverse_graph(current_ancestry_graph)
        self.lineage_data_main = current_lineage_graph
        all_ids_in_trj = set(self.trj_data['particle'].unique().astype(int))
        all_ids_in_ancestry_keys = set(current_ancestry_graph.keys())
        all_ids_in_ancestry_values = set(p for parents in current_ancestry_graph.values() for p in parents)
        all_nodes_considered_for_roots = all_ids_in_trj.union(all_ids_in_ancestry_keys).union(
            all_ids_in_ancestry_values)
        self.all_roots = sorted(
            list(node_id for node_id in all_nodes_considered_for_roots if not current_ancestry_graph.get(node_id)))
        if not self.all_roots:
            roots_from_rev_graph_fallback, _ = build_reverse_graph(current_ancestry_graph)
            self.all_roots = sorted(list(set(roots_from_rev_graph_fallback)))
        
        # Debug: Check if specific track IDs are in the root list
        if 24 in all_ids_in_trj:
            logging.info(f"Track 24 is in trajectory data: {24 in all_ids_in_trj}")
            logging.info(f"Track 24 in ancestry keys: {24 in all_ids_in_ancestry_keys}")
            logging.info(f"Track 24 in ancestry values: {24 in all_ids_in_ancestry_values}")
            logging.info(f"Track 24 in ancestry graph: {current_ancestry_graph.get(24)}")
            logging.info(f"Track 24 in final roots: {24 in self.all_roots}")
        
        if not self.all_roots:
            self.plot_item.setTitle(f"No roots found ({self.current_view_type})")
            self.show_all_button.setEnabled(False);
            self.current_frame_indicator_line.setVisible(False)
            return
        self.root_list_widget.addItems([str(r_id) for r_id in self.all_roots])
        self.show_all_button.setEnabled(True)
        self.draw_all_lineage_trees()

    def _prepare_common_data_for_drawing(self):
        temp_trj_for_np = self.trj_data.copy()
        temp_trj_for_np['y_for_points'] = temp_trj_for_np['y'] if 'y' in temp_trj_for_np else 0.0
        temp_trj_for_np['x_for_points'] = temp_trj_for_np['x'] if 'x' in temp_trj_for_np else 0.0
        mock_data_np_array = temp_trj_for_np[['particle', 'frame', 'y_for_points', 'x_for_points']].to_numpy()
        current_ancestry = self.ancestry_data_main if (self.ancestry_data_main is not None and isinstance(self.ancestry_data_main, dict)) else {}
        current_lineage = self.lineage_data_main if self.lineage_data_main is not None else {}
        track_color_map_default = {tid: self.color_list_main[cidx] for tid, cidx in self.cell_color_idx_main.items()
                                   if self.color_list_main and 0 <= cidx < len(
                self.color_list_main)} if self.cell_color_idx_main else {}
        
        return mock_data_np_array, current_ancestry, current_lineage, track_color_map_default

    def draw_lineage_tree_for_single_root(self, root_track_id: int):
        if self.trj_data is None: return
        self.plot_item.clear();
        self.plot_item.addItem(self.current_frame_indicator_line)
        self.current_root_id = root_track_id
        self.plot_item.setTitle(f"Lineage for Root: {root_track_id} ({self.current_view_type})")
        mock_data_np, ancestry_graph, lineage_graph, color_map = self._prepare_common_data_for_drawing()
        tree_nodes_list = build_subgraph(mock_data_np, ancestry_graph, lineage_graph, root_track_id,
                                         self.track_states_main)
        if not tree_nodes_list:
            self._set_empty_plot_title_if_needed(tree_nodes_list, root_track_id)
            self.node_view_layout_data.clear();
            self.current_frame_indicator_line.setVisible(False)
            return
        y_positions_map, min_y_val, max_y_val = layout_tree_base(tree_nodes_list)
        self._store_node_layout_data(tree_nodes_list, y_positions_map)
        # Store track segment layout data for Track Segments and Class Type views
        if self.current_view_type in ['Track Segments', 'Class Type']:
            self._store_track_segment_layout_data(tree_nodes_list, y_positions_map)
        if self.current_view_type == 'Node View':
            nodes_plot_data, lines_plot_data, annotations_data = layout_node_based_view(
                tree_nodes_list, y_positions_map, color_map, self.cell_visibility_main, self.current_view_type,
                self.state_color_map_internal, ancestry_graph)
            if not nodes_plot_data and not lines_plot_data and not annotations_data:
                self._set_empty_plot_title_if_needed(tree_nodes_list, root_track_id);
                self.current_frame_indicator_line.setVisible(False)
                return
            self._plot_node_view_data(nodes_plot_data, lines_plot_data, annotations_data, min_y_val, max_y_val)
        else:
            edges_data, annotations_data = layout_track_segment_view(
                tree_nodes_list, y_positions_map, color_map, self.cell_visibility_main, self.current_view_type,
                self.state_color_map_internal, ancestry_graph)
            if not edges_data and not annotations_data:
                self._set_empty_plot_title_if_needed(tree_nodes_list, root_track_id);
                self.current_frame_indicator_line.setVisible(False)
                return
            self._plot_track_segment_data(edges_data, annotations_data, min_y_val, max_y_val)
        self.current_frame_indicator_line.setVisible(True)

    def draw_lineage_trees_for_selection(self, root_ids: List[int]):
        """Draws a 'forest' view for a specific list of selected root IDs."""
        
        if self.trj_data is None or not root_ids:
            # If the list is empty, revert to showing all
            self.draw_all_lineage_trees()
            return
        
        self.plot_item.clear()
        self.plot_item.addItem(self.current_frame_indicator_line)
        self.current_root_id = None  # Not in single root mode
        self.plot_item.setTitle(f"Selected Lineages ({len(root_ids)} trees) ({self.current_view_type})")

        mock_data_np_all, ancestry_graph_all, lineage_graph_all, color_map_all_default = self._prepare_common_data_for_drawing()
        
        self.node_view_layout_data.clear()
        y_offset_cumulative = 0.0
        min_y_overall, max_y_overall = float('inf'), float('-inf')
        all_node_plot_items, all_line_plot_items, all_edge_items, all_annotation_items = [], [], [], []
        plotted_anything = False
        
        # Deduplicate trees by finding actual unique roots
        actual_roots = set()
        for root_id_val in sorted(root_ids):
            tree_nodes_for_root = build_subgraph(mock_data_np_all, ancestry_graph_all, lineage_graph_all,
                                                 root_id_val, self.track_states_main)
            if tree_nodes_for_root:
                # Find the actual root of this tree
                actual_root = min(tree_nodes_for_root, key=lambda n: n.generation).ID
                actual_roots.add(actual_root)
        
        # Now draw each unique tree only once
        for actual_root_id in sorted(actual_roots):
            tree_nodes_for_root = build_subgraph(mock_data_np_all, ancestry_graph_all, lineage_graph_all,
                                                 actual_root_id, self.track_states_main)
            
            if not tree_nodes_for_root or not any(
                    self.cell_visibility_main.get(tn.ID, True) for tn in tree_nodes_for_root): 
                continue

            y_pos_local_map, tree_min_y_local, tree_max_y_local = layout_tree_base(tree_nodes_for_root)
            y_shift_for_this_tree = y_offset_cumulative - tree_min_y_local
            y_pos_shifted_map = {tid: y_val + y_shift_for_this_tree for tid, y_val in y_pos_local_map.items()}
            self._store_node_layout_data(tree_nodes_for_root, y_pos_shifted_map)
            # Store track segment layout data for Track Segments and Class Type views
            if self.current_view_type in ['Track Segments', 'Class Type']:
                self._store_track_segment_layout_data(tree_nodes_for_root, y_pos_shifted_map)

            if self.current_view_type == 'Node View':
                n_p_data, l_p_data, a_p_data = layout_node_based_view(tree_nodes_for_root, y_pos_shifted_map,
                                                                      color_map_all_default, self.cell_visibility_main,
                                                                      self.current_view_type,
                                                                      self.state_color_map_internal, ancestry_graph_all)
                if n_p_data or l_p_data or a_p_data: plotted_anything = True
                all_node_plot_items.extend(n_p_data);
                all_line_plot_items.extend(l_p_data)
            else:  # Track Segments or Class Type
                e_p_data, a_p_data = layout_track_segment_view(tree_nodes_for_root, y_pos_shifted_map,
                                                               color_map_all_default, self.cell_visibility_main,
                                                               self.current_view_type, self.state_color_map_internal,
                                                               ancestry_graph_all)
                if e_p_data or a_p_data: plotted_anything = True
                all_edge_items.extend(e_p_data)

            all_annotation_items.extend(a_p_data)
            current_tree_max_y_shifted = tree_max_y_local + y_shift_for_this_tree
            min_y_overall = min(min_y_overall, tree_min_y_local + y_shift_for_this_tree)
            max_y_overall = max(max_y_overall, current_tree_max_y_shifted)
            y_offset_cumulative = current_tree_max_y_shifted + self.Y_PADDING_BETWEEN_TREES

        if not plotted_anything:
            self.plot_item.setTitle(f"No visible lineages in selection ({self.current_view_type})")
            self.current_frame_indicator_line.setVisible(False);
            return

        # Update title to show actual number of trees drawn
        actual_tree_count = len(actual_roots)
        self.plot_item.setTitle(f"Lineage Forest ({actual_tree_count} tree{'s' if actual_tree_count != 1 else ''}) ({self.current_view_type})")

        if self.current_view_type == 'Node View':
            self._plot_node_view_data(all_node_plot_items, all_line_plot_items, all_annotation_items, min_y_overall,
                                      max_y_overall)
        else:
            self._plot_track_segment_data(all_edge_items, all_annotation_items, min_y_overall, max_y_overall)
        self.current_frame_indicator_line.setVisible(True)

    def draw_all_lineage_trees(self):
        # When showing all, clear any selection in the UI
        self.root_list_widget.blockSignals(True)
        self.root_list_widget.clearSelection()
        self.root_list_widget.blockSignals(False)
        self.selected_root_ids = []
        
        # Optimize for faster drawing
        self.draw_lineage_trees_for_selection(self.all_roots)
        # Note: The title is now set in draw_lineage_trees_for_selection based on actual trees drawn
        
        # Auto-fit all data in the view with reduced logging
        self._auto_fit_view_to_data()

    def _auto_fit_view_to_data(self):
        """Automatically fit the view to show all data with appropriate padding."""
        try:
            # Use PyQtGraph's built-in auto-range functionality
            self.plot_item.autoRange()
            
            # Get the auto-ranged view and add some padding
            current_range = self.plot_item.viewRange()
            if current_range and len(current_range) >= 2:
                x_range, y_range = current_range[0], current_range[1]
                
                # Add 10% padding to both axes
                x_padding = (x_range[1] - x_range[0]) * 0.1
                y_padding = (y_range[1] - y_range[0]) * 0.1
                
                # Set the view with padding
                self.plot_item.setXRange(x_range[0] - x_padding, x_range[1] + x_padding, padding=0)
                self.plot_item.setYRange(y_range[0] - y_padding, y_range[1] + y_padding, padding=0)
                
                # Reduced logging for better performance
                logging.debug(f"Auto-fitted view with padding: X({x_range[0] - x_padding:.1f}, {x_range[1] + x_padding:.1f}), Y({y_range[0] - y_padding:.1f}, {y_range[1] + y_padding:.1f})")
                
        except Exception as e:
            logging.warning(f"Error in auto-fit view: {e}")
            # Fallback to default range
            self.plot_item.setXRange(0, 10, padding=0)
            self.plot_item.setYRange(0, 10, padding=0)

    def _store_node_layout_data(self, tree_nodes: List[TreeNode], y_positions: Dict[int, float]):
        for node_obj_store in tree_nodes:
            if node_obj_store.ID in y_positions and node_obj_store.points is not None:
                self.node_view_layout_data[node_obj_store.ID] = (node_obj_store.points[:, 0],
                                                                 y_positions[node_obj_store.ID])

    def _store_track_segment_layout_data(self, tree_nodes: List[TreeNode], y_positions: Dict[int, float]):
        """Store layout data for Track Segments and Class Type views."""
        for node_obj_store in tree_nodes:
            if node_obj_store.ID in y_positions and node_obj_store.points is not None:
                self.track_segment_layout_data[node_obj_store.ID] = (node_obj_store.points[:, 0],
                                                                     y_positions[node_obj_store.ID])

    def _set_empty_plot_title_if_needed(self, tree_nodes_check: List[TreeNode], root_id_for_title_check: int):
        is_visible_any_node = any(self.cell_visibility_main.get(tn_check.ID, True) for tn_check in tree_nodes_check)
        view_info_str = f"({self.current_view_type})"
        title_str = f"Tree for Root ID: {root_id_for_title_check} {view_info_str} "
        title_str += "(all elements hidden)" if not is_visible_any_node and tree_nodes_check else "(empty or no visible elements)"
        self.plot_item.setTitle(title_str)

    def _plot_track_segment_data(self, edges_to_plot: List[Edge], annotations_to_plot: List[Annotation],
                                 min_y_plot: float, max_y_plot: float):
        # Save current view range before redrawing
        saved_range = self._save_current_view_range()
        
        visibility_map = self.cell_visibility_main if self.cell_visibility_main is not None else {}
        min_x_coord_plot, max_x_coord_plot = float('inf'), float('-inf')
        did_plot_edge, did_plot_annotation = False, False
        for edge_item in edges_to_plot:
            draw_this_edge = (
                                     edge_item.is_connector and edge_item.connected_child_id is not None and visibility_map.get(
                                 edge_item.connected_child_id, True)) or \
                             (not edge_item.is_connector and edge_item.track_id is not None and visibility_map.get(
                                 edge_item.track_id, True))
            if not draw_this_edge: 
                continue
            did_plot_edge = True
            is_highlighted_edge = self.highlighted_track_id == edge_item.track_id and not edge_item.is_connector
            pen_style = Qt.DashLine if edge_item.is_connector else Qt.SolidLine
            pen_width = self.HIGHLIGHT_PEN_WIDTH if is_highlighted_edge else (
                self.CONNECTOR_PEN_WIDTH if edge_item.is_connector else self.DEFAULT_PEN_WIDTH)
            current_pen = pg.mkPen(color=edge_item.color, width=pen_width, style=pen_style)
            plot_data_item_ref = self.plot_item.plot(x=list(edge_item.x), y=list(edge_item.y), pen=current_pen)
            if plot_data_item_ref: 
                # plot() returns a list of items, assign metadata to each
                if isinstance(plot_data_item_ref, list):
                    for item in plot_data_item_ref:
                        item._lineage_tree_edge_meta = edge_item
                else:
                    plot_data_item_ref._lineage_tree_edge_meta = edge_item
            min_x_coord_plot = min(min_x_coord_plot, *edge_item.x);
            max_x_coord_plot = max(max_x_coord_plot, *edge_item.x)
        for ann_item in annotations_to_plot:
            ann_track_id_str = ann_item.label.split(" ")[0].replace("D", "").replace("P", "").split("[")[0].split("(")[
                0]
            ann_track_id_val = int(ann_track_id_str) if ann_track_id_str.isdigit() else None
            if ann_track_id_val is not None and not visibility_map.get(ann_track_id_val, True): continue
            did_plot_annotation = True
            # Anchor text to its bottom-center and shift slightly up to not cover the line
            text_plot_item = pg.TextItem(html=ann_item.html if ann_item.html else ann_item.label, anchor=(0.5, 1.0))
            if not ann_item.html: text_plot_item.setColor(ann_item.color)
            self.plot_item.addItem(text_plot_item);
            text_plot_item.setPos(ann_item.x, ann_item.y - 0.75)  # Position slightly above the line
            min_x_coord_plot = min(min_x_coord_plot, ann_item.x);
            max_x_coord_plot = max(max_x_coord_plot, ann_item.x)
        
        if did_plot_edge or did_plot_annotation:
            # Always set the range to ensure the plot shows the data
            if min_x_coord_plot == float('inf') or max_x_coord_plot == float('-inf'):
                # Fallback ranges if no data was plotted
                min_x_coord_plot, max_x_coord_plot = 0.0, 10.0
                min_y_plot, max_y_plot = 0.0, 1.0
            
            padding_x_val = (max_x_coord_plot - min_x_coord_plot) * 0.05 if max_x_coord_plot > min_x_coord_plot else 1.0
            padding_y_val = (max_y_plot - min_y_plot) * 0.05 if max_y_plot > min_y_plot else 1.0
            
            # Set the plot range explicitly
            self.plot_item.setXRange(min_x_coord_plot - padding_x_val, max_x_coord_plot + padding_x_val, padding=0)
            self.plot_item.setYRange(min_y_plot - padding_y_val, max_y_plot + padding_y_val, padding=0)
        else:
            # If no data was plotted, set a default range
            self.plot_item.setXRange(0, 10, padding=0)
            self.plot_item.setYRange(0, 1, padding=0)

    def _plot_node_view_data(self, node_items_to_plot: List[NodePlotData], line_items_to_plot: List[LinePlotData],
                             annotations_to_plot: List[Annotation], min_y_plot_nv: float, max_y_plot_nv: float):
        # Save current view range before redrawing
        saved_range = self._save_current_view_range()
        
        visibility_map_nv = self.cell_visibility_main if self.cell_visibility_main is not None else {}
        min_x_coord_plot_nv, max_x_coord_plot_nv = float('inf'), float('-inf')
        did_plot_node_or_line, did_plot_annotation_nv = False, False
        for line_item_nv in line_items_to_plot:
            if not line_item_nv.is_mitosis_connector and line_item_nv.track_id is not None and not visibility_map_nv.get(
                    line_item_nv.track_id, True): continue
            did_plot_node_or_line = True
            is_hl_line = not line_item_nv.is_mitosis_connector and line_item_nv.track_id == self.highlighted_track_id
            pen_width_line = self.NODE_CONNECTOR_LINE_WIDTH if line_item_nv.is_mitosis_connector else (
                self.HIGHLIGHT_NODE_LINE_WIDTH if is_hl_line else self.NODE_LINE_WIDTH)
            line_style = Qt.DashLine if line_item_nv.is_mitosis_connector else Qt.SolidLine
            current_pen_line = pg.mkPen(color=line_item_nv.color, width=pen_width_line, style=line_style)
            self.plot_item.plot(x=line_item_nv.x_coords, y=line_item_nv.y_coords, pen=current_pen_line)
            min_x_coord_plot_nv = min(min_x_coord_plot_nv, *line_item_nv.x_coords);
            max_x_coord_plot_nv = max(max_x_coord_plot_nv, *line_item_nv.x_coords)
        for node_item_nv in node_items_to_plot:
            if not visibility_map_nv.get(node_item_nv.track_id, True): continue
            did_plot_node_or_line = True
            is_hl_node = self.highlighted_track_id == node_item_nv.track_id
            node_actual_size = self.HIGHLIGHT_NODE_SIZE if is_hl_node else self.NODE_SIZE
            node_pen_color = (255, 255, 0) if is_hl_node else (200, 200, 200)
            scatter_item_ref = pg.ScatterPlotItem(size=node_actual_size, pen=pg.mkPen(node_pen_color, width=1),
                                                  brush=pg.mkBrush(node_item_nv.color))
            scatter_item_ref.addPoints(x=node_item_nv.points[:, 0], y=node_item_nv.points[:, 1])
            self.plot_item.addItem(scatter_item_ref);
            node_item_nv.pg_scatter_item = scatter_item_ref
            min_x_coord_plot_nv = min(min_x_coord_plot_nv, *node_item_nv.points[:, 0]);
            max_x_coord_plot_nv = max(max_x_coord_plot_nv, *node_item_nv.points[:, 0])
        for ann_item_nv in annotations_to_plot:
            ann_track_id_str_nv = \
                ann_item_nv.label.split(" ")[0].replace("D", "").replace("P", "").split("[")[0].split("(")[0]
            ann_track_id_val_nv = int(ann_track_id_str_nv) if ann_track_id_str_nv.isdigit() else None
            if ann_track_id_val_nv is not None and not visibility_map_nv.get(ann_track_id_val_nv, True): continue
            did_plot_annotation_nv = True
            # Anchor text to its bottom-left and shift slightly up to not cover the nodes
            text_plot_item_nv = pg.TextItem(html=ann_item_nv.html if ann_item_nv.html else ann_item_nv.label,
                                            anchor=(0, 1))
            if not ann_item_nv.html: text_plot_item_nv.setColor(ann_item_nv.color)
            self.plot_item.addItem(text_plot_item_nv);
            text_plot_item_nv.setPos(ann_item_nv.x + 5,
                                     ann_item_nv.y - 0.75)  # Position right of first node and above the line
            min_x_coord_plot_nv = min(min_x_coord_plot_nv, ann_item_nv.x);
            max_x_coord_plot_nv = max(max_x_coord_plot_nv, ann_item_nv.x)
        if did_plot_node_or_line or did_plot_annotation_nv:
            # Only set range if we don't have a saved range to restore
            if not saved_range:
                padding_x_val_nv = (
                                       max_x_coord_plot_nv - min_x_coord_plot_nv) * 0.05 if max_x_coord_plot_nv > min_x_coord_plot_nv else 1.0
                padding_y_val_nv = (max_y_plot_nv - min_y_plot_nv) * 0.05 if max_y_plot_nv > min_y_plot_nv else 1.0
                self.plot_item.setXRange(min_x_coord_plot_nv - padding_x_val_nv, max_x_coord_plot_nv + padding_x_val_nv,
                                         padding=0)
                self.plot_item.setYRange(min_y_plot_nv - padding_y_val_nv, max_y_plot_nv + padding_y_val_nv, padding=0)
            else:
                # Restore the saved view range
                self._restore_view_range(saved_range)

    def _getNodeIdAtScenePos(self, view_pos: QPointF) -> Optional[int]:
        # For Node View, use the existing node_view_layout_data
        if self.current_view_type == 'Node View' and self.node_view_layout_data:
            for track_id_nv_click, (time_points_nv_click, y_layout_nv_click) in self.node_view_layout_data.items():
                if not self.cell_visibility_main.get(track_id_nv_click, True) or time_points_nv_click.size == 0: continue
                if abs(view_pos.y() - y_layout_nv_click) < self.NODE_CLICK_Y_TOLERANCE_VIEW_UNITS and \
                        time_points_nv_click[0] <= view_pos.x() <= time_points_nv_click[-1]:
                    return track_id_nv_click
        
        # For Track Segments and Class Type views, use the track segment layout data
        elif self.current_view_type in ['Track Segments', 'Class Type'] and self.track_segment_layout_data:
            for track_id_click, (time_points_click, y_layout_click) in self.track_segment_layout_data.items():
                if not self.cell_visibility_main.get(track_id_click, True) or time_points_click.size == 0: continue
                if abs(view_pos.y() - y_layout_click) < self.NODE_CLICK_Y_TOLERANCE_VIEW_UNITS and \
                        time_points_click[0] <= view_pos.x() <= time_points_click[-1]:
                    return track_id_click
        
        return None

    def _getNodeAndFrameAtScenePos(self, view_pos: QPointF) -> Optional[Tuple[int, int]]:
        """Get node ID and frame at scene position."""
        # For Node View, use the existing node_view_layout_data
        if self.current_view_type == 'Node View' and self.node_view_layout_data:
            for track_id_nv_click, (time_points_nv_click, y_layout_nv_click) in self.node_view_layout_data.items():
                if not self.cell_visibility_main.get(track_id_nv_click, True) or time_points_nv_click.size == 0: continue
                if abs(view_pos.y() - y_layout_nv_click) < self.NODE_CLICK_Y_TOLERANCE_VIEW_UNITS and \
                        time_points_nv_click[0] <= view_pos.x() <= time_points_nv_click[-1]:
                    # Find the closest time point
                    closest_idx = np.argmin(np.abs(time_points_nv_click - view_pos.x()))
                    closest_frame = int(time_points_nv_click[closest_idx])
                    return track_id_nv_click, closest_frame
        
        # For Track Segments and Class Type views, use the track segment layout data
        elif self.current_view_type in ['Track Segments', 'Class Type'] and self.track_segment_layout_data:
            for track_id_click, (time_points_click, y_layout_click) in self.track_segment_layout_data.items():
                if not self.cell_visibility_main.get(track_id_click, True) or time_points_click.size == 0: continue
                if abs(view_pos.y() - y_layout_click) < self.NODE_CLICK_Y_TOLERANCE_VIEW_UNITS and \
                        time_points_click[0] <= view_pos.x() <= time_points_click[-1]:
                    # Find the closest time point
                    closest_idx = np.argmin(np.abs(time_points_click - view_pos.x()))
                    closest_frame = int(time_points_click[closest_idx])
                    return track_id_click, closest_frame
        
        return None

    def customMousePressEvent(self, event: QMouseEvent):
        view_pos = self.plot_item.vb.mapSceneToView(event.scenePos())
        click_info_tuple = self._getNodeAndFrameAtScenePos(view_pos)
        
        # Always emit node click signal if we detect a node
        if click_info_tuple:
            clicked_node_id_val, frame_on_node_val = click_info_tuple
            self.nodeClickedInView.emit(clicked_node_id_val, frame_on_node_val)
        
        # For left button clicks, prepare for potential drag but don't start immediately
        if event.button() == Qt.LeftButton and (self.current_view_type == 'Node View' or self.current_view_type == 'Track Segments'):
            node_id_for_drag_op = self._getNodeIdAtScenePos(view_pos)
            if node_id_for_drag_op is not None:
                # Store potential drag info but don't start drag yet
                self.potential_drag_node_id = node_id_for_drag_op
                self.drag_start_scene_pos = view_pos
                self.drag_started = False
                event.accept()
                return
            else:
                self.potential_drag_node_id = None
                self.drag_started = False
        
        # Don't ignore the event - let it propagate for proper handling
        event.accept()

    def customMouseMoveEvent(self, event: QMouseEvent):
        view_pos_move = self.plot_item.vb.mapSceneToView(event.pos())
        
        # Handle drag operation
        if self.dragged_node_id and self.drag_start_scene_pos:
            # Update drag line
            if self.temp_drag_line_item and self.dragged_node_id in self.node_view_layout_data:
                dragged_node_times_move, dragged_node_y_layout_move = self.node_view_layout_data[self.dragged_node_id]
                if dragged_node_times_move.size > 0:
                    drag_line_start_x_pos_move = dragged_node_times_move[0] + (
                            dragged_node_times_move[-1] - dragged_node_times_move[0]) / 2 if len(
                        dragged_node_times_move) > 1 else dragged_node_times_move[0]
                    self.temp_drag_line_item.setData(x=[drag_line_start_x_pos_move, view_pos_move.x()],
                                                     y=[dragged_node_y_layout_move, view_pos_move.y()])
            
            # Check for potential drop target
            target_node_id = self._getNodeIdAtScenePos(view_pos_move)
            if target_node_id is not None and target_node_id != self.dragged_node_id:
                # Get drop frame
                clicked_info = self._getNodeAndFrameAtScenePos(view_pos_move)
                drop_frame = clicked_info[1] if clicked_info and clicked_info[0] == target_node_id else -1
                
                # Create drop zone highlight
                if drop_frame != -1:
                    self._create_drop_zone_highlight(target_node_id, drop_frame)
                    self._create_drag_preview_text(self.dragged_node_id, target_node_id, drop_frame)
                else:
                    self._remove_drop_zone_highlight()
                    self._remove_drag_preview_text()
            else:
                self._remove_drop_zone_highlight()
                self._remove_drag_preview_text()
            
            event.accept()
            return
        
        # Check if we should start a drag operation
        if self.potential_drag_node_id and self.drag_start_scene_pos and not self.drag_started:
            # Calculate distance moved
            distance = ((view_pos_move.x() - self.drag_start_scene_pos.x()) ** 2 + 
                       (view_pos_move.y() - self.drag_start_scene_pos.y()) ** 2) ** 0.5
            
            if distance > self.DRAG_THRESHOLD_DISTANCE:
                # Start drag operation
                self.dragged_node_id = self.potential_drag_node_id
                self.drag_started = True
                
                # Create drag line
                if self.temp_drag_line_item is None:
                    self.temp_drag_line_item = pg.PlotDataItem(
                        pen=pg.mkPen(DRAG_LINE_COLOR, style=Qt.DashLine, width=2))
                    self.plot_item.addItem(self.temp_drag_line_item)
                
                # Set initial drag line position
                if self.dragged_node_id in self.node_view_layout_data:
                    dragged_node_times_drag, dragged_node_y_layout_drag = self.node_view_layout_data[self.dragged_node_id]
                    if dragged_node_times_drag.size > 0:
                        drag_line_start_x_pos = dragged_node_times_drag[0] + (
                                dragged_node_times_drag[-1] - dragged_node_times_drag[0]) / 2 if len(
                            dragged_node_times_drag) > 1 else dragged_node_times_drag[0]
                        self.temp_drag_line_item.setData(x=[drag_line_start_x_pos, view_pos_move.x()],
                                                         y=[dragged_node_y_layout_drag, view_pos_move.y()])
                        self.temp_drag_line_item.setVisible(True)
                    else:
                        self.temp_drag_line_item.setVisible(False)
                
                # Remove any existing hover highlight
                self._remove_hover_highlight()
                event.accept()
                return
        
        # Handle hover highlighting and tooltips when not dragging
        if not self.dragged_node_id and not self.drag_started:
            # Show tooltips in Track Segments and Node View
            if self.current_view_type == 'Track Segments':
                seg_info = self._getNodeAndFrameAtScenePos(view_pos_move)
                if seg_info:
                    track_id_hover, frame_hover = seg_info
                    if track_id_hover != self.hovered_node_id:
                        self.hovered_node_id = track_id_hover
                        self._create_tooltip(track_id_hover, view_pos_move)
                else:
                    if self.hovered_node_id is not None:
                        self.hovered_node_id = None
                        self._remove_tooltip()
            # Node View: show both hover highlight and tooltip
            elif self.current_view_type == 'Node View':
                node_id_hover = self._getNodeIdAtScenePos(view_pos_move)
                if node_id_hover is not None:
                    if node_id_hover != self.hovered_node_id:
                        self.hovered_node_id = node_id_hover
                        self._create_hover_highlight(node_id_hover)
                        self._create_tooltip(node_id_hover, view_pos_move)
                else:
                    if self.hovered_node_id is not None:
                        self.hovered_node_id = None
                        self._remove_hover_highlight()
                        self._remove_tooltip()
            # Class Type view: show tooltips
            elif self.current_view_type == 'Class Type':
                seg_info = self._getNodeAndFrameAtScenePos(view_pos_move)
                if seg_info:
                    track_id_hover, frame_hover = seg_info
                    if track_id_hover != self.hovered_node_id:
                        self.hovered_node_id = track_id_hover
                        self._create_tooltip(track_id_hover, view_pos_move)
                else:
                    if self.hovered_node_id is not None:
                        self.hovered_node_id = None
                        self._remove_tooltip()
            else:
                # Remove tooltip and highlight if not hovering over anything
                if self.hovered_node_id is not None:
                    self.hovered_node_id = None
                    self._remove_hover_highlight()
                    self._remove_tooltip()
        
        event.accept()

    def _is_drop_in_middle_of_track(self, target_node_id: int, drop_frame: int) -> bool:
        """Check if the drop position is in the middle of the target track (not at start or end)."""
        if target_node_id not in self.node_view_layout_data:
            return False
        
        time_points, _ = self.node_view_layout_data[target_node_id]
        if time_points.size < 3:  # Need at least 3 frames to have a "middle"
            return False
        
        start_frame = time_points[0]
        end_frame = time_points[-1]
        
        # Consider it "middle" if it's not within 1 frame of start or end
        return start_frame + 1 < drop_frame < end_frame - 1

    def _get_drop_zone_type(self, target_node_id: int, drop_frame: int) -> str:
        """Get the type of drop zone: 'start', 'middle', or 'end'."""
        if target_node_id not in self.node_view_layout_data:
            return 'none'
        
        time_points, _ = self.node_view_layout_data[target_node_id]
        if time_points.size < 2:
            return 'none'
        
        start_frame = time_points[0]
        end_frame = time_points[-1]
        
        if drop_frame <= start_frame + 1:
            return 'start'
        elif drop_frame >= end_frame - 1:
            return 'end'
        else:
            return 'middle'

    def _create_hover_highlight(self, node_id: int, color: Tuple[int, int, int] = (255, 255, 0)):
        """Create a highlight around the hovered node."""
        if self.hover_highlight_item:
            self.plot_item.removeItem(self.hover_highlight_item)
        
        if node_id in self.node_view_layout_data:
            time_points, y_pos = self.node_view_layout_data[node_id]
            if time_points.size > 0:
                # Create a rectangle highlight
                x_start, x_end = time_points[0], time_points[-1]
                y_bottom = y_pos - 2
                y_top = y_pos + 2
                
                # Create rectangle points
                rect_x = [x_start, x_end, x_end, x_start, x_start]
                rect_y = [y_bottom, y_bottom, y_top, y_top, y_bottom]
                
                self.hover_highlight_item = pg.PlotDataItem(
                    x=rect_x, y=rect_y,
                    pen=pg.mkPen(color, width=3, style=Qt.DashLine),
                    fillLevel=None
                )
                self.plot_item.addItem(self.hover_highlight_item)

    def _remove_hover_highlight(self):
        """Remove the hover highlight."""
        if self.hover_highlight_item:
            self.plot_item.removeItem(self.hover_highlight_item)
            self.hover_highlight_item = None

    def _create_drop_zone_highlight(self, target_node_id: int, drop_frame: int):
        """Create a highlight for the drop zone."""
        if self.drop_zone_highlight:
            self.plot_item.removeItem(self.drop_zone_highlight)
        
        if target_node_id in self.node_view_layout_data:
            time_points, y_pos = self.node_view_layout_data[target_node_id]
            if time_points.size > 0:
                zone_type = self._get_drop_zone_type(target_node_id, drop_frame)
                
                # Color based on zone type
                if zone_type == 'start':
                    color = (0, 255, 0)  # Green for start
                elif zone_type == 'middle':
                    color = (255, 255, 0)  # Yellow for middle
                elif zone_type == 'end':
                    color = (255, 0, 0)  # Red for end
                else:
                    color = (128, 128, 128)  # Gray for none
                
                # Create a vertical line at the drop position
                y_bottom = y_pos - 3
                y_top = y_pos + 3
                
                self.drop_zone_highlight = pg.PlotDataItem(
                    x=[drop_frame, drop_frame], y=[y_bottom, y_top],
                    pen=pg.mkPen(color, width=4)
                )
                self.plot_item.addItem(self.drop_zone_highlight)

    def _remove_drop_zone_highlight(self):
        """Remove the drop zone highlight."""
        if self.drop_zone_highlight:
            self.plot_item.removeItem(self.drop_zone_highlight)
            self.drop_zone_highlight = None

    def _create_drag_preview_text(self, dragged_id: int, target_id: int, drop_frame: int):
        """Create preview text showing what operation will be performed."""
        if self.drag_preview_text:
            self.plot_item.removeItem(self.drag_preview_text)
        
        zone_type = self._get_drop_zone_type(target_id, drop_frame)
        
        if zone_type == 'middle':
            operation_text = f"Auto-break {target_id} at frame {drop_frame}\nInsert {dragged_id} as daughter"
            color = (255, 255, 0)  # Yellow
        elif zone_type == 'start':
            operation_text = f"Insert {dragged_id} as child of {target_id}"
            color = (0, 255, 0)  # Green
        elif zone_type == 'end':
            operation_text = f"Insert {dragged_id} as parent of {target_id}"
            color = (255, 0, 0)  # Red
        else:
            operation_text = f"Choose operation for {dragged_id} → {target_id}"
            color = (128, 128, 128)  # Gray
        
        # Position text near the drop position
        if target_id in self.node_view_layout_data:
            _, y_pos = self.node_view_layout_data[target_id]
            self.drag_preview_text = pg.TextItem(
                text=operation_text,
                color=color,
                anchor=(0, 0)
            )
            self.drag_preview_text.setPos(drop_frame + 2, y_pos + 2)
            self.plot_item.addItem(self.drag_preview_text)

    def _remove_drag_preview_text(self):
        """Remove the drag preview text."""
        if self.drag_preview_text:
            self.plot_item.removeItem(self.drag_preview_text)
            self.drag_preview_text = None

    def _calculate_doubling_time(self, node_id: int) -> Optional[float]:
        """Calculate the doubling time for a track segment based on its duration."""
        if node_id not in self.node_view_layout_data:
            return None
        
        time_points, _ = self.node_view_layout_data[node_id]
        if time_points.size < 2:
            return None
        
        # Calculate duration in frames
        duration_frames = time_points[-1] - time_points[0] + 1
        
        # Convert frames to hours using timing parameters
        duration_hours = duration_frames / self.frames_per_hour
        
        # Get trajectory data for this node
        if self.trj_data is not None and not self.trj_data.empty:
            node_data = self.trj_data[self.trj_data['particle'] == node_id]
            if len(node_data) >= 2:
                # Calculate area change if available
                if 'area' in node_data.columns:
                    initial_area = node_data.iloc[0]['area']
                    final_area = node_data.iloc[-1]['area']
                    if initial_area > 0 and final_area > initial_area:
                        # Doubling time = duration / log2(final_area / initial_area)
                        area_ratio = final_area / initial_area
                        if area_ratio > 1:
                            doubling_time_hours = duration_hours / np.log2(area_ratio)
                            return doubling_time_hours
                
                # Fallback: estimate based on duration (assuming typical doubling time)
                # This is a rough estimate - in practice you'd want actual area data
                return duration_hours / 2.0  # Rough estimate in hours
        
        return None

    def _create_tooltip(self, node_id: int, view_pos: QPointF):
        """Create a tooltip showing track information including doubling time."""
        if self.tooltip_text_item:
            self.plot_item.removeItem(self.tooltip_text_item)
        
        # Get track information from trajectory data
        if self.trj_data is not None and not self.trj_data.empty:
            node_data = self.trj_data[self.trj_data['particle'] == node_id]
            if len(node_data) > 0:
                # Basic track info
                start_frame = int(node_data['frame'].min())
                end_frame = int(node_data['frame'].max())
                duration_frames = end_frame - start_frame + 1
                duration_hours = duration_frames / self.frames_per_hour
                
                # Calculate doubling time
                doubling_time = self._calculate_doubling_time(node_id)
                
                # Create tooltip text with better HTML styling
                tooltip_lines = [
                    f'<span style="font-weight: bold; color: #FFD700; font-size: 12px;">Track ID: {node_id}</span>',
                    f'<span style="color: #FFFFFF; font-size: 11px;">Duration: {duration_frames} frames ({duration_hours:.2f} hours)</span>',
                    f'<span style="color: #FFFFFF; font-size: 11px;">Frames: {start_frame} → {end_frame}</span>'
                ]
                
                if doubling_time is not None:
                    tooltip_lines.append(f'<span style="color: #90EE90; font-size: 11px;">Est. Doubling Time: {doubling_time:.2f} hours</span>')
                
                # Add state information if available
                if self.track_states_main and node_id in self.track_states_main:
                    state = self.track_states_main[node_id]
                    if state and state not in ["N/A", "Unclassified"]:
                        tooltip_lines.append(f'<span style="color: #87CEEB; font-size: 11px;">State: {state}</span>')
                
                # Add parent/child info if available
                if self.ancestry_data_main and isinstance(self.ancestry_data_main, dict) and node_id in self.ancestry_data_main:
                    parents = self.ancestry_data_main[node_id]
                    if parents:
                        tooltip_lines.append(f'<span style="color: #FFB6C1; font-size: 11px;">Parents: {", ".join(map(str, parents))}</span>')
                
                if self.lineage_data_main and node_id in self.lineage_data_main:
                    children = self.lineage_data_main[node_id]
                    if children:
                        tooltip_lines.append(f'<span style="color: #98FB98; font-size: 11px;">Children: {", ".join(map(str, children))}</span>')
                
                tooltip_text = "<br>".join(tooltip_lines)
                
                # Create tooltip item with much better styling for visibility
                self.tooltip_text_item = pg.TextItem(
                    html=tooltip_text,
                    anchor=(0, 1),  # Top-left anchor
                    border=pg.mkPen((255, 255, 255), width=2),  # White border for contrast
                    fill=pg.mkBrush((20, 20, 20, 240)),  # Very dark, highly opaque background
                )
                
                # Position tooltip very close to mouse cursor
                tooltip_x = view_pos.x() + 5
                tooltip_y = view_pos.y() + 5
                self.tooltip_text_item.setPos(tooltip_x, tooltip_y)
                self.plot_item.addItem(self.tooltip_text_item)
                self.tooltip_text_item.setZValue(1000)  # Ensure tooltip is on top

    def _remove_tooltip(self):
        """Remove the tooltip."""
        if self.tooltip_text_item:
            self.plot_item.removeItem(self.tooltip_text_item)
            self.tooltip_text_item = None

    def customMouseReleaseEvent(self, event: QMouseEvent):
        if self.dragged_node_id and self.drag_start_scene_pos:
            view_pos_release = self.plot_item.vb.mapSceneToView(event.scenePos())
            
            # Clean up visual feedback
            if self.temp_drag_line_item:
                self.plot_item.removeItem(self.temp_drag_line_item)
                self.temp_drag_line_item = None
            self._remove_drop_zone_highlight()
            self._remove_drag_preview_text()
            self._remove_tooltip()

            target_node_id_drop = self._getNodeIdAtScenePos(view_pos_release)

            if target_node_id_drop is not None and target_node_id_drop != self.dragged_node_id:
                clicked_info_on_target = self._getNodeAndFrameAtScenePos(view_pos_release)
                insert_frame_on_target = -1
                if clicked_info_on_target and clicked_info_on_target[0] == target_node_id_drop:
                    insert_frame_on_target = clicked_info_on_target[1]

                # Check if dropping in the middle of the target track
                if insert_frame_on_target != -1 and self._is_drop_in_middle_of_track(target_node_id_drop, insert_frame_on_target):
                    # Automatically use "Insert Dragged as Child & Split Target's Tail" operation
                    logging.info(f"Auto-breaking target track {target_node_id_drop} at frame {insert_frame_on_target} and inserting dragged track {self.dragged_node_id} as daughter")
                    self.insertAndSplitRequested.emit(self.dragged_node_id, target_node_id_drop, insert_frame_on_target)
                else:
                    # Use the unified LineageOperationDialog for other cases
                    dialog = LineageOperationDialog(self.dragged_node_id, target_node_id_drop, insert_frame_on_target, self)
                    if dialog.exec_() == QDialog.Accepted:
                        chosen_operation_name = dialog.get_selected_operation()
                        if chosen_operation_name != "Cancel":
                            if chosen_operation_name == "Insert Dragged as Child & Split Target's Tail":
                                if insert_frame_on_target != -1:
                                    self.insertAndSplitRequested.emit(self.dragged_node_id, target_node_id_drop,
                                                                      insert_frame_on_target)
                                else:
                                    QMessageBox.warning(self, "Operation Error",
                                                        "Could not determine exact frame on target for insertion. Operation cancelled.")
                            else:  # Other standard operations
                                self.nodeOperationRequested.emit(self.dragged_node_id, target_node_id_drop,
                                                                 chosen_operation_name)

            self.dragged_node_id = None
            self.drag_start_scene_pos = None
            self.potential_drag_node_id = None
            self.drag_started = False
            event.accept()
        else:
            # Clean up potential drag state if no drag was started
            self.potential_drag_node_id = None
            self.drag_started = False
            event.ignore()

    def customMouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle double-click events for node operations."""
        if event.button() != Qt.LeftButton:
            return
        
        scene_pos = self.plot_item.vb.mapSceneToView(event.pos())
        node_and_frame = self._getNodeAndFrameAtScenePos(scene_pos)
        
        if node_and_frame:
            track_id, frame_index = node_and_frame
            self.nodeBreakRequested.emit(track_id, frame_index)
        else:
            # No node detected at click position
            pass

    def _save_current_view_range(self):
        """Save the current view range to restore later."""
        try:
            view_range = self.plot_item.viewRange()
            return {
                'x_range': view_range[0],
                'y_range': view_range[1]
            }
        except:
            return None

    def _restore_view_range(self, saved_range):
        """Restore a previously saved view range."""
        if saved_range and 'x_range' in saved_range and 'y_range' in saved_range:
            try:
                self.plot_item.setXRange(*saved_range['x_range'], padding=0)
                self.plot_item.setYRange(*saved_range['y_range'], padding=0)
                return True
            except:
                pass
        return False