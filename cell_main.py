# cell_main.py (Modified for YAML Integration)
# Main application file for Cell Tracking Editor with YAML configuration support.
# V11 (Gemini - This Update): Removed "Non-Mitosis" button and adjusted layout for new selection buttons.
# V12 (Gemini - This Update): Reorganized parameter UI groups for better clarity.
# V13 (Gemini - Optimization): Implemented lazy loading for the parameter tree to improve startup speed.
# V14 (Gemini - Diagnostics): Added detailed timing logs to the startup sequence.

import pyqtgraph as pg
from pyqtgraph.dockarea import *
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QPointF, QTimer
from PyQt5.QtWidgets import (QFileDialog, QApplication, QProgressDialog, QMainWindow,
                             QMessageBox, QPushButton, QTextEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QCheckBox, QWidget,
                             QHBoxLayout, QAbstractItemView, QVBoxLayout, QDialog,
                             QDialogButtonBox, QLabel, QComboBox, QLineEdit, QInputDialog,
                             QAction, QSizePolicy, QGridLayout, QFormLayout, QSpinBox)
from PyQt5.QtGui import QTextCursor, QKeySequence
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
import os
import time
import sys
import yaml
import atexit
import traceback
import copy
import weakref
import json
import matplotlib.pyplot as plt
from os.path import join

# Define available colormap options for dropdown lists
COLORMAP_OPTIONS = [
    'plasma', 'viridis', 'inferno', 'magma', 'turbo', 'cividis',
    'coolwarm', 'RdBu', 'RdYlBu', 'Spectral', 'rainbow', 'jet',
    'hot', 'cool', 'spring', 'summer', 'autumn', 'winter',
    'bone', 'pink', 'copper', 'gray', 'binary', 'gist_earth',
    'gist_rainbow', 'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2',
    'ocean', 'terrain', 'afmhot', 'brg', 'hsv', 'nipy_spectral',
    'gist_ncar', 'tab10', 'tab20', 'tab20b', 'tab20c'
]

# --- STARTUP TIMING ---
startup_time_start = time.perf_counter()

# --- Robust Path Setup ---
try:
    # Get the directory of the currently running script
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is not defined, happens in some environments (e.g., interactive)
    current_dir = os.getcwd()

# Add the script's directory and its parent to Python's path
# This helps resolve imports when run as a script from a subfolder
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure a logger specific to this module
module_logger = logging.getLogger(__name__)
module_logger.info(f"TUI Tracker starting. Current working directory: {os.getcwd()}")
module_logger.info(f"Python sys.path includes: {current_dir}")

# Import the ILP Optimizer module
try:
    from cell_ilp_optimizer import run_optimization, SKOPT_AVAILABLE
except ImportError:
    def run_optimization(*args, **kwargs):
        logging.error("cell_ilp_optimizer.py module not found.")
        QMessageBox.critical(None, "Module Not Found",
                             "The ILP optimizer module (cell_ilp_optimizer.py) could not be found.")
        return None, None, None
    SKOPT_AVAILABLE = False

# Import ILP components and check for Gurobi
try:
    from cell_tracking_ilp import ILP_AVAILABLE, GUROBI_AVAILABLE
except ImportError as e:
    module_logger.error("Failed to import 'cell_tracking_ilp'. This module is required for ILP tracking.")
    module_logger.error(f"ImportError: {e}")
    module_logger.error(f"Traceback: {traceback.format_exc()}")
    ILP_AVAILABLE = False
    GUROBI_AVAILABLE = False

# Import functions/classes with error handling
missing_modules = []

try:
    from cell_io import read_img_sequence, save_results
except ImportError as e:
    module_logger.warning(f"Could not import cell_io: {e}")
    missing_modules.append("cell_io")


    # Create dummy functions
    def read_img_sequence(*args, **kwargs):
        raise NotImplementedError("cell_io module not available")


    def save_results(*args, **kwargs):
        raise NotImplementedError("cell_io module not available")

try:
    from cell_drawing import prepare_mask_colors
except ImportError as e:
    module_logger.warning(f"Could not import cell_drawing: {e}")
    missing_modules.append("cell_drawing")


    def prepare_mask_colors(*args, **kwargs):
        return None, [], [], 0

try:
    from cell_tracking import initialize_next_daughter_id, get_new_daughter_id
except ImportError as e:
    module_logger.warning(f"Could not import cell_tracking: {e}")
    missing_modules.append("cell_tracking")


    def initialize_next_daughter_id(*args, **kwargs):
        return 1


    def get_new_daughter_id(*args, **kwargs):
        return 1

try:
    from lineage_tree import LineageTreeWidget
except ImportError as e:
    module_logger.warning(f"Could not import lineage_tree: {e}")
    missing_modules.append("lineage_tree")


    # Create a dummy LineageTreeWidget
    class LineageTreeWidget(QWidget):
        # Define the signals that the real widget would have
        nodeOperationRequested = pyqtSignal(int, int, str)
        nodeClickedInView = pyqtSignal(int, int)
        nodeBreakRequested = pyqtSignal(int, int)
        insertAndSplitRequested = pyqtSignal(int, int, int)
        splitAndFuseHeadRequested = pyqtSignal(int, int, int)

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("LineageTreeWidget not available"))
            module_logger.warning("Using dummy LineageTreeWidget")

# Import new refactored modules
try:
    import cell_undoredo
except ImportError as e:
    module_logger.warning(f"Could not import cell_undoredo: {e}")
    missing_modules.append("cell_undoredo")


    # Create dummy module
    class DummyUndoRedo:
        @staticmethod
        def update_undo_redo_actions_enabled_state(*args):
            pass

        @staticmethod
        def undo_last_operation(*args):
            pass

        @staticmethod
        def redo_last_operation(*args):
            pass


    cell_undoredo = DummyUndoRedo()

try:
    import cell_file_operations
except ImportError as e:
    module_logger.warning(f"Could not import cell_file_operations: {e}")
    missing_modules.append("cell_file_operations")


    class DummyFileOps:
        @staticmethod
        def load_data_from_folder(*args):
            QMessageBox.warning(None, "Error", "cell_file_operations module not available")

        @staticmethod
        def load_previous_results(*args):
            QMessageBox.warning(None, "Error", "cell_file_operations module not available")

        @staticmethod
        def perform_batch_tracking(*args):
            QMessageBox.warning(None, "Error", "cell_file_operations module not available")


    cell_file_operations = DummyFileOps()

try:
    import cell_tracking_orchestrator
except ImportError as e:
    module_logger.warning(f"Could not import cell_tracking_orchestrator: {e}")
    missing_modules.append("cell_tracking_orchestrator")


    class DummyTrackingOrchestrator:
        @staticmethod
        def initiate_cell_tracking(*args):
            QMessageBox.warning(None, "Error", "cell_tracking_orchestrator module not available")


    cell_tracking_orchestrator = DummyTrackingOrchestrator()

try:
    import cell_lineage_operations
except ImportError as e:
    module_logger.warning(f"Could not import cell_lineage_operations: {e}")
    missing_modules.append("cell_lineage_operations")


    class DummyLineageOps:
        @staticmethod
        def handle_relink_visible_tracks(*args):
            QMessageBox.warning(None, "Error", "cell_lineage_operations module not available")

        @staticmethod
        def handle_merge_track_with_parent(*args):
            QMessageBox.warning(None, "Error", "cell_lineage_operations module not available")

        @staticmethod
        def process_node_operation_request(*args):
            pass

        @staticmethod
        def process_node_break_request(*args):
            pass

        @staticmethod
        def process_insert_dragged_and_split_target(*args):
            pass

        @staticmethod
        def process_split_and_fuse_head_request(*args):
            pass


    cell_lineage_operations = DummyLineageOps()

try:
    import cell_ui_callbacks
except ImportError as e:
    module_logger.warning(f"Could not import cell_ui_callbacks: {e}")
    missing_modules.append("cell_ui_callbacks")


    class DummyUICallbacks:
        @staticmethod
        def handle_update_view_for_current_frame(*args):
            pass

        @staticmethod
        def handle_time_changed_raw_img(*args):
            pass

        @staticmethod
        def handle_time_changed_mask(*args):
            pass

        @staticmethod
        def setup_hover_detection(*args):
            pass

        @staticmethod
        def handle_select_all_clicked(*args):
            pass

        @staticmethod
        def handle_select_none_clicked(*args):
            pass

        @staticmethod
        def handle_select_complete_clicked(*args):
            pass

        @staticmethod
        def handle_select_mitosis_tracks_clicked(*args):
            pass

        @staticmethod
        def handle_select_fusion_tracks_clicked(*args):
            pass

        @staticmethod
        def handle_select_singular_tracks_clicked(*args):
            pass

        @staticmethod
        def handle_calculate_stats_clicked(*args):
            pass

        @staticmethod
        def handle_evaluate_tracking_clicked(*args):
            pass

        @staticmethod
        def handle_cell_table_item_changed(*args):
            pass

        @staticmethod
        def handle_cell_table_cell_clicked(*args):
            pass

        @staticmethod
        def handle_lineage_node_clicked_in_main(*args):
            pass


    cell_ui_callbacks = DummyUICallbacks()

# Log missing modules
if missing_modules:
    module_logger.warning(f"Missing modules: {missing_modules}")
    module_logger.info("TUI Tracker will run with limited functionality")
else:
    module_logger.info("All modules imported successfully")

module_logger.info(f"--- TIME: Imports completed in {time.perf_counter() - startup_time_start:.4f} seconds ---")


# ULTRA-SAFE LOGGING SOLUTION - No Qt inheritance for handlers
class ThreadSafeLogCapture:
    """Thread-safe log capture that doesn't inherit from QObject"""

    def __init__(self):
        self.log_messages = []
        self.is_active = True
        self.max_messages = 1000  # Limit memory usage
        self.last_message_hash = None  # Track last message to prevent duplicates

    def add_message(self, message, level, timestamp):
        """Add a message to the capture buffer"""
        if self.is_active and len(self.log_messages) < self.max_messages:
            # Create a hash of the message to check for duplicates
            message_hash = hash(f"{timestamp}_{level}_{message}")
            if message_hash != self.last_message_hash:
                self.log_messages.append((timestamp, level, message))
                self.last_message_hash = message_hash

    def get_messages(self):
        """Get all captured messages"""
        return self.log_messages[:]

    def clear(self):
        """Clear captured messages"""
        self.log_messages.clear()
        self.last_message_hash = None

    def shutdown(self):
        """Safely shutdown the capture"""
        self.is_active = False


class NonQtLogHandler(logging.Handler):
    """Log handler that doesn't inherit from QObject to avoid Qt lifecycle issues"""

    def __init__(self, log_capture, widget_callback=None):
        super().__init__()
        self.log_capture = log_capture
        self.widget_callback = widget_callback
        self.is_active = True

        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Register for early cleanup - BEFORE Qt cleanup
        atexit.register(self.early_cleanup)

    def early_cleanup(self):
        """Early cleanup that runs before Qt shutdown"""
        if self.is_active:
            self.is_active = False
            try:
                # Remove from all loggers immediately
                for logger_name in list(logging.Logger.manager.loggerDict.keys()):
                    try:
                        logger = logging.getLogger(logger_name)
                        if self in logger.handlers:
                            logger.removeHandler(self)
                    except:
                        pass

                # Remove from root logger
                root_logger = logging.getLogger()
                if self in root_logger.handlers:
                    root_logger.removeHandler(self)

                # Shutdown capture
                if self.log_capture:
                    self.log_capture.shutdown()

            except Exception:
                pass  # Ignore errors during cleanup

    def emit(self, record):
        """Emit log record safely"""
        if not self.is_active:
            return

        try:
            msg = self.format(record)
            timestamp = time.strftime("%H:%M:%S", time.localtime())

            # Store in non-Qt capture
            if self.log_capture and self.log_capture.is_active:
                self.log_capture.add_message(msg, record.levelname, timestamp)

            # Try to update widget via callback if available
            if self.widget_callback and callable(self.widget_callback):
                try:
                    self.widget_callback(msg)
                except Exception:
                    pass  # Widget might be destroyed, ignore

        except Exception:
            pass  # Ignore all errors to prevent logging recursion

    def close(self):
        """Close handler safely"""
        self.early_cleanup()
        super().close()


class CellStateLegend(QWidget):
    """Widget to display cell state legend with colored boxes."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)
        self.state_colors = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the legend UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title_label = QLabel("Cell States")
        title_label.setStyleSheet("font-weight: bold; font-size: 12pt; margin-bottom: 5px;")
        layout.addWidget(title_label)
        
        # Legend content area
        self.legend_content = QWidget()
        self.legend_layout = QHBoxLayout()
        self.legend_content.setLayout(self.legend_layout)
        layout.addWidget(self.legend_content)
        
        # Set background
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
    def update_legend(self, state_colors):
        """Update the legend with new state colors."""
        self.state_colors = state_colors
        
        # Clear existing legend items
        for i in reversed(range(self.legend_layout.count())):
            child = self.legend_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        # Add new legend items
        for state, color in state_colors.items():
            legend_item = self.create_legend_item(state, color)
            self.legend_layout.addWidget(legend_item)
        
        # Add stretch to push items to the left
        self.legend_layout.addStretch()
        
    def create_legend_item(self, state, color):
        """Create a single legend item with colored box and label."""
        item_widget = QWidget()
        item_layout = QHBoxLayout()
        item_widget.setLayout(item_layout)
        item_layout.setContentsMargins(5, 2, 5, 2)
        item_layout.setSpacing(5)
        
        # Create colored box
        color_box = QLabel()
        color_box.setFixedSize(16, 16)
        color_box.setStyleSheet(f"""
            QLabel {{
                background-color: rgb({color[0]}, {color[1]}, {color[2]});
                border: 1px solid black;
                border-radius: 2px;
            }}
        """)
        item_layout.addWidget(color_box)
        
        # Create state label
        state_label = QLabel(str(state))
        state_label.setStyleSheet("font-size: 10pt;")
        item_layout.addWidget(state_label)
        
        return item_widget


class LogDisplay(QTextEdit):
    """Enhanced log display widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.NoWrap)
        # Set left alignment and smaller font size for better visibility
        self.setStyleSheet("font-family: monospace; font-size: 8pt; text-align: left;")
        self.setAlignment(Qt.AlignLeft)
        self.log_capture = None
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_from_capture)
        self.update_timer.start(500)  # Update every 500ms
        # Ensure initial scroll position is at the bottom
        self.scroll_to_bottom()

    def set_log_capture(self, log_capture):
        """Set the log capture source"""
        self.log_capture = log_capture
        self.last_message_count = 0
        # Ensure scroll position is at bottom when log capture is set
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        """Force scroll to the bottom of the log display"""
        try:
            # Use multiple methods to ensure scrolling works
            self.moveCursor(QTextCursor.End)
            self.ensureCursorVisible()
            # Force scroll to bottom using the scrollbar
            scrollbar = self.verticalScrollBar()
            if scrollbar:
                scrollbar.setValue(scrollbar.maximum())
            # Additional method: ensure the last line is visible
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        except Exception:
            pass

    def force_scroll_to_bottom(self):
        """More aggressive scroll to bottom method"""
        try:
            # Block signals temporarily to prevent interference
            self.blockSignals(True)
            # Move cursor to end
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.setTextCursor(cursor)
            # Force scroll
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
            # Unblock signals
            self.blockSignals(False)
        except Exception:
            self.blockSignals(False)  # Ensure signals are unblocked even if error occurs

    def update_from_capture(self):
        """Update display from non-Qt log capture"""
        if not self.log_capture or not self.log_capture.is_active:
            return

        try:
            messages = self.log_capture.get_messages()
            new_message_count = len(messages)

            # Only add new messages
            if new_message_count > self.last_message_count:
                new_messages = messages[self.last_message_count:]
                for timestamp, level, message in new_messages:
                    # Check if this message is already in the display to prevent duplicates
                    if not self.toPlainText().endswith(message):
                        self.append_log_safe(message)
                self.last_message_count = new_message_count
                # Ensure scroll to bottom after adding new messages
                self.force_scroll_to_bottom()

        except Exception:
            pass  # Ignore errors

    def append_log_safe(self, text):
        """Safely append log text, keep left alignment and small font."""
        try:
            self.append(text)
            self.setAlignment(Qt.AlignLeft)
            # Force scroll to bottom after appending
            self.scroll_to_bottom()
        except Exception:
            pass

    def append_log(self, text):
        """Backward compatibility method"""
        self.append_log_safe(text)

    def showEvent(self, event):
        """Override show event to ensure scroll to bottom when widget is shown"""
        super().showEvent(event)
        # Use a timer to ensure this happens after the widget is fully shown
        QTimer.singleShot(100, self.scroll_to_bottom)

    def resizeEvent(self, event):
        """Override resize event to maintain scroll position at bottom"""
        super().resizeEvent(event)
        # Use a timer to ensure this happens after the resize is complete
        QTimer.singleShot(50, self.scroll_to_bottom)




def sender_qt():
    return QObject().sender()


class PatchedImageView(pg.ImageView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_update_time = 0
        self._update_throttle_ms = 50  # Throttle image updates to 50ms
        
    def timeLineChanged(self):
        if not hasattr(self, 'timeLine') or self.timeLine is None:
            return
        (ind, time_val) = self.timeIndex(self.timeLine)
        self.sigTimeChanged.emit(self.timeLine.value(), time_val)  # Use self.timeLine.value() for raw index
        if self.ignoreTimeLine:
            return
        self.play(0)  # Stop playing
        current_rounded_index = int(round(ind))  # Use ind for image update logic
        
        # Throttle image updates to improve performance during rapid scrolling
        current_time = time.time() * 1000
        if (current_rounded_index != self.currentIndex and 
            current_time - self._last_update_time > self._update_throttle_ms):
            self.currentIndex = current_rounded_index
            self._last_update_time = current_time
            self.updateImage()


def convert_yaml_config_to_params(yaml_config):
    """
    Convert YAML configuration structure to the params format expected by cell_main.py.
    """
    params = {}
    
    # Debug logging
    module_logger.info(f"Converting YAML config with keys: {list(yaml_config.keys()) if yaml_config else 'None'}")
    
    # (parameter conversion logic remains largely the same, but we add new ILP params)
    exp_settings = yaml_config.get('experiment_settings', {})
    params['Logging Level'] = (exp_settings.get('logging_level', 'INFO'), 'str', 'Set logging level.')
    data_handling = yaml_config.get('data_handling', {})
    params['Raw image extension'] = (data_handling.get('raw_image_extension', 'png'), 'str',
                                     'Raw image file extension.')
    params['Mask extension'] = (data_handling.get('mask_image_extension', 'png'), 'str', 'Mask image file extension.')
    params['Mask folder suffix'] = (data_handling.get('default_mask_folder_suffix', '_mask_all'), 'str',
                                    'Suffix for mask folder.')
    params['Enable Advanced File Structure'] = (data_handling.get('advanced_file_structure_enabled', False), 'bool',
                                                'Enable to use specific folder name definitions.')
    params['Class Definitions (JSON list)'] = (
        json.dumps(data_handling.get('class_definitions_for_folders', ["cell", "mitosis", "fusion"])), 'str',
        'JSON list of class names for state classification.')
    params['Raw Image Folder Name'] = (data_handling.get('raw_image_folder_name_advanced', 'synthetic_cells_raw'), 'str',
                                       'Name of the raw image folder in advanced mode.')
    params['Generic Mask Folder Name'] = (
        data_handling.get('generic_mask_folder_name_advanced', 'synthetic_cells_mask_all'), 'str',
        'Name of the generic mask folder in advanced mode.')
    params['Class-Specific Mask Folder Name Pattern'] = (
        data_handling.get('class_specific_mask_folder_name_pattern_advanced', 'synthetic_cells_mask_{class_name}'),
        'str', 'Pattern for class mask folders.')

    # Get TUI tracker settings from main level (new structure) or fall back to nested structure
    tracker_params_yaml = yaml_config.get('tui_tracker_settings', {})
    module_logger.info(f"TUI tracker settings found: {bool(tracker_params_yaml)}")
    if tracker_params_yaml:
        module_logger.info(f"TUI tracker settings keys: {list(tracker_params_yaml.keys())}")
    else:
        # Fallback to old nested structure
        module_logger.info("No tui_tracker_settings found, trying fallback to nested structure")
        ai_tasks = yaml_config.get('ai_models_and_tasks', {}).get('processing_tasks', [])
        for task in ai_tasks:
            if task.get('task_id') == 'tui_tracker_default' and 'parameters' in task:
                tracker_params_yaml = task['parameters']
                module_logger.info("Found TUI tracker settings in nested structure")
                break

    display_settings_yaml = tracker_params_yaml.get('display_settings', {})
    params['Show id\'s'] = (display_settings_yaml.get('show_cell_ids', True), 'bool', 'Show cell ID labels.')
    params['Show tracks'] = (display_settings_yaml.get('show_cell_tracks', True), 'bool', 'Show cell tracks.')
    params['Show Mitosis Labels'] = (display_settings_yaml.get('show_mitosis_labels', True), 'bool',
                                     'Include P(D1,D2) or D[P] labels.')

    hardware_config = yaml_config.get('hardware_configuration', {})
    cameras = hardware_config.get('cameras', [])
    pixel_scale_from_camera_m = 0.87e-6
    if cameras:
        # (camera config parsing remains the same)
        pass
    params['Pixel scale'] = (tracker_params_yaml.get('pixel_scale_m_per_px', pixel_scale_from_camera_m), 'float', True,
                             'm', 'Pixel size in meters.')

    params['Frames per hour'] = (tracker_params_yaml.get('frames_per_hour', 12), 'int',
                                 'Frames per hour for time calculations.')
    params['Tracking Mode'] = (tracker_params_yaml.get('tracking_algorithm_mode', 'ILP'), 'str',
                               'Tracking algorithm mode.')
    params['Trackpy search range'] = (tracker_params_yaml.get('trackpy_search_range_px', 60), 'int',
                                      'Max displacement for trackpy.')
    params['Trackpy memory'] = (0, 'int', 'Max frames cell absent for trackpy. Set to 0 for CTC evaluation compatibility.')
    params['Trackpy neighbor strategy'] = (tracker_params_yaml.get('trackpy_neighbor_strategy', 'KDTree'), 'str',
                                           'Neighbor search strategy.')
    params['Min Tracklet Duration'] = (tracker_params_yaml.get('min_tracklet_duration_frames', 3), 'int',
                                       'Min frames for a track to be initially visible.')

    params['Mitosis Max Distance Factor'] = (tracker_params_yaml.get('mitosis_max_distance_factor_diameter', 0.7),
                                             'float', 'Max daughter distance from parent.')
    params['Mitosis Area Sum Min Factor'] = (tracker_params_yaml.get('mitosis_area_sum_min_factor_parent', 0.7),
                                             'float', 'Min sum of daughter areas / parent area.')
    params['Mitosis Area Sum Max Factor'] = (tracker_params_yaml.get('mitosis_area_sum_max_factor_parent', 2.5),
                                             'float', 'Max sum of daughter areas / parent area.')
    params['Mitosis Daughter Area Similarity'] = (
        tracker_params_yaml.get('mitosis_daughter_area_similarity_ratio', 0.5), 'float',
        'Min ratio of smaller daughter area to larger.')

    params['Classification Max Centroid Distance'] = (
        tracker_params_yaml.get('classification_max_centroid_distance_px', 30.0), 'float',
        'Max distance for state classification.')

    # Updated ILP Parameters
    ilp_params_yaml = tracker_params_yaml.get('ilp_settings', {})
    params['ILP Solver'] = (ilp_params_yaml.get('ilp_solver', 'scipy'), 'str', 'ILP solver engine to use.')
    params['Gurobi WLSACCESSID'] = (ilp_params_yaml.get('gurobi_wlsaccessid', ''),
                                    'str', 'Gurobi Web License Service Access ID.')
    params['Gurobi WLSSECRET'] = (ilp_params_yaml.get('gurobi_wlssecret', ''),
                                  'str', 'Gurobi Web License Service Secret.')
    params['Gurobi LICENSEID'] = (ilp_params_yaml.get('gurobi_licenseid',0 ), 'int', 'Gurobi License ID.')

    params['ILP Max Search Distance'] = (ilp_params_yaml.get('max_search_distance_px', 50), 'int',
                                         'Max distance for creating graph edges.')
    params['ILP Transition Cost Weight'] = (ilp_params_yaml.get('cost_weight_transition', 1.0), 'float',
                                            'Weight for transition distance cost.')
    params['ILP Mitosis Cost'] = (ilp_params_yaml.get('cost_base_mitosis', 5.0), 'float',
                                  'Base penalty for a mitosis event.')
    params['ILP Fusion Cost'] = (ilp_params_yaml.get('cost_base_fusion', 10.0), 'float',
                                 'Base penalty for a fusion event.')
    params['ILP Appearance Cost'] = (ilp_params_yaml.get('cost_appearance', 20.0), 'float',
                                     'Penalty for a new track appearing.')
    params['ILP Disappearance Cost'] = (ilp_params_yaml.get('cost_disappearance', 20.0), 'float',
                                        'Penalty for a track disappearing.')

    trackastra_params_yaml = tracker_params_yaml.get('trackastra_settings', {})
    params['Trackastra Model'] = (trackastra_params_yaml.get('model', 'general_2d'), 'str',
                                  'Pretrained Trackastra model.')
    params['Trackastra Linking Mode'] = (trackastra_params_yaml.get('linking_mode', 'greedy'), 'str',
                                         'Linking algorithm for Trackastra.')
    params['Trackastra Device'] = (trackastra_params_yaml.get('device', 'cuda'), 'str', 'Device for Trackastra model.')

    # Output settings from new structure
    output_settings_yaml = tracker_params_yaml.get('output_settings', {})
    params['Save Temporal Outline Stack'] = (output_settings_yaml.get('save_temporal_outline_stack', True), 'bool', 'Save temporal outline stack visualization')
    params['Temporal Outline Colormap'] = (display_settings_yaml.get('temporal_outline_colormap', 'plasma'), 'list', COLORMAP_OPTIONS,
                                           'Colormap for temporal outline stack visualization')
    params['Figure DPI'] = (display_settings_yaml.get('figure_dpi', 300), 'int',
                                      'DPI for all figure outputs (higher = sharper)')
    params['Temporal Outline Scale Factor'] = (display_settings_yaml.get('temporal_outline_scale_factor', 2), 'int',
                                               'Scale factor for temporal outline stack resolution (higher = larger, sharper image)')
    params['Offset Track Plot'] = (output_settings_yaml.get('offset_track_plot', True), 'bool', 'Save offset track plot (each cell normalized at t=0, colored by final position)')
    params['Offset Track Plot Left Color'] = (output_settings_yaml.get('offset_track_plot_left_color', 'black'), 'str', 'Color for tracks ending left of origin')
    params['Offset Track Plot Right Color'] = (output_settings_yaml.get('offset_track_plot_right_color', 'red'), 'str', 'Color for tracks ending right of origin')
    params['Offset Track Plot Output Filename'] = (output_settings_yaml.get('offset_track_plot_output_filename', 'offset_track_plot.png'), 'str', 'Filename for offset track plot image')
    
    # Time-based offset track plot settings
    params['Time-based Offset Track Plot'] = (output_settings_yaml.get('time_based_offset_track_plot', True), 'bool', 'Save time-based offset track plot (each cell normalized at t=0, colored by elapsed frame)')
    params['Time-based Offset Track Plot Colormap'] = (output_settings_yaml.get('time_based_offset_track_plot_colormap', 'plasma'), 'list', COLORMAP_OPTIONS, 'Colormap for time-based offset track plot')
    params['Time-based Offset Track Plot Output Filename'] = (output_settings_yaml.get('time_based_offset_track_plot_output_filename', 'time_based_offset_track_plot.png'), 'str', 'Filename for time-based offset track plot image')

    # GIF Overlay settings
    params['Save ID-based GIF Overlay'] = (output_settings_yaml.get('save_id_based_gif_overlay', True), 'bool', 'Save GIF overlay with each cell ID colored uniquely')
    params['Save State-based GIF Overlay'] = (output_settings_yaml.get('save_state_based_gif_overlay', True), 'bool', 'Save GIF overlay with cells colored by state class')
    params['GIF Overlay Alpha'] = (output_settings_yaml.get('gif_overlay_alpha', 0.6), 'float', 'Transparency of overlay (0-1)')
    params['GIF Overlay FPS'] = (output_settings_yaml.get('gif_overlay_fps', 10), 'int', 'Frames per second for GIF animation')

    # 16-bit Class Mask settings
    params['Save 16-bit Class Masks'] = (output_settings_yaml.get('save_16bit_class_masks', True), 'bool', 'Save 16-bit PNG masks for each state class per frame')
    params['Class Mask Prefix'] = (output_settings_yaml.get('class_mask_prefix', 'class_mask'), 'str', 'Filename prefix for class mask files')

    # CTC Metrics evaluation option
    params['Run CTC Metrics Evaluation'] = (output_settings_yaml.get('run_ctc_metrics_evaluation', True), 'bool', 'Run CTC metrics evaluation after tracking (if GT is available)?')

    # Analysis plot settings from new structure
    analysis_plot_settings_yaml = tracker_params_yaml.get('analysis_plot_settings', {})
    params['Save All Analysis Plots'] = (analysis_plot_settings_yaml.get('save_all_analysis_plots', True), 'bool', 'Toggle all individual analysis plot options')
    params['Save Cell Count Plot'] = (analysis_plot_settings_yaml.get('save_cell_count_plot', True), 'bool', 'Save cell count over time plot')
    params['Save Event Timeline Plot'] = (analysis_plot_settings_yaml.get('save_event_timeline_plot', True), 'bool', 'Save mitosis/fusion event timeline plot')
    params['Save Cell Cycle Duration Plot'] = (analysis_plot_settings_yaml.get('save_cell_cycle_duration_plot', True), 'bool', 'Save cell cycle duration analysis plot')
    params['Save Track Length Distribution Plot'] = (analysis_plot_settings_yaml.get('save_track_length_distribution_plot', True), 'bool', 'Save track length distribution histogram')
    params['Save Cell Density Heatmap'] = (analysis_plot_settings_yaml.get('save_cell_density_heatmap', True), 'bool', 'Save spatial cell density heatmap')
    params['Save Movement Vector Field'] = (analysis_plot_settings_yaml.get('save_movement_vector_field', True), 'bool', 'Save cell movement vector field plot')
    params['Save Spatial Clustering Plot'] = (analysis_plot_settings_yaml.get('save_spatial_clustering_plot', True), 'bool', 'Save spatial clustering analysis plot')
    params['Save Ensemble Outline Plots'] = (analysis_plot_settings_yaml.get('save_ensemble_outline_plots', True), 'bool', 'Save ensemble outline polar and heatmap plots')
    params['Save Phylogenetic Tree Plots'] = (analysis_plot_settings_yaml.get('save_phylogenetic_tree_plots', True), 'bool', 'Save phylogenetic tree plots')
    params['Save State Analysis Plots'] = (analysis_plot_settings_yaml.get('save_state_analysis_plots', True), 'bool', 'Save state transition and distribution plots (if states exist)')
    params['Save Publication Lineage Trees'] = (analysis_plot_settings_yaml.get('save_publication_lineage_trees', True), 'bool', 'Save publication-ready lineage tree plots')

    return params

def ensure_window_visible(window, parent_window=None):
    """Ensure a window is visible on screen"""
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QTimer

        desktop = QApplication.desktop()
        screen_geometry = desktop.availableGeometry(window)
        window_size = window.size()

        if parent_window and hasattr(parent_window, 'geometry'):
            parent_center = parent_window.geometry().center()
            new_x = parent_center.x() - window_size.width() // 2
            new_y = parent_center.y() - window_size.height() // 2
        else:
            new_x = screen_geometry.center().x() - window_size.width() // 2
            new_y = screen_geometry.center().y() - window_size.height() // 2

        new_x = max(screen_geometry.left(), min(new_x, screen_geometry.right() - window_size.width()))
        new_y = max(screen_geometry.top(), min(new_y, screen_geometry.bottom() - window_size.height()))

        current_geometry = window.geometry()
        is_visible_on_any_screen = False
        for i in range(desktop.screenCount()):
            s_geom = desktop.availableGeometry(i)
            if (current_geometry.right() > s_geom.left() and current_geometry.left() < s_geom.right() and
                    current_geometry.bottom() > s_geom.top() and current_geometry.top() < s_geom.bottom()):
                is_visible_on_any_screen = True
                break

        if not is_visible_on_any_screen:
            window.move(new_x, new_y)

        window.show()
        window.raise_()
        window.activateWindow()

        def delayed_positioning():
            try:
                final_geometry = window.geometry()
                screen_after_delay = QApplication.desktop().availableGeometry(window)
                if (final_geometry.right() < screen_after_delay.left() or
                        final_geometry.left() > screen_after_delay.right() or
                        final_geometry.bottom() < screen_after_delay.top() or
                        final_geometry.top() > screen_after_delay.bottom()):
                    window.move(new_x, new_y)
                    window.show();
                    window.raise_();
                    window.activateWindow()
            except Exception as e:
                module_logger.error(f"Error in delayed positioning: {e}")

        QTimer.singleShot(150, delayed_positioning)

    except Exception as e:
        module_logger.error(f"Error positioning window: {e}")
        try:
            window.show();
            window.raise_();
            window.activateWindow()
        except:
            pass


def initialize_cell_tracking_app(yaml_config=None, parent_window=None, external_log_handler=None):
    """
    Initialize the cell tracking application with configuration from YAML.
    """
    global main_app_state, ui_elements
    init_start_time = time.perf_counter()
    module_logger.info("--- initialize_cell_tracking_app called ---")

    if yaml_config:
        copied_yaml_config = copy.deepcopy(yaml_config)
        converted_params = convert_yaml_config_to_params(copied_yaml_config)
        main_app_state['params'].update(converted_params)
        module_logger.info("Cell tracking parameters updated from YAML configuration")
    else:
        module_logger.warning("No YAML configuration provided to TUI Tracker. Using default parameters.")

    app = QApplication.instance() if QApplication.instance() else QApplication(sys.argv)
    module_logger.info(f"--- TIME: QApplication instance retrieved in {time.perf_counter() - init_start_time:.4f} s ---")

    ui_elements['win'] = QMainWindow()
    ui_elements['parent_window_ref'] = weakref.ref(parent_window) if parent_window else None
    if parent_window:
        ui_elements['win'].setWindowFlags(Qt.Window)

    ui_elements['dock_area'] = DockArea()
    ui_elements['win'].setCentralWidget(ui_elements['dock_area'])
    ui_elements['win'].resize(1700, 1000)
    ui_elements['win'].setWindowTitle('TUI Tracker - Cell Tracking Editor')
    module_logger.info(f"--- TIME: Main window created in {time.perf_counter() - init_start_time:.4f} s ---")

    setup_docks_and_widgets()
    module_logger.info(f"--- TIME: Docks and widgets setup in {time.perf_counter() - init_start_time:.4f} s ---")

    if external_log_handler:
        if callable(external_log_handler):
            class SimpleCallbackHandler(logging.Handler):
                def __init__(self, callback_func):
                    super().__init__()
                    self.callback_func = callback_func
                    self.is_active = True
                    atexit.register(self.cleanup)

                def cleanup(self):
                    self.is_active = False
                    try:
                        logging.getLogger().removeHandler(self)
                    except:
                        pass

                def emit(self, record):
                    if self.is_active:
                        try:
                            self.callback_func(record.getMessage(), record.levelname, record.name)
                        except:
                            pass

            logging.getLogger().addHandler(SimpleCallbackHandler(external_log_handler))
            module_logger.info("External log callback handler attached.")
        elif isinstance(external_log_handler, logging.Handler):
            try:
                logging.getLogger().addHandler(external_log_handler)
                module_logger.info("External log handler object attached.")
            except Exception as e:
                module_logger.error(f"Failed to attach external log handler object: {e}")

    # Deferred initialization of the parameter tree
    QTimer.singleShot(50, lambda: initialize_ui_components(deferred=True))
    initialize_ui_components(deferred=False) # Initialize non-deferred components
    module_logger.info(f"--- TIME: UI components initialized (deferred) in {time.perf_counter() - init_start_time:.4f} s ---")


    setup_undo_redo_actions_main()
    cell_undoredo.update_undo_redo_actions_enabled_state(ui_elements)

    actual_parent = parent_window if not ui_elements['parent_window_ref'] else ui_elements['parent_window_ref']()
    ensure_window_visible(ui_elements['win'], actual_parent)
    module_logger.info(f"--- TIME: Window made visible in {time.perf_counter() - init_start_time:.4f} s ---")


    module_logger.info("TUI Tracker application initialized successfully")
    return ui_elements['win']


def setup_docks_and_widgets():
    """Setup the dock widgets and basic UI structure"""
    global ui_elements

    ui_elements['d_io'] = Dock("I/O & Parameters", size=(1, 280))
    ui_elements['d_img'] = Dock("Image Views", size=(800, 400))
    ui_elements['d_tracks'] = Dock("Cell Editor", size=(450, 400))
    ui_elements['d_log'] = Dock("Log Console", size=(1, 150))
    ui_elements['d_lineage'] = Dock("Lineage Tree", size=(600, 400))

    ui_elements['dock_area'].addDock(ui_elements['d_io'], 'top')
    ui_elements['dock_area'].addDock(ui_elements['d_img'], 'bottom', ui_elements['d_io'])
    ui_elements['dock_area'].addDock(ui_elements['d_tracks'], 'right', ui_elements['d_img'])
    ui_elements['dock_area'].addDock(ui_elements['d_lineage'], 'bottom', ui_elements['d_img'])
    ui_elements['dock_area'].addDock(ui_elements['d_log'], 'bottom', ui_elements['d_tracks'])

    ui_elements['log_display_widget'] = LogDisplay()
    ui_elements['d_log'].addWidget(ui_elements['log_display_widget'])
    ui_elements['log_capture'] = ThreadSafeLogCapture()
    ui_elements['log_display_widget'].set_log_capture(ui_elements['log_capture'])

    root_logger = logging.getLogger()

    # Configure root logger once
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)  # Default level
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - TUI_TRACKER - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(console_handler)
        ui_elements['console_log_handler'] = console_handler

    # Ensure GUI handler is added only once
    if 'gui_log_handler' not in ui_elements or not ui_elements['gui_log_handler']:
        ui_elements['gui_log_handler'] = NonQtLogHandler(ui_elements['log_capture'], None)  # Remove widget callback to prevent duplication
        ui_elements['gui_log_handler'].setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(ui_elements['gui_log_handler'])

    initial_log_level_str = main_app_state['params'].get('Logging Level', ('INFO',))[0]
    initial_log_level = getattr(logging, initial_log_level_str.upper(), logging.INFO)
    root_logger.setLevel(initial_log_level)


main_app_state = {
    'path_in': '', 'loaded_class_specific_masks': {}, 'track_states': {}, 'captured_tracking_log': "",
    'params': {
        'Logging Level': ('INFO', 'str', 'Set logging level.'),
        'Raw image extension': ('png', 'str', 'Raw image file extension.'),
        'Mask extension': ('png', 'str', 'Mask image file extension.'),
        'Mask folder suffix': ('_mask_all', 'str', 'Suffix for mask folder.'),
        'Enable Advanced File Structure': (False, 'bool', 'Use specific folder names.'),
        'Class Definitions (JSON list)': ('["cell", "mitosis", "fusion"]', 'str', 'JSON list of class names.'),
        'Raw Image Folder Name': ('synthetic_cells', 'str', 'Name of the raw image folder.'),
        'Generic Mask Folder Name': ('synthetic_cells_mask_all', 'str', 'Name of the generic mask folder.'),
        'Class-Specific Mask Folder Name Pattern': ('synthetic_cells_mask_{class_name}', 'str', 'Pattern for class mask folders.'),
        'Classification Max Centroid Distance': (30.0, 'float', 'Max distance for state classification.'),
        'Show id\'s': (True, 'bool', 'Show cell ID labels.'),
        'Show tracks': (True, 'bool', 'Show cell tracks.'),
        'Show Mitosis Labels': (True, 'bool', 'Include mitosis labels.'),
        'Figure DPI': (300, 'int', 'DPI for all figure outputs (higher = sharper)'),
        'Temporal Outline Colormap': ('plasma', 'list', COLORMAP_OPTIONS, 'Colormap for temporal outline stack visualization'),
        'Temporal Outline Scale Factor': (2, 'int', 'Scale factor for temporal outline stack resolution (higher = larger, sharper image)'),
        'Pixel scale': (0.87e-6, 'float', True, 'm', 'Pixel size in meters.'),
        'Frames per hour': (12, 'int', 'Frames per hour for time calculations.'),
        'Tracking Mode': ('ILP', 'str', 'Tracking algorithm mode.'),
        'Trackpy search range': (60, 'int', 'Max displacement for trackpy.'),
        'Trackpy memory': (0, 'int', 'Max frames cell absent for trackpy. Set to 0 for CTC evaluation compatibility.'),
        'Trackpy neighbor strategy': ('KDTree', 'str', 'Neighbor search strategy.'),
        'Min Tracklet Duration': (1, 'int', 'Min frames for a track to be initially visible.'),
        'Handle Missing Cells in CTC': (True, 'bool', 'Remove tracks with missing mask pixels for CTC evaluation.'),
        'Mitosis Max Distance Factor': (0.7, 'float', 'Max daughter distance from parent.'),
        'Mitosis Area Sum Min Factor': (0.7, 'float', 'Min sum of daughter areas / parent area.'),
        'Mitosis Area Sum Max Factor': (2.5, 'float', 'Max sum of daughter areas / parent area.'),
        'Mitosis Daughter Area Similarity': (0.5, 'float', 'Min ratio of smaller daughter area to larger.'),
        'ILP Solver': ('gurobi', 'str', 'ILP solver engine to use.'),
        'Gurobi WLSACCESSID': ('', 'str', 'Gurobi Web License Service Access ID.'),
        'Gurobi WLSSECRET': ('', 'str', 'Gurobi Web License Service Secret.'),
        'Gurobi LICENSEID': (0, 'int', 'Gurobi License ID.'),
        'ILP Max Search Distance': (50, 'int', 'Max distance for ILP graph edges.'),
        'ILP Transition Cost Weight': (1.0, 'float', 'Weight for ILP transition cost.'),
        'ILP Mitosis Cost': (5.0, 'float', 'Base penalty for an ILP mitosis event.'),
        'ILP Fusion Cost': (10.0, 'float', 'Base penalty for an ILP fusion event.'),
        'ILP Appearance Cost': (20.0, 'float', 'Penalty for an ILP track appearance.'),
        'ILP Disappearance Cost': (20.0, 'float', 'Penalty for an ILP track disappearance.'),
        'Trackastra Model': ('general_2d', 'str', 'Pretrained Trackastra model.'),
        'Trackastra Linking Mode': ('greedy', 'str', 'Linking algorithm for Trackastra.'),
        'Trackastra Device': ('cuda', 'str', 'Device for Trackastra model (cuda, cpu, mps).'),
        'Offset Track Plot': (True, 'bool', 'Save offset track plot (each cell normalized at t=0, colored by final position)'),
        'Offset Track Plot Left Color': ('black', 'str', 'Color for tracks ending left of origin'),
        'Offset Track Plot Right Color': ('red', 'str', 'Color for tracks ending right of origin'),
        'Offset Track Plot Output Filename': ('offset_track_plot.png', 'str', 'Filename for offset track plot image'),
        'Time-based Offset Track Plot': (True, 'bool', 'Save time-based offset track plot (each cell normalized at t=0, colored by elapsed frame)'),
        'Time-based Offset Track Plot Colormap': ('plasma', 'list', COLORMAP_OPTIONS, 'Colormap for time-based offset track plot'),
        'Time-based Offset Track Plot Output Filename': ('time_based_offset_track_plot.png', 'str', 'Filename for time-based offset track plot image'),
        'Save ID-based GIF Overlay': (True, 'bool', 'Save GIF overlay with each cell ID colored uniquely'),
        'Save State-based GIF Overlay': (True, 'bool', 'Save GIF overlay with cells colored by state class'),
        'GIF Overlay Alpha': (0.8, 'float', 'Transparency of overlay (0-1)'),
        'GIF Overlay FPS': (10, 'int', 'Frames per second for GIF animation'),
        'Save 16-bit Class Masks': (True, 'bool', 'Save 16-bit PNG masks for each state class per frame'),
        'Class Mask Prefix': ('class_mask', 'str', 'Filename prefix for class mask files'),
        'Run CTC Metrics Evaluation': (True, 'bool', 'Run CTC metrics evaluation after tracking (if GT is available)?'),
        'Save State Masks and Colors': (True, 'bool', 'Save state-based masks and color visualizations'),
        'Save Track Type CSVs and Plots': (True, 'bool', 'Save singular, mitosis, and fusion track analysis'),
        'Save Comprehensive Analysis Plots': (True, 'bool', 'Save comprehensive statistical analysis plots'),
        'Analysis Plots Fast Mode': (True, 'bool', 'Use fast mode for analysis plots (reduces quality but much faster for large datasets)'),
        # Granular Analysis Plot Controls
        'Save Cell Count Plot': (True, 'bool', 'Save cell count over time plot'),
        'Save Event Timeline Plot': (True, 'bool', 'Save mitosis/fusion event timeline plot'),
        'Save Cell Cycle Duration Plot': (True, 'bool', 'Save cell cycle duration analysis plot'),
        'Save Track Length Distribution Plot': (True, 'bool', 'Save track length distribution histogram'),
        'Save Cell Density Heatmap': (True, 'bool', 'Save spatial cell density heatmap'),
        'Save Movement Vector Field': (True, 'bool', 'Save cell movement vector field plot'),
        'Save Spatial Clustering Plot': (True, 'bool', 'Save spatial clustering analysis plot'),
        'Save Ensemble Outline Plots': (True, 'bool', 'Save ensemble outline polar and heatmap plots'),
        'Save Phylogenetic Tree Plots': (True, 'bool', 'Save phylogenetic tree plots'),
        'Save State Analysis Plots': (True, 'bool', 'Save state transition and distribution plots (if states exist)'),
        'Save Publication Lineage Trees': (True, 'bool', 'Save publication-ready lineage tree plots'),
        'Save Cell Distribution Stacked Bar Plot': (True, 'bool', 'Save comprehensive cell distribution stacked bar plot'),
        'Cell Distribution Plot Categories': ('both', 'list', ['all', 'track_type', 'state', 'both'], 'Categories to include in cell distribution plot'),
        'Save Simplified Cell Distribution Plot': (True, 'bool', 'Save simplified cell distribution plot (more readable)'),
        'Save Enhanced Cell Distribution Plots': (True, 'bool', 'Save enhanced cell distribution plots with separate track type and state charts including percentages'),
        'Save Temporal Outline Stack': (True, 'bool', 'Save temporal outline stack visualization'),
        'Temporal Outline Colormap': ('plasma', 'list', COLORMAP_OPTIONS, 'Colormap for temporal outline stack visualization'),
        'Temporal Outline Scale Factor': (2, 'int', 'Scale factor for temporal outline stack resolution (higher = larger, sharper image)'),
        'Temporal Outline Fast Mode': (True, 'bool', 'Use fast mode for temporal outline (reduces quality but much faster for large datasets)'),
        'Offset Track Plot': (True, 'bool', 'Save offset track plot (each cell normalized at t=0, colored by final position)'),
        'Offset Track Plot Left Color': ('black', 'str', 'Color for tracks ending left of origin'),
        'Offset Track Plot Right Color': ('red', 'str', 'Color for tracks ending right of origin'),
        'Offset Track Plot Output Filename': ('offset_track_plot.png', 'str', 'Filename for offset track plot image'),
        'Time-based Offset Track Plot': (True, 'bool', 'Save time-based offset track plot (each cell normalized at t=0, colored by elapsed frame)'),
        'Time-based Offset Track Plot Colormap': ('plasma', 'list', COLORMAP_OPTIONS, 'Colormap for time-based offset track plot'),
        'Time-based Offset Track Plot Output Filename': ('time_based_offset_track_plot.png', 'str', 'Filename for time-based offset track plot image'),
        'Save Lineage Plots': (True, 'bool', 'Save lineage tree visualizations'),
        'Save Track Overview': (True, 'bool', 'Save overview image showing all tracks'),
        'Save ID Mask Animations': (True, 'bool', 'Save animated GIFs of ID-based masks'),
        'Save Colorized Mask Animations': (True, 'bool', 'Save animated GIFs of colorized segmentation masks'),
        'Save Track Animations': (True, 'bool', 'Save animated GIFs showing cell tracks'),
        'Save Raw Images as PNG': (True, 'bool', 'Save raw images as PNG files for later loading'),
        'Save Merged Masks': (True, 'bool', 'Save merged mask data if available'),
        'Save Cell Editor Table': (True, 'bool', 'Save complete cell editor table data'),
        'Save Track States': (True, 'bool', 'Save track state classifications'),
        'Save Cell Visibility': (True, 'bool', 'Save cell visibility settings'),
        'Save Cell Coordinates': (True, 'bool', 'Save cell coordinate data'),
        'Save Cell Frame Presence': (True, 'bool', 'Save which frames each cell appears in'),
        'Save Tracking Log': (True, 'bool', 'Save tracking operation log'),
        'Save Experiment Parameters': (True, 'bool', 'Save experiment settings and parameters'),
        'Save Lineage Relationships': (True, 'bool', 'Save parent-daughter lineage information'),
        'Save All Analysis Plots': (True, 'bool', 'Toggle all individual analysis plot options'),
    },
    'current_frame_index': 0, 'background_id': 0, 'raw_imgs': None, 'raw_masks': None, 'merged_masks': None,
    'id_masks': None, 'id_masks_initial': None, 'cell_ids': [], 'color_list': None, 'cell_color_idx': {},
    'trj': pd.DataFrame(), 'initial_trj': pd.DataFrame(), 'col_tuple': {}, 'col_weights': {}, 'show_ids_current': True, 'show_tracks_current': True,
    'cell_visibility': {}, 'cell_frame_presence': {}, 'cell_ids_raw_img': {}, 'cell_ids_mask': {},
    'cell_y': defaultdict(dict), 'cell_x': defaultdict(dict), 'hover_label_raw': None, 'hover_label_mask': None, 'wide_track_cell_id': None,
    'track_data_per_frame': defaultdict(lambda: defaultdict(lambda: np.zeros((0, 2), dtype=float))),
    'track_plots_per_cell': {}, 'ancestry': defaultdict(list), 'cell_lineage': defaultdict(list),
    'has_lineage': False, 'v_raw_img_original_state': None, 'v_mask_original_state': None,
    'next_available_daughter_id': 1, 'last_saved_res_path': None, 'calculated_stats': {}, 'ctc_eval_results': {},
}

ui_elements = {
    'win': None, 'dock_area': None, 'd_io': None, 'd_img': None, 'd_tracks': None, 'd_log': None, 'd_lineage': None,
    'log_display_widget': None, 'gui_log_handler': None, 'console_log_handler': None, 'log_capture': None,
    'parent_window_ref': None, 'pt_io_params': None, 'p_root_params': None, 'button_widgets_map': {},
    'v_raw_img': None, 'pi_raw_img': None, 'v_mask': None, 'pi_mask': None,
    'table_cell_selection': None, 'b_merge_track': None, 'lineage_tree_widget': None,
    'undo_action': None, 'redo_action': None,
}


def param_changed_callback(param_tree_item, changes):
    """Handle parameter changes in the UI"""
    global main_app_state, ui_elements
    for param_obj, change_type, new_value in changes:
        param_name = param_obj.name()
        if change_type == 'value' and param_name in main_app_state['params']:
            current_param_tuple = main_app_state['params'][param_name]
            main_app_state['params'][param_name] = (new_value,) + current_param_tuple[1:]
            if param_name == 'Show id\'s':
                main_app_state['show_ids_current'] = new_value
            elif param_name == 'Show tracks':
                main_app_state['show_tracks_current'] = new_value
            elif param_name == 'Logging Level':
                numeric_level = getattr(logging, str(new_value).strip().upper(), None)
                if numeric_level is not None:
                    logging.getLogger().setLevel(numeric_level)
                else:
                    param_obj.setValue(logging.getLevelName(logging.getLogger().getEffectiveLevel()))

            if param_name in ['Show id\'s', 'Show tracks', 'Show Mitosis Labels']:
                cell_ui_callbacks.handle_update_view_for_current_frame(main_app_state, ui_elements)
            
            # Handle dependencies between master checkbox and individual analysis plot controls
            if param_name == 'Save Comprehensive Analysis Plots':
                # List of all individual analysis plot parameters that should be controlled by the master switch
                dependent_params = [
                    'Save Cell Count Plot',
                    'Save Event Timeline Plot', 
                    'Save Cell Cycle Duration Plot',
                    'Save Track Length Distribution Plot',
                    'Save Cell Density Heatmap',
                    'Save Movement Vector Field',
                    'Save Spatial Clustering Plot',
                    'Save Ensemble Outline Plots',
                    'Save Phylogenetic Tree Plots',
                    'Save State Analysis Plots',
                    'Save Publication Lineage Trees',
                    'Save Temporal Outline Stack',
                    'Temporal Outline Colormap',
                    'Figure DPI',
                    'Temporal Outline Scale Factor',
                    'Temporal Outline Fast Mode',
                    'Analysis Plots Fast Mode',
                    'Offset Track Plot',
                    'Offset Track Plot Left Color',
                    'Offset Track Plot Right Color',
                    'Offset Track Plot Output Filename',
                    'Time-based Offset Track Plot',
                    'Time-based Offset Track Plot Colormap',
                    'Time-based Offset Track Plot Output Filename'
                ]
                
                # Get the parameter tree to access individual parameters
                param_tree = ui_elements.get('pt_io_params')
                if param_tree and hasattr(param_tree, 'param'):
                    try:
                        # Find the Analysis Plot Options group
                        analysis_group = param_tree.param('Analysis Plot Options')
                        if analysis_group:
                            # Enable/disable each dependent parameter based on master switch
                            for dependent_param in dependent_params:
                                try:
                                    child_param = analysis_group.child(dependent_param)
                                    if child_param:
                                        # Enable/disable the parameter based on master checkbox state
                                        child_param.setReadonly(not new_value)
                                        # Optionally, you could also uncheck them when disabled:
                                        # if not new_value:
                                        #     child_param.setValue(False)
                                except Exception as e:
                                    logging.debug(f"Could not find or modify parameter '{dependent_param}': {e}")
                    except Exception as e:
                        logging.debug(f"Could not access Analysis Plot Options group: {e}")
            
            # Handle "Select All Analysis Plots" checkbox
            elif param_name == 'Save All Analysis Plots':
                logging.info(f"Save All Analysis Plots changed to: {new_value}")
                # List of all individual analysis plot parameters that should be toggled
                analysis_plot_params = [
                    'Save Cell Count Plot',
                    'Save Event Timeline Plot', 
                    'Save Cell Cycle Duration Plot',
                    'Save Track Length Distribution Plot',
                    'Save Cell Density Heatmap',
                    'Save Movement Vector Field',
                    'Save Spatial Clustering Plot',
                    'Save Ensemble Outline Plots',
                    'Save Phylogenetic Tree Plots',
                    'Save State Analysis Plots',
                    'Save Publication Lineage Trees',
                    'Save Temporal Outline Stack',
                    'Temporal Outline Colormap',
                    'Figure DPI',
                    'Temporal Outline Scale Factor',
                    'Offset Track Plot',
                    'Offset Track Plot Left Color',
                    'Offset Track Plot Right Color',
                    'Offset Track Plot Output Filename',
                    'Time-based Offset Track Plot',
                    'Time-based Offset Track Plot Colormap',
                    'Time-based Offset Track Plot Output Filename'
                ]
                
                # Get the root parameter tree to access individual parameters
                p_root = ui_elements.get('p_root_params')
                if p_root:
                    try:
                        # Find the Analysis Plot Options group
                        analysis_group = p_root.param('Analysis Plot Options')
                        if analysis_group:
                            # Set all individual analysis plot parameters to the same value as the select all checkbox
                            for plot_param in analysis_plot_params:
                                try:
                                    child_param = analysis_group.child(plot_param)
                                    if child_param:
                                        # Always update the parameter value, regardless of read-only state
                                        child_param.setValue(new_value)
                                        # Also update the main_app_state
                                        if plot_param in main_app_state['params']:
                                            current_param_tuple = main_app_state['params'][plot_param]
                                            main_app_state['params'][plot_param] = (new_value,) + current_param_tuple[1:]
                                            logging.info(f"Updated {plot_param} to {new_value}")
                                except Exception as e:
                                    logging.error(f"Could not find or modify parameter '{plot_param}': {e}")
                    except Exception as e:
                        logging.error(f"Could not access Analysis Plot Options group: {e}")
                else:
                    logging.error("Could not access parameter tree for Save All Analysis Plots")
            



class OptimizationTargetsDialog(QDialog):
    """A dialog to let the user specify optimization targets for the ILP tracker."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set ILP Optimization Targets")
        layout = QFormLayout(self)

        # Iteration count
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(3, 500)
        self.iterations_spin.setValue(50)
        self.iterations_spin.setToolTip("Total optimizer iterations. Must be at least 3.")
        layout.addRow(QLabel("Number of Iterations:"), self.iterations_spin)

        # Spacer
        layout.addRow(QLabel("-" * 40))

        # Targets
        self.target_first_frame_check = QCheckBox("Target Cell Count in First Frame:")
        self.target_first_frame_spin = QSpinBox()
        self.target_first_frame_spin.setRange(1, 10000)
        self.target_first_frame_spin.setValue(50)
        layout.addRow(self.target_first_frame_check, self.target_first_frame_spin)

        self.target_mitosis_check = QCheckBox("Target Mitosis Event Count:")
        self.target_mitosis_spin = QSpinBox()
        self.target_mitosis_spin.setRange(0, 1000)
        self.target_mitosis_spin.setValue(10)
        layout.addRow(self.target_mitosis_check, self.target_mitosis_spin)

        self.target_fusion_check = QCheckBox("Target Fusion Event Count:")
        self.target_fusion_spin = QSpinBox()
        self.target_fusion_spin.setRange(0, 1000)
        self.target_fusion_spin.setValue(5)
        layout.addRow(self.target_fusion_check, self.target_fusion_spin)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)

    def get_settings(self):
        """Returns a dictionary of the selected targets and settings."""
        settings = {
            "targets": {},
            "n_calls": self.iterations_spin.value()
        }
        if self.target_first_frame_check.isChecked():
            settings['targets']['num_cells_frame_zero'] = self.target_first_frame_spin.value()
        if self.target_mitosis_check.isChecked():
            settings['targets']['num_mitosis'] = self.target_mitosis_spin.value()
        if self.target_fusion_check.isChecked():
            settings['targets']['num_fusion'] = self.target_fusion_spin.value()
        return settings


def handle_optimize_ilp_clicked(main_app_state, ui_elements):
    """Orchestrates the ILP parameter optimization process."""
    win = ui_elements.get('win')
    if not SKOPT_AVAILABLE:
        QMessageBox.critical(win, "Dependency Missing", "The 'scikit-optimize' library is required.")
        return
    if main_app_state.get('raw_masks') is None:
        QMessageBox.warning(win, "No Data", "Please open a folder with mask data first.")
        return

    dialog = OptimizationTargetsDialog(win)
    if dialog.exec_() == QDialog.Accepted:
        settings = dialog.get_settings()
        targets, n_calls = settings['targets'], settings['n_calls']
        if not targets:
            QMessageBox.warning(win, "No Targets", "Operation cancelled.")
            return

        progress = QProgressDialog(f"Optimizing ILP Parameters...", "Cancel", 0, n_calls, win)
        progress.setWindowModality(Qt.WindowModal)

        def progress_callback():
            progress.setValue(progress.value() + 1)
            QApplication.processEvents()
            return not progress.wasCanceled()

        best_params, best_score, log_filepath = run_optimization(main_app_state, targets, n_calls, progress_callback)
        progress.setValue(n_calls)

        if best_params is None:
            # Handle cancellation or error...
            return

        # V14 FIX: Corrected the parameter group name to 'ILP Tracking' to match the UI definition.
        # This was 'ILP Tracking (Experimental)' which caused a silent failure.
        p_root = ui_elements['p_root_params']
        ilp_param_group = p_root.param('ILP Tracking')
        ilp_param_group.child('ILP Max Search Distance').setValue(best_params['ILP Max Search Distance'])
        ilp_param_group.child('ILP Transition Cost Weight').setValue(best_params['ILP Transition Cost Weight'])
        ilp_param_group.child('ILP Mitosis Cost').setValue(best_params['ILP Mitosis Cost'])
        ilp_param_group.child('ILP Fusion Cost').setValue(best_params['ILP Fusion Cost'])
        ilp_param_group.child('ILP Appearance Cost').setValue(best_params['ILP Appearance Cost'])
        ilp_param_group.child('ILP Disappearance Cost').setValue(best_params['ILP Disappearance Cost'])
        QMessageBox.information(win, "Success", "New ILP parameters have been applied.")

def initialize_ui_components(deferred=False):
    """Initialize the UI components with proper parameter handling"""
    init_ui_start_time = time.perf_counter()
    if deferred:
        module_logger.info("--- LAZY LOAD: Initializing Parameter Tree ---")
        l_io_main_layout_widget = QWidget()
        l_io_main_hbox = QHBoxLayout(l_io_main_layout_widget)
        ui_elements['d_io'].addWidget(l_io_main_layout_widget)

        ui_elements['pt_io_params'] = pg.parametertree.ParameterTree(showHeader=False)
        l_io_main_hbox.addWidget(ui_elements['pt_io_params'], 1)

        # Define available ILP solvers based on environment check
        ilp_solver_options = ['scipy']
        if GUROBI_AVAILABLE:
            ilp_solver_options.append('gurobi')

        param_group_defs_list = [
            ('General Settings', [('Logging Level', 'list', ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])]),
            ('File Settings (Standard Mode)', [('Raw image extension', 'str'), ('Mask extension', 'str'), ('Mask folder suffix', 'str')]),
            ('File Settings (Advanced Mode)', [('Enable Advanced File Structure', 'bool'), ('Class Definitions (JSON list)', 'text'), ('Raw Image Folder Name', 'str'), ('Generic Mask Folder Name', 'str'), ('Class-Specific Mask Folder Name Pattern', 'str')]),
            ('Display Settings', [('Show id\'s', 'bool'), ('Show tracks', 'bool'), ('Show Mitosis Labels', 'bool')]),
            ('Measurement Settings', [('Pixel scale', 'float'), ('Frames per hour', 'int')]),
            ('Tracking Algorithm', [('Tracking Mode', 'list', ['Backward', 'Forward', 'Basic', 'ILP', 'Trackastra']), ('Min Tracklet Duration', 'int')]),
            ('Trackpy Parameters', [('Trackpy search range', 'int'), ('Trackpy memory', 'int'), ('Trackpy neighbor strategy', 'list', ['KDTree', 'BTree']), ('Mitosis Max Distance Factor', 'float'), ('Mitosis Area Sum Min Factor', 'float'), ('Mitosis Area Sum Max Factor', 'float'), ('Mitosis Daughter Area Similarity', 'float')]),
            ('State Classification', [('Classification Max Centroid Distance', 'float')]),
            ('ILP Tracking', [
              ('ILP Solver', 'list', ilp_solver_options),
              ('ILP Max Search Distance', 'int'),
              ('ILP Transition Cost Weight', 'float'),
              ('ILP Mitosis Cost', 'float'),
              ('ILP Fusion Cost', 'float'),
              ('ILP Appearance Cost', 'float'),
              ('ILP Disappearance Cost', 'float'),
              # Add Gurobi license fields
              ('Gurobi WLSACCESSID', 'str'),
              ('Gurobi WLSSECRET', 'str'),
              ('Gurobi LICENSEID', 'int')
            ]),
            ('Trackastra Settings', [('Trackastra Model', 'list', ['general_2d', 'ctc']), ('Trackastra Linking Mode', 'list', ['greedy', 'ilp', 'greedy_nodiv']), ('Trackastra Device', 'list', ['cuda', 'cpu', 'mps', 'automatic'])]),
            ('Output Settings', [
              ('Save ID-based GIF Overlay', 'bool'),
              ('Save State-based GIF Overlay', 'bool'),
              ('GIF Overlay Alpha', 'float'),
              ('GIF Overlay FPS', 'int'),
              ('Save 16-bit Class Masks', 'bool'),
              ('Class Mask Prefix', 'str'),
              ('Run CTC Metrics Evaluation', 'bool')
            ]),
            ('Analysis Plot Options', [
              ('Save All Analysis Plots', 'bool'),
              ('Save Cell Count Plot', 'bool'),
              ('Save Event Timeline Plot', 'bool'),
              ('Save Cell Cycle Duration Plot', 'bool'),
              ('Save Track Length Distribution Plot', 'bool'),
              ('Save Cell Density Heatmap', 'bool'),
              ('Save Movement Vector Field', 'bool'),
              ('Save Spatial Clustering Plot', 'bool'),
              ('Save Ensemble Outline Plots', 'bool'),
              ('Save Phylogenetic Tree Plots', 'bool'),
              ('Save State Analysis Plots', 'bool'),
              ('Save Publication Lineage Trees', 'bool'),
              ('Save Cell Distribution Stacked Bar Plot', 'bool'),
              ('Cell Distribution Plot Categories', 'list', ['all', 'track_type', 'state', 'both']),
              ('Save Simplified Cell Distribution Plot', 'bool'),
              ('Save Enhanced Cell Distribution Plots', 'bool'),
                              ('Save Temporal Outline Stack', 'bool'),
                ('Temporal Outline Colormap', 'list', COLORMAP_OPTIONS),
                ('Figure DPI', 'int'),
                ('Temporal Outline Scale Factor', 'int'),
                ('Temporal Outline Fast Mode', 'bool'),
              ('Offset Track Plot', 'bool'),
              ('Offset Track Plot Left Color', 'str'),
              ('Offset Track Plot Right Color', 'str'),
              ('Offset Track Plot Output Filename', 'str'),
              ('Time-based Offset Track Plot', 'bool'),
              ('Time-based Offset Track Plot Colormap', 'list', COLORMAP_OPTIONS),
              ('Time-based Offset Track Plot Output Filename', 'str')
            ])
        ]

        param_items_def_list = []
        for group_name_ui, children_defs_ui in param_group_defs_list:
            group_children_ui = []
            for child_def_ui in children_defs_ui:
                name_ui, param_type_ui = child_def_ui[0], child_def_ui[1]
                param_config_from_state = main_app_state['params'].get(name_ui)
                if param_config_from_state is None: continue
                child_param_def = {'name': name_ui, 'type': param_type_ui, 'value': param_config_from_state[0]}
                if param_type_ui == 'list' and len(child_def_ui) > 2: child_param_def['limits'] = child_def_ui[2]
                if len(param_config_from_state) > 2 and isinstance(param_config_from_state[-1], str): child_param_def['tip'] = param_config_from_state[-1]
                group_children_ui.append(child_param_def)
            param_items_def_list.append({'name': group_name_ui, 'type': 'group', 'children': group_children_ui})
        
        ui_elements['p_root_params'] = pg.parametertree.Parameter.create(name='params', type='group', children=param_items_def_list)
        ui_elements['pt_io_params'].setParameters(ui_elements['p_root_params'], showTop=False)
        ui_elements['p_root_params'].sigTreeStateChanged.connect(param_changed_callback)
        
        # Initialize parameter dependencies
        def setup_parameter_dependencies():
            """Set up initial parameter dependencies after the tree is created."""
            try:
                # Get the current state of the master checkbox
                master_enabled = main_app_state['params'].get('Save Comprehensive Analysis Plots', (True,))[0]
                
                # List of dependent parameters
                dependent_params = [
                    'Save Cell Count Plot',
                    'Save Event Timeline Plot', 
                    'Save Cell Cycle Duration Plot',
                    'Save Track Length Distribution Plot',
                    'Save Cell Density Heatmap',
                    'Save Movement Vector Field',
                    'Save Spatial Clustering Plot',
                    'Save Ensemble Outline Plots',
                    'Save Phylogenetic Tree Plots',
                    'Save State Analysis Plots',
                    'Save Publication Lineage Trees',
                    'Save Cell Distribution Stacked Bar Plot',
                    'Cell Distribution Plot Categories',
                    'Save Simplified Cell Distribution Plot',
                    'Save Enhanced Cell Distribution Plots',
                    'Save Temporal Outline Stack',
                    'Temporal Outline Colormap',
                    'Figure DPI',
                    'Temporal Outline Scale Factor',
                    'Offset Track Plot',
                    'Offset Track Plot Left Color',
                    'Offset Track Plot Right Color',
                    'Offset Track Plot Output Filename',
                    'Time-based Offset Track Plot',
                    'Time-based Offset Track Plot Colormap',
                    'Time-based Offset Track Plot Output Filename'
                ]
                
                # Find the Analysis Plot Options group and set initial states
                analysis_group = ui_elements['p_root_params'].param('Analysis Plot Options')
                if analysis_group:
                    # Set all individual checkboxes to read-only if comprehensive analysis is disabled
                    for dependent_param in dependent_params:
                        try:
                            child_param = analysis_group.child(dependent_param)
                            if child_param:
                                child_param.setReadonly(not master_enabled)
                        except Exception as e:
                            logging.debug(f"Could not find parameter '{dependent_param}' during initialization: {e}")
            except Exception as e:
                logging.debug(f"Could not set up parameter dependencies: {e}")
        
        # Set up dependencies after parameter tree is ready
        setup_parameter_dependencies()
        
        setup_buttons(l_io_main_hbox)
        module_logger.info(f"--- TIME: Parameter Tree populated in {time.perf_counter() - init_ui_start_time:.4f} s ---")
    else:
        # Initialize other non-deferred components
        module_logger.info("--- Initializing non-deferred UI components ---")
        setup_image_views()
        setup_tracking_table()
        setup_lineage_tree()
        if 'cell_ui_callbacks' in sys.modules:
            cell_ui_callbacks.setup_hover_detection(ui_elements['pi_raw_img'], ui_elements['pi_mask'], main_app_state, ui_elements)
        module_logger.info(f"--- TIME: Non-deferred UI setup in {time.perf_counter() - init_ui_start_time:.4f} s ---")


def setup_buttons(parent_layout):
    """Setup the button grid using stable internal keys."""
    buttons_container_widget = QWidget()
    l_buttons_grid_layout = QGridLayout(buttons_container_widget)
    parent_layout.addWidget(buttons_container_widget, 2)
    button_style_sheet = "QPushButton { padding: 10px; font-size: 9pt; margin: 2px; } QPushButton:disabled { background-color: #d3d3d3; color: #808080; }"

    button_definitions = [
        {'key': 'open_data_folder', 'text': "Open Raw Data Folder", 'action': lambda: cell_file_operations.load_data_from_folder(main_app_state, ui_elements, cell_ui_callbacks, cell_undoredo), 'grid_pos': (0, 0, 1, 2), 'enabled': True},
        {'key': 'load_previous_results', 'text': "Load Previous Results", 'action': lambda: cell_file_operations.load_previous_results(main_app_state, ui_elements, cell_ui_callbacks, cell_undoredo), 'grid_pos': (0, 2, 1, 2), 'enabled': True},
        {'key': 'run_tracking', 'text': "Run Tracking", 'action': lambda: cell_tracking_orchestrator.initiate_cell_tracking(main_app_state, ui_elements, cell_ui_callbacks, cell_undoredo), 'grid_pos': (1, 0, 1, 2), 'enabled': False},
        {'key': 'batch_tracking', 'text': "Batch Tracking", 'action': lambda: cell_file_operations.perform_batch_tracking(main_app_state, ui_elements), 'grid_pos': (1, 2, 1, 2), 'enabled': True},
        {'key': 'select_all_visible', 'text': "Select All Visible", 'action': lambda: cell_ui_callbacks.handle_select_all_clicked(main_app_state, ui_elements, cell_ui_callbacks), 'grid_pos': (2, 0, 1, 1), 'enabled': False},
        {'key': 'select_none_visible', 'text': "Select None Visible", 'action': lambda: cell_ui_callbacks.handle_select_none_clicked(main_app_state, ui_elements, cell_ui_callbacks), 'grid_pos': (2, 1, 1, 1), 'enabled': False},
        {'key': 'select_complete_tracks', 'text': "Select Complete Tracks", 'action': lambda: cell_ui_callbacks.handle_select_complete_clicked(main_app_state, ui_elements, cell_ui_callbacks), 'grid_pos': (2, 2, 1, 1), 'enabled': False},
        {'key': 'select_mitosis_tracks', 'text': "Select Mitosis Tracks", 'action': lambda: cell_ui_callbacks.handle_select_mitosis_tracks_clicked(main_app_state, ui_elements, cell_ui_callbacks), 'grid_pos': (3, 0, 1, 1), 'enabled': False},
        {'key': 'select_fusion_tracks', 'text': "Select Fusion Tracks", 'action': lambda: cell_ui_callbacks.handle_select_fusion_tracks_clicked(main_app_state, ui_elements, cell_ui_callbacks), 'grid_pos': (3, 1, 1, 1), 'enabled': False},
        {'key': 'select_singular_tracks', 'text': "Select Singular Tracks", 'action': lambda: cell_ui_callbacks.handle_select_singular_tracks_clicked(main_app_state, ui_elements, cell_ui_callbacks), 'grid_pos': (3, 2, 1, 1), 'enabled': False},
        {'key': 'relink_visible_tracks', 'text': "Relink Visible Tracks", 'action': lambda: cell_lineage_operations.handle_relink_visible_tracks(main_app_state, ui_elements, cell_ui_callbacks, cell_undoredo), 'grid_pos': (3, 3, 1, 1), 'enabled': False},
        {'key': 'calculate_stats', 'text': "Calculate Stats", 'action': lambda: cell_ui_callbacks.handle_calculate_stats_clicked(main_app_state, ui_elements), 'grid_pos': (4, 0, 1, 1), 'enabled': False},
        {'key': 'save_results', 'text': "Save Results", 'action': lambda: save_results(
            main_app_state.get('path_in'),
            main_app_state.get('trj'),
            main_app_state.get('col_tuple'),
            main_app_state.get('col_weights'),
            main_app_state.get('id_masks'),
            main_app_state.get('cell_ids'),
            main_app_state.get('background_id'),
            main_app_state.get('color_list'),
            main_app_state.get('cell_color_idx'),
            main_app_state.get('cell_visibility'),
            main_app_state['params']['Pixel scale'][0],
            main_app_state['params']['Pixel scale'][3],
            main_app_state['params']['Show id\'s'][0],
            True,
            main_app_state['params']['Show tracks'][0],
            main_app_state['params']['Mask extension'][0],
            cell_lineage=main_app_state.get('cell_lineage'),
            ancestry_map=main_app_state.get('ancestry'),
            use_thick_line=True,
            show_mitosis=main_app_state['params']['Show Mitosis Labels'][0],
            lineage_plot_widget=ui_elements.get('lineage_tree_widget'),
            command_output_log=main_app_state.get('captured_tracking_log'),
            main_app_state=main_app_state,
            ui_elements=ui_elements
        ), 'grid_pos': (4, 1, 1, 1), 'enabled': False},
        {'key': 'optimize_ilp', 'text': "Optimize ILP", 'action': lambda: handle_optimize_ilp_clicked(main_app_state, ui_elements), 'grid_pos': (4, 2, 1, 1), 'enabled': False},
    ]

    for btn_def in button_definitions:
        btn = QPushButton(btn_def['text'])
        btn.clicked.connect(btn_def['action'])
        btn.setEnabled(btn_def['enabled'])
        btn.setStyleSheet(button_style_sheet)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        l_buttons_grid_layout.addWidget(btn, *btn_def['grid_pos'])
        ui_elements['button_widgets_map'][btn_def['key']] = btn
def setup_image_views():
    """Setup the image view widgets"""
    l_img_views_layout = pg.LayoutWidget()
    ui_elements['d_img'].addWidget(l_img_views_layout)

    ui_elements['pi_raw_img'] = pg.PlotItem(title="Raw Image - Frame 0")
    ui_elements['v_raw_img'] = PatchedImageView(view=ui_elements['pi_raw_img'])
    main_app_state['v_raw_img_original_state'] = ui_elements['v_raw_img'].ui.histogram.gradient.saveState()
    ui_elements['v_raw_img'].sigTimeChanged.connect(
        lambda time_val, _: cell_ui_callbacks.handle_time_changed_raw_img(time_val, _, main_app_state, ui_elements,
                                                                          cell_ui_callbacks))
    l_img_views_layout.addWidget(ui_elements['v_raw_img'], row=0, col=0)

    ui_elements['pi_mask'] = pg.PlotItem(title="Mask - Frame 0")
    ui_elements['v_mask'] = PatchedImageView(view=ui_elements['pi_mask'])
    main_app_state['v_mask_original_state'] = ui_elements['v_mask'].ui.histogram.gradient.saveState()
    ui_elements['v_mask'].sigTimeChanged.connect(
        lambda time_val, _: cell_ui_callbacks.handle_time_changed_mask(time_val, _, main_app_state, ui_elements,
                                                                       cell_ui_callbacks))
    l_img_views_layout.addWidget(ui_elements['v_mask'], row=0, col=1)

    for v in [ui_elements['v_raw_img'], ui_elements['v_mask']]:
        v.setImage(np.zeros((100, 100, 3), dtype=np.uint8))
        v.ui.histogram.hide();
        v.ui.menuBtn.hide();
        v.ui.roiBtn.hide()
        # Performance optimizations for PyQtGraph
        # Note: autoLevels and autoRange are set via setImage parameters, not as properties


def setup_tracking_table():
    """Setup the cell tracking table"""
    tracks_dock_layout_v = QVBoxLayout()
    ui_elements['table_cell_selection'] = QTableWidget()
    ui_elements['table_cell_selection'].setColumnCount(6)
    ui_elements['table_cell_selection'].setHorizontalHeaderLabels(
        ["Cell ID", "Visible", "Parent ID(s)", "Daughters", "State", "Original Seg. Label"])
    ui_elements['table_cell_selection'].horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    ui_elements['table_cell_selection'].setSelectionBehavior(QAbstractItemView.SelectRows)
    # Optimize table performance by reducing signal connections
    ui_elements['table_cell_selection'].itemChanged.connect(
        lambda item: cell_ui_callbacks.handle_cell_table_item_changed(item, main_app_state, ui_elements,
                                                                      cell_ui_callbacks, cell_undoredo))
    ui_elements['table_cell_selection'].cellClicked.connect(
        lambda r, c: cell_ui_callbacks.handle_cell_table_cell_clicked(r, c, main_app_state, ui_elements,
                                                                      cell_ui_callbacks))
    # Performance optimization: Set table properties for better scrolling
    ui_elements['table_cell_selection'].setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
    ui_elements['table_cell_selection'].setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
    tracks_dock_layout_v.addWidget(ui_elements['table_cell_selection'])

    ui_elements['b_merge_track'] = QPushButton("Merge Selected Track with Parent")
    ui_elements['b_merge_track'].clicked.connect(
        lambda: cell_lineage_operations.handle_merge_track_with_parent(main_app_state, ui_elements, cell_ui_callbacks,
                                                                       cell_undoredo))
    ui_elements['b_merge_track'].setEnabled(False)
    tracks_dock_layout_v.addWidget(ui_elements['b_merge_track'])
    ui_elements['button_widgets_map']['merge_selected_track_with_parent'] = ui_elements['b_merge_track']
    
    # Add export cell editor table button
    ui_elements['b_export_cell_table'] = QPushButton("Export Cell Editor Table to CSV")
    ui_elements['b_export_cell_table'].clicked.connect(
        lambda: export_cell_editor_table(main_app_state, ui_elements))
    tracks_dock_layout_v.addWidget(ui_elements['b_export_cell_table'])

    tracks_dock_content_widget = QWidget()
    tracks_dock_content_widget.setLayout(tracks_dock_layout_v)
    ui_elements['d_tracks'].addWidget(tracks_dock_content_widget)


def setup_lineage_tree():
    """Setup the lineage tree widget"""
    ui_elements['lineage_tree_widget'] = LineageTreeWidget()
    ui_elements['d_lineage'].addWidget(ui_elements['lineage_tree_widget'])
    ui_elements['lineage_tree_widget'].nodeOperationRequested.connect(
        lambda did, tid, op: cell_lineage_operations.process_node_operation_request(did, tid, op, main_app_state,
                                                                                    ui_elements, cell_ui_callbacks,
                                                                                    cell_undoredo))
    ui_elements['lineage_tree_widget'].nodeClickedInView.connect(
        lambda tid, fid: cell_ui_callbacks.handle_lineage_node_clicked_in_main(tid, fid, main_app_state, ui_elements,
                                                                               cell_ui_callbacks))
    ui_elements['lineage_tree_widget'].nodeBreakRequested.connect(
        lambda tid, fid: cell_lineage_operations.process_node_break_request(tid, fid, main_app_state, ui_elements,
                                                                            cell_ui_callbacks, cell_undoredo))
    ui_elements['lineage_tree_widget'].insertAndSplitRequested.connect(
        lambda did, tid, ins_f: cell_lineage_operations.process_insert_dragged_and_split_target(did, tid, ins_f,
                                                                                                main_app_state,
                                                                                                ui_elements,
                                                                                                cell_ui_callbacks,
                                                                                                cell_undoredo))
    ui_elements['lineage_tree_widget'].splitAndFuseHeadRequested.connect(
        lambda oid, b_fid, t_fid: cell_lineage_operations.process_split_and_fuse_head_request(oid, b_fid, t_fid,
                                                                                              main_app_state,
                                                                                              ui_elements,
                                                                                              cell_ui_callbacks,
                                                                                              cell_undoredo))


def setup_undo_redo_actions_main():
    """Setup undo/redo actions in the menu bar"""
    main_window = ui_elements['win']
    edit_menu = main_window.menuBar().addMenu("&Edit")

    ui_elements['undo_action'] = QAction("Undo", main_window, shortcut=QKeySequence.Undo)
    ui_elements['undo_action'].triggered.connect(
        lambda: cell_undoredo.undo_last_operation(main_app_state, ui_elements, cell_ui_callbacks, ui_elements))
    edit_menu.addAction(ui_elements['undo_action'])

    ui_elements['redo_action'] = QAction("Redo", main_window, shortcut=QKeySequence.Redo)
    ui_elements['redo_action'].triggered.connect(
        lambda: cell_undoredo.redo_last_operation(main_app_state, ui_elements, cell_ui_callbacks, ui_elements))
    edit_menu.addAction(ui_elements['redo_action'])

    # Add refresh lineage tree action
    ui_elements['refresh_lineage_action'] = QAction("Refresh Lineage Tree", main_window, shortcut="F5")
    ui_elements['refresh_lineage_action'].triggered.connect(
        lambda: cell_ui_callbacks.force_refresh_lineage_tree(main_app_state, ui_elements))
    edit_menu.addAction(ui_elements['refresh_lineage_action'])


def export_cell_editor_table(main_app_state, ui_elements):
    """Export the current cell editor table to a CSV file."""
    try:
        from PyQt5.QtWidgets import QFileDialog
        import pandas as pd
        import numpy as np
        
        # Get save location from user
        file_path, _ = QFileDialog.getSaveFileName(
            ui_elements.get('win'),
            "Export Cell Editor Table",
            "cell_editor_table.csv",
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        # Get all the data needed for the cell editor table
        trj = main_app_state.get('trj')
        if trj is None or trj.empty:
            QMessageBox.warning(ui_elements.get('win'), "No Data", "No tracking data available to export.")
            return
        
        # Debug: Print available columns to help troubleshoot
        print(f"Available columns in tracking data: {list(trj.columns)}")
        print(f"Number of cells: {len(trj['particle'].unique())}")
        if 'angle' in trj.columns:
            print(f"Angle column found with {trj['angle'].notna().sum()} non-null values")
        elif 'orientation' in trj.columns:
            print(f"Orientation column found with {trj['orientation'].notna().sum()} non-null values")
        else:
            print("WARNING: Neither 'angle' nor 'orientation' column found in tracking data")
            print("This means the tracking data needs to be enriched with geometric properties.")
            print("Try running tracking again or check if the data was processed with regionprops.")
        
        cell_visibility = main_app_state.get('cell_visibility', {})
        ancestry = main_app_state.get('ancestry', {})
        cell_lineage = main_app_state.get('cell_lineage', {})
        track_states = main_app_state.get('track_states', {})
        
        # Create comprehensive cell editor table
        cell_editor_data = []
        unique_cell_ids = sorted(trj['particle'].unique())
        
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
            
            # Get average position
            avg_x = float(cell_data['x'].mean()) if len(cell_data) > 0 else 0.0
            avg_y = float(cell_data['y'].mean()) if len(cell_data) > 0 else 0.0
            
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
            try:
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
                    
                    # Debug: Print values for first few cells
                    if cell_id_int <= 3:  # Only print for first 3 cells to avoid spam
                        print(f"Cell {cell_id_int}: x_std={x_displacement_std:.3f}, y_std={y_displacement_std:.3f}")
            except Exception as e:
                print(f"Error calculating displacement std for cell {cell_id_int}: {e}")
                x_displacement_std = 0.0
                y_displacement_std = 0.0
            
            # Get orientation angle statistics (in radians)
            avg_angle_rad = 0.0
            angle_std_rad = 0.0
            try:
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
                        
                        # Debug: Print angle values for first few cells
                        if cell_id_int <= 3:  # Only print for first 3 cells to avoid spam
                            print(f"Cell {cell_id_int}: avg_angle_rad={avg_angle_rad:.3f}, angle_std_rad={angle_std_rad:.3f} (from {angle_column})")
                else:
                    # Debug: Print when angle data is missing
                    if cell_id_int <= 3:  # Only print for first 3 cells to avoid spam
                        print(f"Cell {cell_id_int}: No angle/orientation data available")
            except Exception as e:
                print(f"Error calculating orientation stats for cell {cell_id_int}: {e}")
                avg_angle_rad = 0.0
                angle_std_rad = 0.0
            
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
                'average_x': avg_x,
                'average_y': avg_y,
                'x_displacement': x_displacement,
                'y_displacement': y_displacement,
                'total_displacement': total_displacement,
                'x_displacement_std': x_displacement_std,
                'y_displacement_std': y_displacement_std,
                'total_detections': len(cell_data),
                'avg_orientation_rad': avg_angle_rad,
                'orientation_std_rad': angle_std_rad
            })
        
        # Save complete cell editor table with proper formatting
        cell_editor_df = pd.DataFrame(cell_editor_data)
        
        # Format displacement and position values to 2 decimal places
        if 'x_displacement' in cell_editor_df.columns:
            cell_editor_df['x_displacement'] = cell_editor_df['x_displacement'].round(2)
        if 'y_displacement' in cell_editor_df.columns:
            cell_editor_df['y_displacement'] = cell_editor_df['y_displacement'].round(2)
        if 'total_displacement' in cell_editor_df.columns:
            cell_editor_df['total_displacement'] = cell_editor_df['total_displacement'].round(2)
        if 'average_x' in cell_editor_df.columns:
            cell_editor_df['average_x'] = cell_editor_df['average_x'].round(2)
        if 'average_y' in cell_editor_df.columns:
            cell_editor_df['average_y'] = cell_editor_df['average_y'].round(2)
        
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
        
        cell_editor_df.to_csv(file_path, index=False, float_format='%.3f')
        
        QMessageBox.information(
            ui_elements.get('win'), 
            "Export Successful", 
            f"Cell editor table exported to:\n{file_path}\n\nExported {len(cell_editor_df)} cells with orientation data."
        )
        
    except Exception as e:
        QMessageBox.critical(
            ui_elements.get('win'), 
            "Export Error", 
            f"Error exporting cell editor table:\n{str(e)}"
        )


def launch_tui_tracker(yaml_config=None, parent_window=None, external_log_handler=None):
    """ Main entry point for launching TUI Tracker from main.py """
    return initialize_cell_tracking_app(yaml_config, parent_window, external_log_handler)


if __name__ == '__main__':
    try:
        app_instance = QApplication.instance() or QApplication(sys.argv)
        window = initialize_cell_tracking_app()

        from PyQt5.QtWidgets import QDesktopWidget

        screen_geometry = QDesktopWidget().availableGeometry(QDesktopWidget().primaryScreen())
        window.move(screen_geometry.center() - window.rect().center())
        window.show()


        def cleanup_standalone_tui():
            if 'gui_log_handler' in ui_elements and ui_elements['gui_log_handler']:
                ui_elements['gui_log_handler'].early_cleanup()
            if 'console_log_handler' in ui_elements and ui_elements['console_log_handler']:
                logging.getLogger().removeHandler(ui_elements['console_log_handler'])
                if hasattr(ui_elements['console_log_handler'], 'close'): ui_elements['console_log_handler'].close()
            if 'log_capture' in ui_elements and ui_elements['log_capture']:
                ui_elements['log_capture'].shutdown()


        atexit.register(cleanup_standalone_tui)
        sys.exit(app_instance.exec_())
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
