# Tui Tracker: Cell Tracking and Lineage Analysis Tool

<img width="2558" height="1541" alt="Screenshot" src="https://github.com/user-attachments/assets/64b4e13f-f65b-4621-bdbd-f9ed874e6959" />


## Overview


Tui Tracker is a Python-based tool designed for tracking cells in time-lapse microscopy image sequences, analyzing their lineage (including mitosis events), and visualizing the results. It provides a graphical user interface (GUI) built with PyQt5 and pyqtgraph for interactive data exploration, parameter tuning, and manual correction of tracking data.

**Advantages of the ILP Method (Compared to TrackAstra, TrackPy, and other software):**
- Robustly handles complex biological events, including **multiple mitosis and fusion events** within the same sequence, which many other trackers cannot process simultaneously.
- Provides **interactive lineage plot visualization**, allowing users to explore and understand cell ancestry and division history in detail.
- **Automatically evaluates CTC metrics** (tracking, segmentation, lineage accuracy) if manually annotated ground truth is available, streamlining benchmarking and validation.
- Supports **manual verification and editing of lineage plot nodes** via the GUI, enabling correction of tracking or lineage errors and fine-tuning of results.

---

## Main Modules and Functions

- **cell_main.py**: Main application and GUI logic.
- **cell_tracking.py**: Core tracking algorithms (TrackPy, ILP [default], TrackAstra).
- **cell_drawing.py**: Visualization and mask/animation generation (color masks, state masks, temporal outline stack, etc.).
- **cell_io.py**: Input/output, file reading/writing, result saving, statistics calculation.
- **cell_lineage_operations.py**: Lineage and ancestry management.
- **cell_tracking_orchestrator.py**: Orchestrates tracking pipeline, batch, and advanced modes.
- **cell_file_operations.py**: File and batch operations.
- **lineage_tree.py**: Lineage tree widget and visualization.

---

## Tracking Methods

- **ILP (Integer Linear Programming)**: Default method. Global optimization-based tracking. Can use Gurobi (if available) for acceleration.
- **TrackPy**: Fast, robust, particle-tracking-based method.
- **TrackAstra**: Deep learning-based tracking (if available).
- **How to select**: In the GUI parameter tree, set “Tracking Mode” (e.g., “ILP”, “TrackPy”, “TrackAstra”).
- **No particular workflow is required**; users can choose any method as needed.
- **Configurable parameters**: Search range, memory, cost weights, etc.

---

## Output Data Format & Structure

### Main Output Folder
- Named: `{InputFolder}_Exp_{YYYY-MM-DDTHHMMSS}_{tracking_mode}`

### Key Output Files/Folders
- **tracks.csv**: All cell features, positions, and track IDs per frame.
- **cell_editor_table.csv**: Per-cell summary table with the following columns:
    - `cell_id`: Unique cell/track identifier
    - `visible`: Whether the cell is visible/enabled in the editor
    - `parent_ids`: Parent cell IDs (for mitosis/fusion events)
    - `daughters`: Daughter cell IDs (for mitosis/fusion events)
    - `state`: Cell state/class (if available)
    - `original_segmentation_labels`: Original mask label(s) for the cell
    - `start_frame`, `end_frame`: First and last frame where the cell appears
    - `track_length_frames`: Number of frames the cell is tracked
    - `x_displacement`, `y_displacement`, `total_displacement`: Net displacement from first to last frame
    - `x_displacement_std`, `y_displacement_std`: Standard deviation of frame-to-frame x and y displacements (measures movement variability)
    - `avg_orientation_rad`, `orientation_std_rad`: Average and standard deviation of the cell's orientation angle (in radians) across all frames, calculated from the `orientation` property of `skimage.measure.regionprops` (describes predominant cell elongation direction and its variability)
    - `total_detections`: Number of detections/appearances for the cell

  These columns allow for advanced analysis, including assessment of cell motility consistency, shape dynamics, and quality control for tracking and segmentation.
- **experiment_parameters.csv**: All parameters used for this run.
- **tracking_statistics.txt**: Summary of tracking and lineage statistics (including lifecycle, doubling time, etc.).
- **lineage_relationships.csv**: Parent-daughter relationships.
- **ctc_evaluation_results.txt**: CTC challenge metrics (if GT available).

### Masks and Visualizations
- **Id_masks_raw_per_frame_original/**: Raw ID masks (cell ID per pixel, per frame).
- **Masks_segmented_per_frame_original/**: Colorized segmentation masks (all objects, colored by ID).
- **Tracks_per_frame_original/**: Track overlays (all tracks, colored by ID).
- **id_mask_colored_animation_original.gif**: Animated colored ID masks (with/without text).
- **id_mask_colored_animation_no_text.gif**: Animated colored ID masks (no text).
- **masks_segmented_animation_original.gif**: Animated colorized masks.
- **tracks_animation_original.gif**: Animated tracks.
- **all_tracks_overview_original.png**: Overview of all tracks.

### Temporal Outline Stack
- **temporal_outline_stack.png**: All cell outlines overlaid, colored by time, with a colorbar. Colormap is user-configurable via the GUI.

### State Mask Outputs
- **State_masks_per_frame/**: 16-bit PNG masks, each pixel = state index (0=background). State labels can be strings; mapping is handled automatically.
- **State_masks_color_per_frame/**: Color mask per frame, all objects colored by state (consistent colors across time; colormap user-configurable via GUI).
- **State_masks_by_type/State_X/**: For each state X:
    - `state_X_mask_000.png`: Binary mask for state X in frame 0.
    - `state_X_color_mask_000.png`: Color mask for state X in frame 0 (all objects of X filled with same color).
- **A legend or mapping file is generated if needed to relate state labels to indices/colors.**

### CTC Challenge Format
- **{InputFolder}_RES/**: CTC-compliant results (res_track.txt, maskNNN.tif, etc.)
- **If CTC ground truth is available (man_track.txt), CTC evaluation is run and results are saved. If not, this step is skipped and a message is shown.**

### CTC-Compliant Tracking File Format (res_track.txt / man_track.txt)

The CTC-compliant tracking file is a plain text file with **four columns** per line, separated by spaces:

| Column | Name           | Description                                                                 |
|--------|----------------|-----------------------------------------------------------------------------|
| 1      | L (Label/ID)   | Unique integer ID for the track (cell/particle).                            |
| 2      | B (Begin)      | First frame index where this track appears (inclusive, 0-based).            |
| 3      | E (End)        | Last frame index where this track appears (inclusive, 0-based).             |
| 4      | P (Parent ID)  | ID of the parent track (integer). Use 0 if the track has no parent (root).  |

- Each line describes a single cell track.
- The parent ID (P) links daughter tracks to their parent for mitosis/fusion events. If a track is a root (no parent), P is 0.
- Frame indices (B, E) are **inclusive** and typically 0-based (first frame is 0).

**Example:**
```
1 0 10 0
2 5 15 1
3 12 20 2
```
- Track 1 appears from frame 0 to 10 and has no parent.
- Track 2 appears from frame 5 to 15 and its parent is track 1.
- Track 3 appears from frame 12 to 20 and its parent is track 2.

### Offset Track Plot
- **offset_track_plot.png** (or user-specified filename):
    - Shows all tracks offset so the centroid of all objects in the first frame is at (0,0).
    - Each track is colored by its final x position: left of origin (default: black), right of origin (default: red).
    - Axis lines at (0,0) for reference.
    - Customizable via GUI parameters:
        - **Offset Track Plot**: Enable/disable this output.
        - **Offset Track Plot Left Color**: Color for tracks ending left of origin.
        - **Offset Track Plot Right Color**: Color for tracks ending right of origin.
        - **Offset Track Plot Output Filename**: Output filename for the plot.

### Comprehensive Analysis Plots

The tool generates a comprehensive set of analysis plots in the `analysis_plots/` subfolder, providing detailed insights into cell behavior, lineage dynamics, and spatial patterns. All plots are saved as high-quality PNG images.

#### Temporal Analysis Plots

- **cell_count_over_time.png**:
    - Shows the number of cells present in each frame over time.
    - Useful for understanding population dynamics, growth patterns, and cell division events.
    - Y-axis: Number of cells, X-axis: Frame number.

- **event_timeline.png**:
    - Visualizes mitosis and fusion events over time.
    - Mitosis events are marked in red, fusion events in blue.
    - Helps identify temporal patterns in cell division and fusion behavior.
    - Y-axis: Event count, X-axis: Frame number.

- **cell_cycle_duration.png**:
    - Histogram showing the distribution of cell cycle durations.
    - Calculated from parent-daughter relationships in the lineage data.
    - Useful for understanding cell division timing and variability.
    - X-axis: Cell cycle duration (frames), Y-axis: Frequency.

- **track_length_distribution.png**:
    - Histogram of track durations (how long each cell was tracked).
    - Helps identify tracking quality and cell lifespan patterns.
    - X-axis: Track length (frames), Y-axis: Frequency.

#### Spatial Analysis Plots

- **cell_density_heatmap.png**:
    - 2D heatmap showing cell density across the image field over time.
    - Useful for identifying spatial clustering, migration patterns, and crowded regions.
    - Color intensity represents cell density, with a colorbar for reference.

- **movement_vector_field.png**:
    - Vector field visualization showing cell movement patterns.
    - Arrows indicate direction and magnitude of cell movement between frames.
    - Helps identify collective migration, directional movement, and spatial organization.

- **spatial_clustering.png**:
    - Analysis of spatial clustering patterns in cell positions.
    - Shows how cells are distributed relative to each other across time.
    - Useful for identifying cell aggregation, dispersion, or organized spatial patterns.

#### Cell Shape and Morphology Analysis

- **ensemble_outline_polar.png**:
    - Polar plot showing ensemble cell outline changes over time.
    - Each angle represents a direction from the cell center, with radius showing outline distance.
    - Color-coded by time to show how cell shape evolves.
    - Useful for understanding cell morphology dynamics and shape changes.

- **ensemble_outline_heatmap.png**:
    - Heatmap visualization of ensemble cell outline changes.
    - X-axis: Angle (0-360°), Y-axis: Time (frames).
    - Color intensity represents outline distance from center.
    - Alternative view to the polar plot for analyzing cell shape evolution.

#### Lineage and Phylogenetic Analysis

- **lineage_tree_track_segments_publication.png**:
    - Publication-ready lineage tree showing all cell tracks as horizontal lines.
    - Tracks are arranged vertically by lineage, with parent-child connections shown.
    - All tracks are colored uniformly (default: black) for clean visualization.
    - Time axis shows frame progression, lineage axis shows different cell lineages.

- **lineage_tree_class_type_publication.png**:
    - Similar to track segments plot but colored by cell state/class.
    - Different cell states (e.g., normal, mitosis, fusion) are shown in different colors.
    - Includes a legend showing state-color mappings.
    - Useful for understanding how cell states relate to lineage structure.

- **phylogenetic_tree_track_segments.png**:
    - Traditional phylogenetic tree layout showing cell lineage relationships.
    - Branch lengths represent track duration (time).
    - Terminal nodes (tips) represent end points of cell tracks.
    - All branches colored uniformly for clean visualization.

- **phylogenetic_tree_class_type.png**:
    - Phylogenetic tree with branches colored by cell state/class.
    - Terminal nodes are colored according to the most common state of that cell.
    - Includes hover information and legend for state identification.
    - Useful for understanding state evolution along lineages.

#### State Analysis Plots (if cell states are available)

- **state_transition_diagram.png**:
    - Network diagram showing transitions between different cell states.
    - Nodes represent states, edges represent transitions.
    - Edge thickness indicates transition frequency.
    - Useful for understanding cell state dynamics and transition patterns.

- **state_distribution_over_time.png**:
    - Stacked area plot showing the distribution of cell states over time.
    - Each area represents a different cell state, colored distinctly.
    - Y-axis: Number of cells, X-axis: Frame number.
    - Helps visualize how cell populations change state over time.

#### GIF Overlays

- **id_based_overlay.gif**:
    - Animated overlay showing cell IDs overlaid on raw images.
    - Each cell ID gets a unique color that remains consistent across frames.
    - Useful for visualizing cell tracking and movement patterns.
    - Configurable transparency and frame rate.

- **state_based_overlay.gif**:
    - Animated overlay showing cell states overlaid on raw images.
    - Each state class gets the same color across all cells and frames.
    - Missing states are assigned "unknown" and colored accordingly.
    - Useful for visualizing cell state patterns and transitions over time.

#### Additional Outputs

- **class_masks_16bit/**:
    - 16-bit PNG masks for different cell states/classes for each timeframe.
    - Each pixel value corresponds to a state class ID.
    - Includes a mapping file (`class_mask_class_mapping.txt`) relating class IDs to state names.
    - Useful for downstream analysis and state-based processing.

- **track_types/**:
    - CSV files and tree plots for different track types:
        - **singular_tracks.csv**: Tracks that neither divide nor fuse.
        - **mitosis_tracks.csv**: Tracks that undergo cell division.
        - **fusion_tracks.csv**: Tracks that undergo fusion events.
    - Tree visualizations for each track type showing their lineage structure.

---

## CTC Challenge Evaluation and Metrics

This tool supports automatic evaluation using the [Cell Tracking Challenge (CTC)](http://celltrackingchallenge.net/) metrics, which are widely used for benchmarking cell tracking and lineage analysis algorithms.

### How Evaluation Works
- **Automatic Trigger:** If CTC-format ground truth (GT) is available (i.e., a `man_track.txt` file and corresponding masks in a `TRA` subfolder), the tool will automatically run CTC evaluation when you save results.
- **Library Used:** Evaluation is performed using the [`py-ctcmetrics`](https://github.com/CellTrackingChallenge/py-ctcmetrics) Python library, which implements the official CTC metrics and additional modern tracking metrics.
- **Results Location:** Evaluation results are saved as `ctc_evaluation_results.txt` in your experiment output folder. If evaluation fails, a log file is created with the error message.

### What is Evaluated
- **Input:** The tool generates a CTC-compliant results folder (`*_RES/`) containing `res_track.txt` and per-frame masks (`maskNNN.tif`).
- **Comparison:** These results are compared to the ground truth using py-ctcmetrics, which computes a suite of tracking and segmentation metrics.

### Key Metrics Explained
- **TRA (Tracking Accuracy):** Measures how well cell tracks are reconstructed, accounting for correct linking, splits, and merges.
- **SEG (Segmentation Accuracy):** Measures the overlap between predicted and ground truth cell masks.
- **DET (Detection Accuracy):** Measures the accuracy of cell detection (regardless of track correctness).
- **IDF1:** The ratio of correctly identified detections over the average number of ground-truth and computed detections (identity F1 score).
- **MOTA (Multiple Object Tracking Accuracy):** Combines false positives, missed targets, and identity switches into a single score.
- **HOTA (Higher Order Tracking Accuracy):** A modern metric that balances detection, association, and localization accuracy.
- **CHOTA:** Class-balanced HOTA, for datasets with multiple cell types/states.
- **LNK:** Measures the accuracy of parent-daughter (lineage) relationships.
- **CT, TF, BC, CCA:** Additional metrics for cell tracking and lineage (see CTC documentation for details).
- **MT (Mostly Tracked) / ML (Mostly Lost):** Fraction of ground-truth tracks that are mostly tracked (≥80% overlap) or mostly lost (<20% overlap).
- **AOGM, IDSW, TP, FN, FP, Precision, Recall:** Standard tracking and detection statistics.
- **BIO(0):** The "Best Identification Overlap" at threshold 0. Measures the best possible overlap-based matching between predicted and ground-truth tracks, regardless of identity switches. Higher values indicate better segmentation and association quality.
- **OP_CLB(0):** The "Optimal Cell Lineage Benchmark" at threshold 0. A composite metric that summarizes both tracking and lineage accuracy, combining parent-daughter relationships and segmentation overlap. Higher values indicate better overall performance. Saved as `OP_CLB(0)` in the results file.

### Customization and Troubleshooting
- **Metrics List:** The tool requests a broad set of metrics from py-ctcmetrics, including all standard CTC metrics and modern extensions (HOTA, IDF1, etc.).
- **Threading:** Evaluation uses all available CPU threads by default for speed.
- **If GT is missing:** Evaluation is skipped and a message is shown in the log/output.
- **If using TrackPy:** For CTC evaluation to work, set the `Trackpy memory` parameter to `0` (see Known Issues).

### References
- [Cell Tracking Challenge](http://celltrackingchallenge.net/)
- [py-ctcmetrics GitHub](https://github.com/CellTrackingChallenge/py-ctcmetrics)
- [CTC Metrics Documentation](https://github.com/CellTrackingChallenge/py-ctcmetrics#metrics)

---

### Key Parameters (Configurable in GUI)

* **File Settings:** `Raw image extension`, `Mask extension`, `Mask folder suffix`.
* **Display Settings:** `Show id's` (on canvas), `Show tracks` (on canvas), `Show Mitosis Labels` (in hover tooltips), `Lineage View Type`.
* **Measurement Settings:** `Pixel scale`, `Frames per hour`.
* **Tracking Algorithm (TrackPy):** `Trackpy search range`, `Trackpy memory`, `Trackpy neighbor strategy`.
* **Mitosis Detection:** Parameters controlling the criteria for identifying cell divisions (distance, area ratios, similarity).

### Output Settings and Configuration

The tool provides extensive configuration options for controlling what outputs are generated and how they appear. These settings can be found in the GUI parameter tree under various categories:

#### **Analysis Plot Settings**

*Note: Analysis plots are automatically generated when saving results. There are no specific GUI parameters to control individual plot generation.*

#### **GIF Overlay Settings**

- **Save ID-based GIF Overlay** (bool):
    - Enable/disable generation of animated GIF overlay with each cell ID colored uniquely.
    - Default: `True`

- **Save State-based GIF Overlay** (bool):
    - Enable/disable generation of animated GIF overlay with cells colored by state class.
    - Default: `True`

- **GIF Overlay Alpha** (float):
    - Transparency level for cell overlays in GIF animations (0.0 = transparent, 1.0 = opaque).
    - Controls how much the original image shows through the cell overlays.
    - Default: `0.6`

- **GIF Overlay FPS** (int):
    - Frames per second for GIF animations.
    - Higher values create faster animations, lower values create slower, more detailed viewing.
    - Default: `10`

#### **State Mask and Classification Settings**

- **Save 16-bit Class Masks** (bool):
    - Enable/disable generation of 16-bit PNG masks for different cell states/classes.
    - Creates masks where each pixel value corresponds to a state class ID.
    - Useful for downstream analysis and state-based processing.
    - Default: `True`

- **Class Mask Prefix** (str):
    - Prefix for class mask filenames (e.g., "class_mask" creates files like "class_mask_000.png").
    - Default: `"class_mask"`

- **Classification Max Centroid Distance** (float):
    - Maximum distance for state classification between centroids.
    - Used when assigning cell states based on class-specific masks.
    - Default: `30.0`

#### **Temporal Outline Stack Settings**

- **Temporal Outline Colormap** (str):
    - Colormap for the temporal outline stack visualization.
    - Controls how cell outlines are colored by time in the overlay image.
    - Options include: "plasma", "viridis", "inferno", "magma", "turbo", etc.
    - Default: `"plasma"`

- **Figure DPI** (int):
    - Resolution for all figure outputs (higher = sharper).
    - Default: `300`

- **Temporal Outline Scale Factor** (int):
    - Scaling factor for the temporal outline stack resolution.
    - Higher values create larger, sharper images.
    - Default: `2`

#### **Offset Track Plot Settings**

- **Offset Track Plot** (bool):
    - Enable/disable saving the offset track plot.
    - Shows all tracks offset so the centroid of all objects in the first frame is at (0,0).
    - Default: `True`

- **Offset Track Plot Left Color** (str):
    - Color for tracks ending left of origin in the offset plot.
    - Can be any valid color name or hex code.
    - Default: `"black"`

- **Offset Track Plot Right Color** (str):
    - Color for tracks ending right of origin in the offset plot.
    - Can be any valid color name or hex code.
    - Default: `"red"`

- **Offset Track Plot Output Filename** (str):
    - Output filename for the offset track plot.
    - Default: `"offset_track_plot.png"`


#### **CTC Evaluation Settings**

- **Run CTC Metrics Evaluation** (bool):
    - Enable/disable automatic CTC challenge evaluation when ground truth is available.
    - When enabled, automatically runs evaluation if `man_track.txt` is found.
    - Default: `True`

#### **Display Settings**

- **Show IDs** (bool):
    - Enable/disable display of cell ID labels on the main canvas.
    - Default: `True`

- **Show tracks** (bool):
    - Enable/disable display of cell tracks on the main canvas.
    - Default: `True`

- **Show Mitosis Labels** (bool):
    - Enable/disable mitosis event labels in hover tooltips and visualizations.
    - Default: `True`

- **Temporal Outline Colormap** (str):
    - Colormap for temporal outline stack visualization.
    - Options include: "plasma", "viridis", "inferno", "magma", "turbo", etc.
    - Default: `"plasma"`

- **Figure DPI** (int):
    - Resolution for all figure outputs (higher = sharper).
    - Default: `300`

- **Temporal Outline Scale Factor** (int):
    - Scale factor for temporal outline stack resolution (higher = larger, sharper image).
    - Default: `2`





#### **File Format Settings**

- **Mask Extension** (str):
    - File extension for mask images (e.g., "tif", "png", "jpg").
    - Should match your input mask file format.
    - Default: `"tif"`

- **Raw Image Extension** (str):
    - File extension for raw images (e.g., "tif", "png", "jpg").
    - Should match your input image file format.
    - Default: `"tif"`

- **Mask Folder Suffix** (str):
    - Suffix added to image folder name to find corresponding mask folder.
    - Example: If images are in "Dataset1" and suffix is "_mask", masks should be in "Dataset1_mask".
    - Default: `"_mask"`

---

### Other
- **tracking_operations_log.txt**: Log of operations.
- **lineage_plot_*.png**: Lineage tree visualizations.

---

## Requirements

* Python 3.7 or higher (Python 3.6 might work, but 3.7+ is recommended for better compatibility with recent library versions).
* The following Python libraries:
    * `PyQt5`
    * `pyqtgraph`
    * `numpy`
    * `pandas`
    * `trackpy`
    * `scikit-image`
    * `Pillow`
    * `imageio` (version 3.x or higher recommended, as `imageio.v3` is used in `cell_io.py`)
* **Gurobi** (optional, for ILP acceleration if available)

You can typically install these using pip. It's highly recommended to use a virtual environment.

---

## Installation

1.  **Clone or download the repository:**
    If your project is on a Git platform (like GitHub), clone it:
    ```bash
    git clone [https://github.com/yourusername/your-cell-tracking-tool.git](https://github.com/yourusername/your-cell-tracking-tool.git) 
    cd your-cell-tracking-tool
    ```
    (Replace `https://github.com/yourusername/your-cell-tracking-tool.git` with your actual repository URL if applicable. If you have the files locally, navigate to the project directory.)

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    You can install the required libraries individually using pip:
    ```bash
    pip install PyQt5 pyqtgraph numpy pandas trackpy scikit-image Pillow imageio
    ```
    Alternatively, if you create a `requirements.txt` file listing these packages (e.g., `PyQt5`, `pyqtgraph`, etc., one per line), you can install them all with:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage


### Basic Workflow

#### **Step 1: Launch the Application**
```bash
python cell_main.py
```
The GUI will open with several dockable panels: Image View, Cell Editor, Lineage Tree, I/O & Parameters, and Log Console.

#### **Step 2: Load Your Data**
1. **Open Image Folder:**
   - Click the "Open Data Folder" button
   - Select the directory containing your raw image sequence (e.g., TIF, PNG files)
   - The application automatically looks for a corresponding mask folder in the same parent directory
   - Mask folder naming: `{ImageFolderName}_{MaskFolderSuffix}` (default suffix: `_mask`)
   - Example: If images are in `Dataset1`, masks should be in `Dataset1_mask`
   
2. **Advanced File Format:**
   The tool supports two levels of file organization to suit different project needs:

   **Simple Structure (Default):**
   - All output files placed directly in the main output folder
   - Suitable for quick analysis and smaller projects
   - Most straightforward organization

   **Advanced Structure (Multiclass Classification):**
   - Enables multiclass cell state classification and analysis
   - Users can define custom cell classes/states (e.g., "cell", "mitosis", "apoptosis", "fused")
   - Software automatically detects objects and assigns states based on the defined classes
   - Creates additional outputs including:
     - State-based masks and visualizations
     - Class-specific analysis plots
     - State transition diagrams and statistics
     - Enhanced lineage trees colored by cell state
   - Generates 16-bit class masks where each pixel value corresponds to a state class ID
   - Includes class mapping files for state identification
   - Enable with "Enable Advanced File Structure" parameter

   **How to Use Multiclass Classification:**
   1. **Enable Advanced Structure:** Set "Enable Advanced File Structure" to `True`
   2. **Define Classes:** Configure your cell classes/states in the parameters
   3. **Provide Class Masks:** Ensure you have class-specific mask folders for each state
   4. **Run Tracking:** The software will automatically assign states based on class detection
   5. **Review Results:** Check state assignments in the Cell Table Editor and lineage plots
   6. **Manual Correction:** Edit cell states in the Cell Table Editor - changes are reflected in lineage plots

3. **Verify Data Loading:**
   - Check the Log Console for loading status and any warnings
   - Ensure the number of frames matches between images and masks
   - Verify file extensions match your parameter settings

#### **Step 3: Configure Parameters (Recommended)**
Review and adjust parameters in the "I/O & Parameters" dock:

- **File Settings:** Verify `Raw image extension`, `Mask extension`, `Mask folder suffix`
- **Tracking Algorithm:** Choose between ILP (default), TrackPy, or TrackAstra
- **Display Settings:** Configure `Show IDs`, `Show tracks`, `Show contours`, `Lineage View Type`
- **Output Settings:** Enable/disable analysis plots, GIF overlays, class masks, etc.
- **Measurement Settings:** Set `Pixel scale` and `Frames per hour` for accurate measurements

#### **Step 4: Run Cell Tracking**
1. **Execute Tracking:**
   - Click the "Run Cell Tracking" button
   - Monitor progress in the Log Console
   - Processing time depends on sequence length and tracking method

2. **Review Results:**
   - Cell tracks appear overlaid on the raw image view
   - Cell IDs and lineage information populate the Cell Editor table
   - Lineage Tree shows parent-child relationships

#### **Step 5: Interactive Analysis and Manual Correction**
1. **Navigate and Explore:**
   - Use time sliders to move through frames
   - Click on cells in the image view to select them
   - Use the Cell Editor to toggle track visibility and view cell properties

2. **Lineage Tree Operations:**
   - Switch between "Track Segments" and "Node View" for different visualizations
   - Select root tracks to focus on specific lineages
   - Use drag-and-drop operations to correct lineage relationships:
     - Merge tracks, set parent-child relationships
     - Add fusion events, split tracks at specific frames
   - Double-click nodes to break tracks into segments

3. **Manual Corrections:**
   - Edit parent-child relationships in the Cell Table Editor is also possible and the changes are reflected back in the lineage plot.
   - Use the "Merge Selected Track with Parent" button when applicable
   - Adjust cell states/classes for classification analysis

#### **Step 6: Save Results and Generate Outputs**
1. **Save Results:**
   - Click the "Save Results" button
   - All data is saved to a timestamped folder: `{InputFolder}_Exp_{YYYY-MM-DDTHHMMSS}_{tracking_mode}`
   - Includes tracking data, visualizations, parameters, and analysis plots

2. **Generated Outputs:**
   - **Core Data:** `tracks.csv`, `lineage_relationships.csv`, `experiment_parameters.csv`
   - **Visualizations:** Track overlays, masks, animations, analysis plots
   - **CTC Format:** `{InputFolder}_RES/` folder with `res_track.txt` and masks
   - **Analysis Plots:** Temporal, spatial, morphological, and lineage analysis (if enabled)
   - **CTC Evaluation:** Automatic evaluation results if ground truth is available

#### **Step 7: Batch Processing (Optional)**

**Overview:**
Batch tracking allows you to process multiple datasets automatically using the same parameter settings. This is particularly useful for processing large numbers of experiments or when you need consistent analysis across multiple datasets.

**Dataset Organization Requirements:**
- Create a parent folder containing all your individual dataset folders
- Each dataset folder should contain the raw image sequence
- Each dataset must have a corresponding mask folder in the same parent directory
- Mask folders should follow the naming convention: `{DatasetName}_{MaskFolderSuffix}` (default: `_mask`)

**Example Folder Structure:**
```
Parent_Folder/
├── Dataset1/
│   ├── frame_000.tif
│   ├── frame_001.tif
│   └── ...
├── Dataset1_mask/
│   ├── mask_000.tif
│   ├── mask_001.tif
│   └── ...
├── Dataset2/
│   ├── frame_000.tif
│   └── ...
├── Dataset2_mask/
│   ├── mask_000.tif
│   └── ...
└── ...
```

**Batch Processing Workflow:**
1. **Configure Parameters:** Set data folder path, all tracking parameters, output settings, and analysis options as desired
2. **Select Parent Folder:** Click "Batch Tracking" to run the parent folder containing all datasets
3. **Automatic Processing:** The tool will:
   - Scan the parent folder for valid dataset pairs (image + mask folders)
   - Process each dataset sequentially using the current parameter settings
   - Generate timestamped output folders for each dataset
   - Run CTC evaluation if ground truth is available for each dataset
4. **Monitor Progress:** Check the Log Console for processing status and any errors
5. **Review Results:** Each dataset gets its own output folder with complete analysis results

**Advantages of Batch Processing:**
- **Consistency:** All datasets processed with identical parameters
- **Efficiency:** Automated processing saves time for large datasets
- **Organization:** Results clearly separated by dataset
- **Scalability:** No limit on the number of datasets that can be processed
- **Error Handling:** Individual dataset failures don't stop the entire batch

**Best Practices:**
- Test parameters on a single dataset before running batch processing
- Ensure all datasets have consistent file formats and naming conventions
- Monitor system resources when processing large batches
- Check the Log Console for any processing errors or warnings
- Verify output folder names and contents after batch completion


---

## Lineage Tree Operations and Manual Editing

The Lineage Tree widget provides powerful tools for visualizing, editing, and correcting cell lineage relationships. Below are the available operations and their usage:

### User Operations (via GUI)

- **Drag-and-Drop Operations (between nodes/tracks):**
  - **Join Track (Merge Dragged into Target):** Merge two tracks into one, combining their histories.
  - **Set Target as Parent of Dragged:** Make the target node the parent of the dragged node (relink lineage).
  - **Set Dragged as Parent of Target:** Make the dragged node the parent of the target node.
  - **Set Dragged as Additional Parent of Target:** Add the dragged node as an additional parent (for fusion events).
  - **Insert Dragged as Child & Split Target's Tail:** Insert the dragged node as a child of the target, splitting the target’s track at the drop frame (for mitosis/fusion editing).
  - **Cancel:** Cancel the operation.

- **Double-Click on Node:**
  - **Break into Two Tracks:** Split a track at the selected frame, creating two separate tracks (useful for correcting over-merged tracks).

- **Root Selection and Filtering:**
  - Select one or more root tracks to focus the tree view.
  - Filter root tracks by ID.

- **View Type Switching:**
  - Switch between “Track Segments View” (each track as a line) and “Node View” (each detection as a node).
  - (Optionally) “Class Type” view if available.

- **Highlighting and Tooltips:**
  - Hover over nodes to see detailed information (ID, frame, state, etc.).
  - Highlight tracks or nodes by clicking or hovering.

- **Export:**
  - Export the current lineage tree plot as an image.

### Dialog Operations

- **LineageOperationDialog:**
  - Pops up on drag-and-drop, lets user choose the operation to perform.
- **SplitFuseOperationDialog:**
  - Pops up on double-click, lets user break a track.

### Typical Use Cases

- **Correcting lineage errors:** Merge, split, or relink tracks to fix automatic tracking mistakes.
- **Editing fusions/mitoses:** Add or remove parent/child relationships.
- **Visualizing and exporting:** Focus on specific lineages, export images for publication.

## Manual Lineage Editing: Breaking Tracks and Updating Cell State

You can manually edit cell lineages and states to correct tracking or classification errors:

1. **Break a Track (Split a Lineage):**
   - In the Lineage Tree or Cell Editor, select a track and choose to break it at a specific frame.
   - The original track will be split into two segments: the original (up to the break frame) and a new segment (from the break frame onward, with a new cell ID).
   - The new segment will appear as a new root or as a daughter, depending on the context.

2. **Update Cell State for Class Classification:**
   - After breaking a track, you can update the state (class) of the new segment.
   - In the Cell Editor table, locate the new cell ID and edit its state (e.g., change from "cell" to "mitosis" or another class).
   - You can also re-run the cell state classification to automatically assign states based on class-specific masks, or manually override as needed.

3. **Save Results:**
   - When you save results, all manual edits to lineage and cell state are included in the output files (e.g., `res_track.txt`, lineage tables, and state tables if enabled).

This workflow allows you to correct lineage errors, split or merge tracks, and ensure that cell state/classification is accurate for downstream analysis.

---

## Troubleshooting

### Common Issues

1.  **No valid datasets found during batch processing**:
    * Ensure each dataset folder (e.g., `DatasetA`) has a corresponding mask folder (e.g., `DatasetA_mask`) in the *same parent directory* selected for batch processing.
    * Verify that the "Mask folder suffix" parameter matches the naming convention of your mask folders.
    * Check that image files within these folders have the correct file extensions as specified in the parameters.

2.  **Cell tracking fails or produces poor results**:
    * Verify that mask images are of good quality and accurately segment the cells.
    * Ensure the raw image sequence and mask sequence have the same number of frames and correspond to each other.
    * Adjust "Trackpy search range" and "Trackpy memory" parameters. Too small a search range might break tracks; too large might incorrectly link distant cells.
    * Review mitosis detection parameters if lineage is incorrect.

3.  **Application crashes or shows errors**:
    * Check the "Log Console" within the application for error messages.
    * If started from a terminal, check the terminal output for more detailed tracebacks.
    * Ensure all dependencies listed in the "Requirements" section are correctly installed and accessible in your Python environment. Using a virtual environment is highly recommended to avoid conflicts.
    * Confirm that your image and mask files are not corrupted and are in a supported format.

4.  **Node View in Lineage Tree is not displaying as expected**:
    * Ensure the "Lineage View Type" parameter is set to "Node View".

---

## Known Issues and Solutions

Here are some common issues users may encounter, along with recommended solutions:

- **CTC Evaluation Fails with TrackPy and Nonzero Memory:**
  - **Problem:** If you use the TrackPy tracking method and set the `memory` parameter to a value greater than 0, the CTC evaluation step may fail or produce invalid results.
  - **Solution:** Set the `Trackpy memory` parameter to `0` before running tracking if you plan to use CTC evaluation.

- **Error: Subnet Exceeds 31 (or Similar):**
  - **Problem:** If you see an error message about subnet size exceeding 31 (or similar), this usually means the `Trackpy search range` parameter is set too high for your data.
  - **Solution:** Decrease the `Trackpy search range` parameter in the GUI or parameter file and rerun tracking.

- **Out of Memory or Crashes on Large Datasets:**
  - **Problem:** Processing very large image sequences or datasets may exhaust system memory.
  - **Solution:** Try processing smaller batches, increase system RAM, or use a machine with more memory.

- **Gurobi Not Found (for ILP):**
  - **Problem:** If you select ILP tracking and Gurobi is not installed, tracking may fall back to a slower solver or fail.
  - **Solution:** Install Gurobi and ensure it is licensed and available in your Python environment, or use default Scipy solver, or TrackPy as an alternative.

If you encounter other issues, please check the log console for error messages or consult the Troubleshooting section above.

---

## License

This software is provided under the MIT License.

Copyright (c) 2025 Paul Hsieh-Fu Tsai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgements

* Original framework by ImagineA / Andrei Rares.
* Uses the TrackPy library for particle tracking algorithms.
* Uses the PyQtGraph library for interactive plotting and visualization.
* Portions of the lineage tree logic were adapted from concepts in napari-arboretum.

## Authors

- Paul Hsieh-Fu Tsai
- I-Ming Chang

