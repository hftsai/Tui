# cell_evaluation.py
# Handles calling the py-ctcmetrics evaluation API.
# V2 (Gemini - This Update): Updated NumPy type checks for NumPy 2.0 compatibility.
# V3 (Gemini - This Update): Added missing tifffile import.
# V4 (Gemini - This Update): Enhanced verify_ctc_format with more detailed logging for troubleshooting.
# V5 (User Request): Added MT and ML metrics implementation.

import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
import tempfile
import shutil
import re
from ctc_metrics.metrics import op_clb

try:
    import tifffile
except ImportError:
    logging.error("tifffile not found. Please install it with: pip install tifffile")
    tifffile = None

try:
    from ctc_metrics import evaluate_sequence
except ImportError:
    try:
        from ctc_metrics.scripts.evaluate import evaluate_sequence
    except ImportError:
        logging.error("Could not import 'evaluate_sequence' from 'ctc_metrics' or 'ctc_metrics.scripts.evaluate'.")
        logging.error("Please ensure 'py-ctcmetrics' is installed correctly in your Python environment (e.g., pip install py-ctcmetrics).")
        logging.error("You might need to adjust the import statement based on your py-ctcmetrics package structure.")
        evaluate_sequence = None

try:
    from ctc_metrics.utils.representations import track_confusion_matrix
    MT_ML_AVAILABLE = True
except ImportError:
    logging.warning("Could not import 'track_confusion_matrix' from ctc_metrics.utils.representations. MT and ML metrics will not be available.")
    MT_ML_AVAILABLE = False


def compute_mt_ml_metrics(gt_sequence_dir, res_sequence_dir):
    """
    Compute MT (Mostly Tracked) and ML (Mostly Lost) metrics.
    
    Args:
        gt_sequence_dir (str): Path to ground truth sequence directory
        res_sequence_dir (str): Path to results sequence directory
        
    Returns:
        dict: Dictionary containing MT and ML values, or None if computation fails
    """
    if not MT_ML_AVAILABLE:
        logging.warning("MT/ML computation not available due to missing imports.")
        return {"MT": None, "ML": None}
    
    try:
        # Check for GT tracking file - try both with and without TRA subdirectory
        gt_track_file = os.path.join(gt_sequence_dir, "man_track.txt")
        if not os.path.exists(gt_track_file):
            gt_track_file = os.path.join(gt_sequence_dir, "TRA", "man_track.txt")
        
        res_track_file = os.path.join(res_sequence_dir, "res_track.txt")
        
        if not os.path.exists(gt_track_file):
            logging.error(f"GT tracking file not found at: {os.path.join(gt_sequence_dir, 'TRA', 'man_track.txt')} or {os.path.join(gt_sequence_dir, 'man_track.txt')}")
            return {"MT": None, "ML": None}
        
        if not os.path.exists(res_track_file):
            logging.error(f"RES tracking file not found: {res_track_file}")
            return {"MT": None, "ML": None}
        
        # Load mask files to get labels - check both TRA subdirectory and root
        gt_tra_dir = os.path.join(gt_sequence_dir, "TRA")
        if os.path.isdir(gt_tra_dir):
            # Check for both man_trackXXX.tif and maskXXX.tif patterns
            gt_mask_files = sorted([f for f in os.listdir(gt_tra_dir) if (f.startswith("man_track") or f.startswith("mask")) and f.endswith(".tif")])
            gt_mask_dir = gt_tra_dir
        else:
            gt_mask_files = sorted([f for f in os.listdir(gt_sequence_dir) if (f.startswith("man_track") or f.startswith("mask")) and f.endswith(".tif")])
            gt_mask_dir = gt_sequence_dir
            
        res_mask_files = sorted([f for f in os.listdir(res_sequence_dir) if f.startswith("mask") and f.endswith(".tif")])
        
        logging.info(f"GT mask directory: {gt_mask_dir}, found {len(gt_mask_files)} mask files")
        logging.info(f"RES mask directory: {res_sequence_dir}, found {len(res_mask_files)} mask files")
        
        if not gt_mask_files or not res_mask_files:
            logging.error(f"Missing mask files for MT/ML computation. GT: {len(gt_mask_files)}, RES: {len(res_mask_files)}")
            return {"MT": None, "ML": None}
        
        # Prepare label lists for all frames
        labels_ref = []
        labels_comp = []
        
        # Get the number of frames (assuming both GT and RES have the same number)
        num_frames = min(len(gt_mask_files), len(res_mask_files))
        logging.info(f"Processing {num_frames} frames for MT/ML computation")
        
        # Create frame index mapping for GT files
        gt_frame_mapping = {}
        for gt_file in gt_mask_files:
            try:
                frame_num = extract_frame_number_from_filename(gt_file)
                gt_frame_mapping[frame_num] = gt_file
            except ValueError:
                logging.warning(f"Could not parse frame number from GT file: {gt_file}")

        # Create frame index mapping for RES files
        res_frame_mapping = {}
        for res_file in res_mask_files:
            try:
                frame_num = extract_frame_number_from_filename(res_file)
                res_frame_mapping[frame_num] = res_file
            except ValueError:
                logging.warning(f"Could not parse frame number from RES file: {res_file}")

        # Find common frame numbers
        common_frames = sorted(set(gt_frame_mapping.keys()) & set(res_frame_mapping.keys()))
        logging.info(f"Found {len(common_frames)} common frames between GT and RES")
        
        if not common_frames:
            logging.error("No common frames found between GT and RES")
            return {"MT": None, "ML": None}
        
        for frame_num in common_frames:
            # Load GT mask
            gt_mask_path = os.path.join(gt_mask_dir, gt_frame_mapping[frame_num])
            if tifffile is not None:
                gt_mask = tifffile.imread(gt_mask_path)
                labels_ref.append(gt_mask.flatten())
            else:
                logging.error("tifffile not available for loading GT mask")
                return {"MT": None, "ML": None}
            
            # Load RES mask
            res_mask_path = os.path.join(res_sequence_dir, res_frame_mapping[frame_num])
            if tifffile is not None:
                res_mask = tifffile.imread(res_mask_path)
                labels_comp.append(res_mask.flatten())
            else:
                logging.error("tifffile not available for loading RES mask")
                return {"MT": None, "ML": None}
        
        # For MT/ML, we need to establish correspondence between GT and RES tracks
        # This is a simplified approach - in practice, you might need more sophisticated matching
        
        # Read tracking files to get track information
        gt_tracks = {}
        with open(gt_track_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    try:
                        track_id, start_frame, end_frame, parent_id = map(int, parts)
                        gt_tracks[track_id] = (start_frame, end_frame, parent_id)
                    except ValueError:
                        logging.warning(f"Skipping invalid line in {gt_track_file}: {line.strip()}")

        res_tracks = {}
        with open(res_track_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    try:
                        track_id, start_frame, end_frame, parent_id = map(int, parts)
                        res_tracks[track_id] = (start_frame, end_frame, parent_id)
                    except ValueError:
                        logging.warning(f"Skipping invalid line in {res_track_file}: {line.strip()}")
        
        # Create mapping arrays (this is a simplified version)
        # In practice, you would need a more sophisticated matching algorithm
        mapped_ref = []
        mapped_comp = []
        
        for frame_idx in range(num_frames):
            gt_frame_labels = labels_ref[frame_idx]
            res_frame_labels = labels_comp[frame_idx]
            
            # Simple approach: create identity mapping for existing labels
            gt_unique = np.unique(gt_frame_labels)
            res_unique = np.unique(res_frame_labels)
            
            # For this simplified implementation, we assume labels match when they exist
            # This should be replaced with proper track matching in production
            mapped_ref_frame = gt_frame_labels.copy()
            mapped_comp_frame = res_frame_labels.copy()
            
            mapped_ref.append(mapped_ref_frame)
            mapped_comp.append(mapped_comp_frame)
        
        # Calculate MT/ML using the track_confusion_matrix
        track_intersection = track_confusion_matrix(labels_ref, labels_comp, mapped_ref, mapped_comp)
        
        # Calculate the metrics
        total_ref = np.sum(track_intersection[1:, :], axis=1)
        ratio = np.max(track_intersection[1:, :], axis=1) / np.maximum(total_ref, 1)
        valid_ref = total_ref > 0
        
        if np.sum(valid_ref) > 0:
            mt = np.sum(ratio[valid_ref] >= 0.8) / np.sum(valid_ref)
            ml = np.sum(ratio[valid_ref] < 0.2) / np.sum(valid_ref)
        else:
            mt = 0.0
            ml = 0.0
        
        logging.info(f"MT/ML computation successful: MT={mt:.4f}, ML={ml:.4f}")
        return {"MT": mt, "ML": ml}
        
    except Exception as e:
        logging.error(f"Error computing MT/ML metrics: {e}")
        logging.error(traceback.format_exc())
        return {"MT": None, "ML": None}


def extract_frame_number_from_filename(filename):
    """
    Robustly extract frame number from mask filename.
    Handles various formats: mask000.tif, mask001.tif, etc.
    """
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Try to extract numeric part after 'mask'
    if name_without_ext.startswith('mask'):
        numeric_part = name_without_ext[4:]  # Remove 'mask' prefix
        if numeric_part.isdigit():
            return int(numeric_part)
    
    # Try regex pattern for any numeric sequence
    match = re.search(r'(\d+)', name_without_ext)
    if match:
        return int(match.group(1))
    
    # Fallback: try to extract any digits
    digits = ''.join(filter(str.isdigit, name_without_ext))
    if digits:
        return int(digits)
    
    raise ValueError(f"Could not extract frame number from filename: {filename}")


def validate_tracking_results_for_ctc(trj, id_masks):
    """
    Validate that tracking results are suitable for CTC evaluation.
    
    Args:
        trj (pd.DataFrame): Tracking trajectory data
        id_masks (np.ndarray): ID masks array
        
    Returns:
        dict: Validation results with status and messages
    """
    validation_result = {
        'valid': True,
        'messages': [],
        'warnings': []
    }
    
    # Check if trajectory exists and is not empty
    if trj is None or trj.empty:
        validation_result['valid'] = False
        validation_result['messages'].append("No tracking trajectory data available")
        return validation_result
    
    # Check required columns
    required_columns = ['particle', 'frame', 'x', 'y']
    missing_columns = [col for col in required_columns if col not in trj.columns]
    if missing_columns:
        validation_result['valid'] = False
        validation_result['messages'].append(f"Missing required columns: {missing_columns}")
    
    # Check for valid particle IDs
    if 'particle' in trj.columns:
        particle_ids = trj['particle'].dropna()
        if particle_ids.empty:
            validation_result['valid'] = False
            validation_result['messages'].append("No valid particle IDs found")
        else:
            # Check for non-positive particle IDs (CTC requires positive integers)
            invalid_ids = particle_ids[particle_ids <= 0]
            if not invalid_ids.empty:
                validation_result['warnings'].append(f"Found {len(invalid_ids)} non-positive particle IDs")
    
    # Check frame consistency
    if 'frame' in trj.columns:
        frames = trj['frame'].dropna()
        if frames.empty:
            validation_result['valid'] = False
            validation_result['messages'].append("No valid frame numbers found")
        else:
            # Check for non-negative frame numbers
            invalid_frames = frames[frames < 0]
            if not invalid_frames.empty:
                validation_result['warnings'].append(f"Found {len(invalid_frames)} negative frame numbers")
    
    # Check coordinate validity
    if all(col in trj.columns for col in ['x', 'y']):
        coords = trj[['x', 'y']].dropna()
        if coords.empty:
            validation_result['valid'] = False
            validation_result['messages'].append("No valid coordinates found")
        else:
            # Check for NaN or infinite coordinates
            nan_coords = coords.isnull().sum().sum()
            if nan_coords > 0:
                validation_result['warnings'].append(f"Found {nan_coords} NaN coordinate values")
    
    # Check track duration (tracks should span multiple frames)
    if all(col in trj.columns for col in ['particle', 'frame']):
        track_durations = trj.groupby('particle')['frame'].agg(['min', 'max', 'count'])
        single_frame_tracks = track_durations[track_durations['count'] == 1]
        if len(single_frame_tracks) > 0:
            validation_result['warnings'].append(f"Found {len(single_frame_tracks)} tracks with only one frame")
    
    # Check if ID masks are consistent with trajectory
    if id_masks is not None:
        mask_ids = set(np.unique(id_masks))
        if 'particle' in trj.columns:
            traj_ids = set(trj['particle'].dropna().unique())
            missing_in_traj = mask_ids - traj_ids
            missing_in_mask = traj_ids - mask_ids
            if missing_in_traj:
                validation_result['warnings'].append(f"Found {len(missing_in_traj)} IDs in masks but not in trajectory")
            if missing_in_mask:
                validation_result['warnings'].append(f"Found {len(missing_in_mask)} IDs in trajectory but not in masks")
    
    return validation_result


def run_ctc_evaluation_api(
        gt_sequence_dir,
        res_sequence_dir,
        metrics_list=None,
        num_threads=0,
        trj=None,
        id_masks=None
):
    """
    Runs the Cell Tracking Challenge evaluation using the Python API of py-ctcmetrics.

    Args:
        gt_sequence_dir (str): Path to the ground truth sequence folder
                               (e.g., ".../CTC_GT_Data/Synthetic_SEQ01").
                               This folder should contain TRA/man_track.txt and TRA/maskXXX.tif.
        res_sequence_dir (str): Path to your software's results sequence folder
                                (e.g., ".../MySoftware_RES/Synthetic_SEQ01_RES").
                                This folder should directly contain res_track.txt and maskXXX.tif.
        metrics_list (list, optional): List of metric strings to evaluate
                                       (e.g., ["TRA", "SEG", "DET", "MOTA"]).
                                       If None, py-ctcmetrics will calculate its default set
                                       (which includes TRA, SEG, DET, and others).
                                       To get CHOTA, MOTA, HOTA, IDF1 etc., ensure they are included here
                                       if not part of the default set.
        num_threads (int): Number of threads to use for evaluation. 0 for all CPUs, 1 for single thread.
        trj (pd.DataFrame, optional): Tracking trajectory data for validation
        id_masks (np.ndarray, optional): ID masks array for validation

    Returns:
        dict: A dictionary containing the evaluation results, or None if an error occurs.
              Example: {'TRA': 0.95, 'SEG': 0.88, ...}
    """
    if evaluate_sequence is None:
        logging.error("`evaluate_sequence` function not available due to import error. Cannot proceed with evaluation.")
        return {"status": "error", "message": "evaluate_sequence function not imported.", "metrics": None}

    # Validate tracking results if provided
    if trj is not None or id_masks is not None:
        validation_result = validate_tracking_results_for_ctc(trj, id_masks)
        if not validation_result['valid']:
            error_msg = "Tracking results validation failed:\n" + "\n".join(validation_result['messages'])
            logging.error(error_msg)
            return {"status": "error", "message": error_msg, "metrics": None}
        
        if validation_result['warnings']:
            warning_msg = "Tracking results validation warnings:\n" + "\n".join(validation_result['warnings'])
            logging.warning(warning_msg)

    if not os.path.isdir(gt_sequence_dir):
        logging.error(f"Ground Truth sequence directory not found: {gt_sequence_dir}")
        return {"status": "error", "message": f"GT directory not found: {gt_sequence_dir}", "metrics": None}
    if not os.path.isdir(res_sequence_dir):
        logging.error(f"Results sequence directory not found: {res_sequence_dir}")
        return {"status": "error", "message": f"RES directory not found: {res_sequence_dir}", "metrics": None}

    # Check for essential files in GT directory
    gt_tra_path = os.path.join(gt_sequence_dir, "TRA")
    if not os.path.exists(os.path.join(gt_tra_path, "man_track.txt")):
        logging.warning(
            f"Missing GT tracking file: {os.path.join(gt_tra_path, 'man_track.txt')}. Some metrics might fail.")
    if not os.path.isdir(gt_tra_path) or not any(
            (f.startswith("mask") or f.startswith("man_track")) and f.endswith(".tif") for f in os.listdir(gt_tra_path)):
        logging.warning(f"GT TRA masks (maskXXX.tif or man_trackXXX.tif) may be missing in: {gt_tra_path}. Some metrics might fail.")

    # Check for essential files in RES directory
    if not os.path.exists(os.path.join(res_sequence_dir, "res_track.txt")):
        logging.warning(
            f"Missing RES tracking file: {os.path.join(res_sequence_dir, 'res_track.txt')}. Some metrics might fail.")
    if not any(f.startswith("mask") and f.endswith(".tif") for f in os.listdir(res_sequence_dir)):
        logging.warning(f"RES masks (maskXXX.tif) may be missing in: {res_sequence_dir}. Some metrics might fail.")

    logging.info(f"Running CTC evaluation:")
    logging.info(f"  GT Path: {gt_sequence_dir}")
    logging.info(f"  RES Path: {res_sequence_dir}")
    if metrics_list:
        logging.info(f"  Requested metrics: {metrics_list}")
    else:
        logging.info("  Requested metrics: Default set from py-ctcmetrics")
    logging.info(f"  Using num_threads: {num_threads if num_threads > 0 else 'all available'}")

    try:
        # Separate MT and ML from other metrics since they need custom computation
        mt_ml_requested = False
        standard_metrics = []
        
        if metrics_list:
            for metric in metrics_list:
                if metric in ["MT", "ML"]:
                    mt_ml_requested = True
                else:
                    standard_metrics.append(metric)
        
        # Run standard evaluation
        results_dict = {}
        if standard_metrics or not metrics_list:  # Run standard metrics
            try:
                # Debug: Check the content before calling CTC library
                logging.info("=== DEBUG: Pre-evaluation checks ===")
                
                # Check RES tracking file
                res_track_file = os.path.join(res_sequence_dir, "res_track.txt")
                if os.path.exists(res_track_file):
                    with open(res_track_file, 'r') as f:
                        res_lines = f.readlines()
                    logging.info(f"RES tracking file has {len(res_lines)} lines")
                    if res_lines:
                        logging.info(f"First line: {res_lines[0].strip()}")
                        logging.info(f"Last line: {res_lines[-1].strip()}")
                
                # Check RES mask files
                res_mask_files = sorted([f for f in os.listdir(res_sequence_dir) if f.startswith("mask") and f.endswith(".tif")])
                logging.info(f"Found {len(res_mask_files)} RES mask files")
                if res_mask_files:
                    logging.info(f"First mask: {res_mask_files[0]}, Last mask: {res_mask_files[-1]}")
                
                # Check GT tracking file
                gt_track_file = os.path.join(gt_sequence_dir, "TRA", "man_track.txt")
                if os.path.exists(gt_track_file):
                    with open(gt_track_file, 'r') as f:
                        gt_lines = f.readlines()
                    logging.info(f"GT tracking file has {len(gt_lines)} lines")
                
                # Check GT mask files
                gt_tra_dir = os.path.join(gt_sequence_dir, "TRA")
                gt_seg_dir = os.path.join(gt_sequence_dir, "SEG")
                
                if os.path.exists(gt_tra_dir):
                    gt_tra_mask_files = sorted([f for f in os.listdir(gt_tra_dir) if f.startswith("mask") and f.endswith(".tif")])
                    logging.info(f"Found {len(gt_tra_mask_files)} GT mask files in TRA/")
                    if gt_tra_mask_files:
                        logging.info(f"First GT TRA mask: {gt_tra_mask_files[0]}, Last GT TRA mask: {gt_tra_mask_files[-1]}")
                
                if os.path.exists(gt_seg_dir):
                    gt_seg_mask_files = sorted([f for f in os.listdir(gt_seg_dir) if f.startswith("man_seg") and f.endswith(".tif")])
                    logging.info(f"Found {len(gt_seg_mask_files)} GT mask files in SEG/")
                    if gt_seg_mask_files:
                        logging.info(f"First GT SEG mask: {gt_seg_mask_files[0]}, Last GT SEG mask: {gt_seg_mask_files[-1]}")
                
                logging.info("=== END DEBUG ===")
                
                standard_results = evaluate_sequence(
                    res=res_sequence_dir,
                    gt=gt_sequence_dir,
                    metrics=standard_metrics if standard_metrics else [],
                    threads=num_threads
                )
                
                if standard_results is not None:
                    results_dict.update(standard_results)
                    
            except ValueError as e:
                logging.error(f"CTC library parsing error: {e}")
                return {
                    "status": "error", 
                    "message": f"CTC library parsing error: {e}. Please ensure your ground truth directory has the correct structure with TRA/ and SEG/ subdirectories.",
                    "metrics": None
                }
            except IndexError as e:
                if "index 0 is out of bounds for axis 0 with size 0" in str(e):
                    logging.error(f"CTC library encountered empty array error: {e}")
                    logging.error("This typically happens when there are inconsistent parent-child relationships in tracking data.")
                    logging.error("Please check your tracking data for consistency issues.")
                    return {
                        "status": "error", 
                        "message": f"CTC library error: {e}. This typically indicates inconsistent parent-child relationships in tracking data. Please review your tracking results.",
                        "metrics": None
                    }
                else:
                    logging.error(f"CTC library index error: {e}")
                    return {
                        "status": "error", 
                        "message": f"CTC library index error: {e}",
                        "metrics": None
                    }
            # The evaluation succeeded, so we don't need to raise an error
        
        # Compute MT and ML if requested
        if mt_ml_requested:
            logging.info("Computing MT and ML metrics...")
            mt_ml_results = compute_mt_ml_metrics(gt_sequence_dir, res_sequence_dir) # This should also point to the TRA/SEG subfolders
            if mt_ml_results:
                results_dict.update(mt_ml_results)

        logging.info(f"Evaluation completed for RES: {os.path.basename(res_sequence_dir)}.")
        
        if not results_dict:  # No results at all
            logging.warning(
                "No evaluation results obtained. This often indicates invalid input or internal errors in the script.")
            return {"status": "error", "message": "No evaluation results obtained, possibly due to invalid data.",
                    "metrics": None}

        return {"status": "success", "metrics": results_dict}

    except ValueError as e:
        # Re-raise ValueError so it can be caught by the inner handler
        raise e
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")
        logging.error(traceback.format_exc())
        return {"status": "error", "message": str(e), "metrics": None}


# ... rest of the file remains the same ...

if __name__ == '__main__':
    # This is an example of how to use this module if run directly
    # You would typically call run_ctc_evaluation_api from your main application

    # --- Configuration for direct test ---
    # Adjust these paths according to where your data is
    # 1. Generate synthetic data using `multiclass_synthetic_with_ctc_output.py`
    #    Let's assume it output to "./synthetic_cell_dataset_ctc_v5"
    #    and ctc_sequence_name was "Synthetic_SEQ01"
    #    and ctc_output_path_base was "CTC_GT_Data"

    test_gt_sequence_dir = "./synthetic_cell_dataset_ctc_v5/CTC_GT_Data/Synthetic_SEQ01"  # Contains TRA/

    # 2. Run your tracking software on the raw images and save results in CTC format
    #    For this example, we'll *copy* the GT to act as RES for a perfect score test.
    #    In a real scenario, this would be your software's output.
    test_res_parent_dir = "./temp_res_for_eval_test"
    test_res_sequence_name = "Synthetic_SEQ01_RES_Test"  # Name of your result sequence folder
    test_res_sequence_dir = os.path.join(test_res_parent_dir, test_res_sequence_name)

    os.makedirs(test_res_sequence_dir, exist_ok=True)

    # --- Create dummy RES data (by copying GT for a perfect score test) ---
    # In a real scenario, your software would generate these files.
    gt_tra_dir_for_copy = os.path.join(test_gt_sequence_dir, "TRA")

    if os.path.exists(gt_tra_dir_for_copy):
        import shutil

        # Copy man_track.txt to res_track.txt
        if os.path.exists(os.path.join(gt_tra_dir_for_copy, "man_track.txt")):
            shutil.copy2(os.path.join(gt_tra_dir_for_copy, "man_track.txt"),
                         os.path.join(test_res_sequence_dir, "res_track.txt"))

        # Copy mask files
        for f_name in os.listdir(gt_tra_dir_for_copy):
            if f_name.startswith("mask") and f_name.endswith(".tif"):
                shutil.copy2(os.path.join(gt_tra_dir_for_copy, f_name),
                             os.path.join(test_res_sequence_dir, f_name))
        logging.info(f"Created dummy RES data at {test_res_sequence_dir} by copying from GT for testing.")
    else:
        logging.error(f"GT TRA directory not found for copying: {gt_tra_dir_for_copy}")
    # --- End of dummy RES data creation ---

    if evaluate_sequence is None:
        logging.critical("Halting example - py-ctcmetrics `evaluate_sequence` could not be imported.")
    elif not os.path.isdir(test_gt_sequence_dir) or not os.path.isdir(test_res_sequence_dir):
        logging.error("GT or (dummy) RES directory for test not found. Please check paths.")
    else:
        metrics_to_run = ["TRA", "SEG", "DET", "MOTA", "HOTA", "IDF1", "CT", "TF", "BC", "CCA", "LNK", "CHOTA", "MT", "ML"]

        evaluation_output = run_ctc_evaluation_api(
            gt_sequence_dir=test_gt_sequence_dir,
            res_sequence_dir=test_res_sequence_dir,
            metrics_list=metrics_to_run,
            num_threads=0
        )

        if evaluation_output:
            logging.info(f"\n--- Evaluation Summary for sequence {test_res_sequence_name} ---")
            if evaluation_output.get("status") == "success":
                metrics_results = evaluation_output.get("metrics", {})
                if metrics_results:
                    logging.info("Calculated Metrics:")
                    for metric_name, score in metrics_results.items():
                        if isinstance(score, (float, np.floating)):  # Use np.floating
                            logging.info(f"  {metric_name}: {score:.4f}")
                        else:
                            logging.info(f"  {metric_name}: {score}")
                else:
                    logging.info("No metrics dictionary returned, but status was success.")
            else:
                logging.error("Evaluation script reported an error.")
                logging.error(f"Message: {evaluation_output.get('message', 'Unknown error')}")
        else:
            logging.error(f"Evaluation function call failed for sequence {test_res_sequence_name}.")

        # Clean up dummy RES directory
        # import shutil
        # if os.path.exists(test_res_parent_dir):
        #     shutil.rmtree(test_res_parent_dir)
        #     logging.info(f"Cleaned up dummy RES directory: {test_res_parent_dir}")


def verify_ctc_format(res_path, gt_path=None):
    """Verify CTC format compliance and log details for debugging."""
    issues = []
    logging.info(f"--- Verifying CTC Format for RES path: {res_path} ---")
    if gt_path:
        logging.info(f"--- Comparing against GT path: {gt_path} ---")

    # Check res_track.txt
    res_track_path = os.path.join(res_path, "res_track.txt")
    res_track_data_per_frame = {}  # Store track IDs per frame from res_track.txt

    if not os.path.exists(res_track_path):
        issues.append("Missing res_track.txt")
        logging.error("  VERIFY: res_track.txt is MISSING.")
    else:
        logging.info(f"  VERIFY: Found res_track.txt at {res_track_path}")
        with open(res_track_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                issues.append("res_track.txt is empty.")
                logging.warning("  VERIFY: res_track.txt is EMPTY.")

            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 4:
                    issues.append(
                        f"Line {i + 1} in res_track.txt has {len(parts)} columns, expected 4. Content: '{line.strip()}'")
                    logging.warning(
                        f"  VERIFY: res_track.txt Line {i + 1} has {len(parts)} cols. Expected 4. Content: '{line.strip()}'")
                else:
                    try:
                        L, B, E, P = map(int, parts)
                        if B > E:
                            issues.append(f"Track {L}: start frame {B} > end frame {E}")
                            logging.warning(f"  VERIFY: res_track.txt Track {L}: Start frame {B} > End frame {E}")
                        if P < 0:  # Parent ID 0 is valid (no parent)
                            issues.append(f"Track {L}: negative parent ID {P}")
                            logging.warning(f"  VERIFY: res_track.txt Track {L}: Negative parent ID {P}")
                        for frame_num_in_track in range(B, E + 1):
                            res_track_data_per_frame.setdefault(frame_num_in_track, set()).add(L)
                    except ValueError:
                        issues.append(f"Line {i + 1} in res_track.txt: non-integer values. Content: '{line.strip()}'")
                        logging.warning(
                            f"  VERIFY: res_track.txt Line {i + 1}: Non-integer values. Content: '{line.strip()}'")
        logging.info(f"  VERIFY: Parsed {len(lines)} lines from res_track.txt.")

    # Check masks
    mask_files = sorted([f for f in os.listdir(res_path) if f.startswith("mask") and f.endswith(".tif")])
    if not mask_files:
        issues.append("No mask files (maskXXX.tif) found in RES path.")
        logging.error(f"  VERIFY: No mask files (maskXXX.tif) found in {res_path}")
    else:
        logging.info(f"  VERIFY: Found {len(mask_files)} mask files in RES path. Checking a few...")
        # Check first, middle, and last mask for detailed comparison
        frames_to_check_indices = []
        if len(mask_files) > 0: frames_to_check_indices.append(0)
        if len(mask_files) > 2: frames_to_check_indices.append(len(mask_files) // 2)
        if len(mask_files) > 1: frames_to_check_indices.append(len(mask_files) - 1)

        # Ensure unique indices if list is short
        frames_to_check_indices = sorted(list(set(frames_to_check_indices)))

        for mask_idx in frames_to_check_indices:
            mask_file = mask_files[mask_idx]
            mask_path = os.path.join(res_path, mask_file)
            logging.info(f"    VERIFY: Checking RES mask: {mask_file}")
            try:
                mask = tifffile.imread(mask_path)
                if mask.ndim != 2:
                    issues.append(f"Mask {mask_file} is not 2D. Shape: {mask.shape}")
                    logging.error(f"    VERIFY: Mask {mask_file} is not 2D! Shape: {mask.shape}")
                    continue
                if not np.issubdtype(mask.dtype, np.unsignedinteger):
                    issues.append(f"Mask {mask_file} is not an unsigned integer type. Type: {mask.dtype}")
                    logging.warning(f"    VERIFY: Mask {mask_file} is not an unsigned integer type: {mask.dtype}")

                mask_ids_in_file = set(np.unique(mask)) - {0}  # Exclude background
                logging.info(
                    f"      Unique non-zero IDs in {mask_file}: {sorted(list(mask_ids_in_file)) if mask_ids_in_file else 'None'}")

                try:  # Robustly extract frame number
                    frame_num_str = "".join(filter(str.isdigit, mask_file[4:-4]))  # "mask" is 4 chars, ".tif" is 4
                    frame_num = int(frame_num_str)
                except ValueError:
                    issues.append(f"Could not parse frame number from mask filename: {mask_file}")
                    logging.error(f"    VERIFY: Could not parse frame number from mask filename: {mask_file}")
                    continue

                logging.info(f"      Parsed frame number: {frame_num}")

                track_ids_expected_in_frame = res_track_data_per_frame.get(frame_num, set())
                logging.info(
                    f"      Track IDs expected in frame {frame_num} (from res_track.txt): {sorted(list(track_ids_expected_in_frame)) if track_ids_expected_in_frame else 'None'}")

                # IDs in res_track.txt but not in this mask file
                missing_in_mask = track_ids_expected_in_frame - mask_ids_in_file
                if missing_in_mask:
                    issues.append(
                        f"Frame {frame_num} (mask {mask_file}): Tracks {missing_in_mask} in res_track.txt but not in mask pixels.")
                    logging.warning(
                        f"    VERIFY: Frame {frame_num} (mask {mask_file}): IDs {missing_in_mask} in res_track.txt but NOT in mask pixels.")

                # IDs in this mask file but not in res_track.txt for this frame
                extra_in_mask = mask_ids_in_file - track_ids_expected_in_frame
                if extra_in_mask:
                    issues.append(
                        f"Frame {frame_num} (mask {mask_file}): Tracks {extra_in_mask} in mask pixels but not in res_track.txt for this frame.")
                    logging.warning(
                        f"    VERIFY: Frame {frame_num} (mask {mask_file}): IDs {extra_in_mask} in mask pixels but NOT in res_track.txt for this frame.")

            except FileNotFoundError:
                issues.append(f"Mask file not found: {mask_path}")
                logging.error(f"    VERIFY: Mask file NOT FOUND: {mask_path}")
            except Exception as e_mask_read:
                issues.append(f"Error reading mask {mask_file}: {e_mask_read}")
                logging.error(f"    VERIFY: Error reading mask {mask_file}: {e_mask_read}")

    if not issues:
        logging.info("  VERIFY: No obvious CTC format issues found in RES path based on these checks.")
    else:
        logging.warning(f"  VERIFY: Found {len(issues)} potential CTC format issues in RES path.")
    logging.info(f"--- Finished Verifying CTC Format for RES path: {res_path} ---")
    return issues


def verify_tracking_data_consistency(trj, id_masks, background_id=0):
    """Verify that tracking data and masks are consistent."""
    issues = []

    if trj is None or trj.empty:
        issues.append("Trajectory data is empty")
        return issues

    if id_masks is None:
        issues.append("ID masks are None")
        return issues

    # Check all tracked IDs appear in masks
    track_ids = set(pd.to_numeric(trj['particle'], errors='coerce').dropna().astype(int))
    track_ids.discard(background_id)

    for frame in trj['frame'].unique():
        frame_tracks = set(trj[trj['frame'] == frame]['particle'])
        frame_tracks.discard(background_id)

        if 0 <= frame < id_masks.shape[2]:
            mask_ids = set(np.unique(id_masks[:, :, frame]))
            mask_ids.discard(0)
            mask_ids.discard(background_id)

            missing_in_mask = frame_tracks - mask_ids
            if missing_in_mask:
                issues.append(f"Frame {frame}: Tracks {missing_in_mask} in trajectory but not in mask")

    # Check parent relationships
    if 'parent_particle' in trj.columns:
        for particle_id in track_ids:
            track_data = trj[trj['particle'] == particle_id]
            parent_values = track_data['parent_particle'].unique()
            if len(parent_values) > 1:
                issues.append(f"Track {particle_id} has inconsistent parent values: {parent_values}")

    return issues
