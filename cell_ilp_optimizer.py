# cell_ilp_optimizer.py
# New module to optimize ILP tracking parameters based on user-defined targets.
# V10 (Gemini): Fixed ValueError from skopt by ensuring n_calls is always valid.
# V12 (This Update): Made iteration handling more transparent and predictable.

import numpy as np
import logging
import traceback
from collections import defaultdict
import os
import csv
from datetime import datetime

# The optimizer library. This is a dependency that needs to be installed.
# pip install scikit-optimize
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# Import necessary components from the existing codebase
# Note: Ensure cell_tracking_ilp.py is in the Python path
from cell_tracking_ilp import TrackingGraph, solve_ilp, reconstruct_trajectories

# Logging is configured in the main application. We just get the logger for this module.
logger = logging.getLogger(__name__)

# This context is a way to pass necessary data (like masks and targets)
# into the objective function, which the optimizer calls with only one argument (the parameters).
optimization_context = {}


def objective_function(params):
    """
    The objective function for the optimizer.
    It runs the tracking with a given set of parameters and returns an "error" score
    based on how far the results are from the user's targets.
    """
    try:
        # Unpack the parameters that the optimizer is testing
        cost_transition, cost_mitosis, cost_fusion, cost_appearance, cost_disappearance, max_dist = params

        # Get data and targets from the global context
        masks = optimization_context.get('masks')
        targets = optimization_context.get('targets')
        log_filepath = optimization_context.get('log_filepath')

        # Retrieve solver settings from the context
        solver = optimization_context.get('solver', 'scipy')  # Default to scipy if not found
        gurobi_params = optimization_context.get('gurobi_params')

        if masks is None or targets is None:
            logger.error("Optimizer objective function missing masks or targets in context.")
            return 1e9  # Return a very high error

        # 1. Run the full ILP tracking pipeline with the given parameters
        graph = TrackingGraph(
            masks=masks,
            max_distance=int(max_dist),  # Ensure max_distance is an integer
            cost_weight_transition=cost_transition,
            cost_weight_mitosis=cost_mitosis,
            cost_weight_fusion=cost_fusion,
            cost_appearance=cost_appearance,
            cost_disappearance=cost_disappearance
        )
        graph.build()

        # Pass the solver and gurobi_params to solve_ilp
        selected_edges = solve_ilp(graph, solver=solver, gurobi_params=gurobi_params)

        # If the solver fails, return a high penalty
        if selected_edges is None:
            logger.warning(f"ILP solver failed for params: {[f'{p:.2f}' for p in params]}. Assigning high error.")
            return 1e6

        trj, lineage = reconstruct_trajectories(graph, selected_edges)

        # If tracking produces no results, return a high penalty
        if trj.empty:
            logger.warning(
                f"Tracking produced no tracks for params: {[f'{p:.2f}' for p in params]}. Assigning high error.")
            if log_filepath:
                with open(log_filepath, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        cost_transition, cost_mitosis, cost_fusion, cost_appearance, cost_disappearance, max_dist,
                        'N/A', 'N/A', 'N/A', 1e5
                    ])
            return 1e5

        # 2. Calculate the metrics from the tracking result
        num_tracks = trj['particle'].nunique()
        num_cells_frame_zero = trj[trj['frame'] == 0]['particle'].nunique()

        # Count mitosis events (parent with >= 2 children)
        num_mitosis = 0
        if lineage:
            num_mitosis = sum(1 for children in lineage.values() if len(children) >= 2)

        # Count fusion events (child with >= 2 parents)
        child_to_parents = defaultdict(list)
        if lineage:
            for parent, children in lineage.items():
                for child in children:
                    child_to_parents[child].append(parent)
        num_fusion = sum(1 for parents in child_to_parents.values() if len(parents) >= 2)

        # 3. Calculate the normalized squared error against the targets
        error = 0.0

        if targets.get('num_cells_frame_zero') is not None:
            target_cells_f0 = targets['num_cells_frame_zero']
            error += ((num_cells_frame_zero - target_cells_f0) / (target_cells_f0 + 1e-6)) ** 2

        if targets.get('num_tracks') is not None:
            target_tracks = targets['num_tracks']
            error += ((num_tracks - target_tracks) / (target_tracks + 1e-6)) ** 2

        if targets.get('num_mitosis') is not None:
            target_mitosis = targets['num_mitosis']
            error += ((num_mitosis - target_mitosis) / (target_mitosis + 1e-6)) ** 2

        if targets.get('num_fusion') is not None:
            target_fusion = targets['num_fusion']
            error += ((num_fusion - target_fusion) / (target_fusion + 1e-6)) ** 2

        console_log_msg = (
            f"Params: [Tr:{cost_transition:.1f}, Mit:{cost_mitosis:.1f}, Fus:{cost_fusion:.1f}, "
            f"App:{cost_appearance:.1f}, Dis:{cost_disappearance:.1f}, Dist:{max_dist}] -> "
            f"Results: [F0_Cells:{num_cells_frame_zero}, Mit:{num_mitosis}, Fus:{num_fusion}] -> Error: {error:.4f}"
        )
        logger.info(console_log_msg)

        if log_filepath:
            try:
                with open(log_filepath, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        cost_transition, cost_mitosis, cost_fusion, cost_appearance, cost_disappearance, max_dist,
                        num_cells_frame_zero, num_mitosis, num_fusion, error
                    ])
            except IOError as e:
                logger.error(f"Could not write to optimizer log file {log_filepath}: {e}")

        # Update the progress bar if a callback is provided
        progress_callback = optimization_context.get('progress_callback')
        if progress_callback:
            if not progress_callback():
                raise StopIteration("Optimization cancelled by user.")

        return error

    except StopIteration as e:
        raise e
    except Exception as e:
        logger.error(f"Exception in objective function: {e}\n{traceback.format_exc()}")
        return 1e9


def run_optimization(main_app_state, targets, n_calls, progress_callback):
    """
    Sets up and runs the Bayesian optimization.
    Accepts n_calls to control the number of iterations.
    """
    global optimization_context
    if not SKOPT_AVAILABLE:
        raise ImportError(
            "The 'scikit-optimize' library is required for this feature. Please install it using 'pip install scikit-optimize'.")

    data_path = main_app_state.get('path_in')
    log_dir = os.path.join(data_path, "Optimizer_Logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"ilp_optimization_log_{timestamp}.csv"
    log_filepath = os.path.join(log_dir, log_filename)

    log_header = [
        "cost_transition", "cost_mitosis", "cost_fusion", "cost_appearance", "cost_disappearance", "max_dist",
        "result_cells_f0", "result_mitosis", "result_fusion", "error_score"
    ]
    try:
        with open(log_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)
        logger.info(f"Optimizer log will be saved to: {log_filepath}")
    except IOError as e:
        logger.error(f"Failed to create optimizer log file: {e}")
        log_filepath = None

    optimization_context['masks'] = main_app_state.get('raw_masks')
    optimization_context['targets'] = targets
    optimization_context['progress_callback'] = progress_callback
    optimization_context['log_filepath'] = log_filepath

    solver = main_app_state['params'].get('ILP Solver', ['scipy'])[0]
    optimization_context['solver'] = solver

    if solver == 'gurobi':
        gurobi_params = {
            "WLSACCESSID": main_app_state['params']['Gurobi WLSACCESSID'][0],
            "WLSSECRET": main_app_state['params']['Gurobi WLSSECRET'][0],
            "LICENSEID": main_app_state['params']['Gurobi LICENSEID'][0],
        }
        optimization_context['gurobi_params'] = gurobi_params
    else:
        optimization_context['gurobi_params'] = None

    space = [
        Real(0.1, 15.0, name='cost_transition'),
        Real(5.0, 250.0, name='cost_mitosis'),
        Real(5.0, 250.0, name='cost_fusion'),
        Real(10.0, 500.0, name='cost_appearance'),
        Real(10.0, 500.0, name='cost_disappearance'),
        Integer(20, 150, name='max_dist')
    ]

    params = main_app_state['params']
    x0 = [
        params['ILP Transition Cost Weight'][0],
        params['ILP Mitosis Cost'][0],
        params['ILP Fusion Cost'][0],
        params['ILP Appearance Cost'][0],
        params['ILP Disappearance Cost'][0],
        params['ILP Max Search Distance'][0]
    ]

    # --- FIX: Make iteration handling transparent ---
    # Define a fixed number of random points to build the initial model, in addition to x0.
    n_initial_points = 2

    # To prevent crashes, the total number of calls must be at least the number of initial points.
    # We assume skopt uses 1 point from x0, plus n_initial_points.
    min_required_calls = 1 + n_initial_points
    if n_calls < min_required_calls:
        logger.warning(
            f"User requested n_calls={n_calls}, but a minimum of {min_required_calls} is required "
            f"to run. Adjusting to {min_required_calls}."
        )
        n_calls = min_required_calls

    # Also warn the user if they won't get any optimization steps.
    # This happens if n_calls equals the number of initial points.
    if n_calls == min_required_calls:
        logger.warning(
            f"With n_calls={n_calls}, the optimizer will only evaluate initial points and "
            f"perform no optimization steps. Increase n_calls to get optimization."
        )
    # --- END FIX ---

    logger.info(
        f"Starting optimizer with {n_calls} total iterations ({n_initial_points} random points + 1 from current params).")

    best_params_result = None
    best_score_result = None
    try:
        result = gp_minimize(
            func=objective_function,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            x0=x0,
            random_state=42
        )
        best_params_result = {
            'ILP Transition Cost Weight': result.x[0],
            'ILP Mitosis Cost': result.x[1],
            'ILP Fusion Cost': result.x[2],
            'ILP Appearance Cost': result.x[3],
            'ILP Disappearance Cost': result.x[4],
            'ILP Max Search Distance': result.x[5]
        }
        best_score_result = result.fun

    except StopIteration:
        logger.info("Optimization was stopped prematurely by the user.")
        return None, None, log_filepath
    except Exception as e:
        logger.error(f"An unexpected error occurred during optimization: {e}", exc_info=True)
        return None, None, log_filepath

    return best_params_result, best_score_result, log_filepath
