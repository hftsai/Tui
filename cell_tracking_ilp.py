# cell_tracking_ilp.py
# New module for Integer Linear Programming (ILP) based cell tracking.
# V13 (This Update): Reverted to a modified version of the original reconstruction logic
# to ensure fusions are captured, and made parent assignment deterministic to fix ID switches.

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import milp, Bounds, LinearConstraint
from skimage.measure import regionprops
import logging
from collections import defaultdict
from itertools import combinations

# --- Availability Flags ---
ILP_AVAILABLE = True

# --- Solver Imports ---
# Try to import Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
    logging.info("Gurobi solver found and imported successfully.")
except ImportError:
    GUROBI_AVAILABLE = False
    logging.warning("Gurobi not found. ILP tracking will be limited to the SciPy solver.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def determine_background_label(masks):
    """
    Automatically determines the background label from mask data.
    Background is typically the most frequent label or the label with the largest total area.
    """
    if masks is None:
        return 0
    
    # Analyze all frames to find the most common label
    label_areas = {}
    
    for frame_idx in range(masks.shape[2]):
        frame_mask = masks[:, :, frame_idx]
        unique_labels, counts = np.unique(frame_mask, return_counts=True)
        
        for label, count in zip(unique_labels, counts):
            if label in label_areas:
                label_areas[label] += count
            else:
                label_areas[label] = count
    
    if not label_areas:
        return 0
    
    # Find the label with the largest total area (likely background)
    background_label = max(label_areas.keys(), key=lambda x: label_areas[x])
    
    logger.info(f"Determined background label: {background_label} (total area: {label_areas[background_label]})")
    logger.info(f"All labels found: {sorted(label_areas.keys())}")
    
    return background_label


class TrackingGraph:
    """
    Represents the cell tracking problem as a graph, where nodes are cell
    detections and edges represent potential links (transitions, mitoses, fusions).
    The costs of these events are configurable.
    """

    def __init__(self, masks,
                 max_distance=50,
                 max_frame_skip=2,
                 cost_weight_transition=1.0,
                 cost_weight_mitosis=25.0,
                 cost_weight_fusion=25.0,
                 cost_appearance=100.0,
                 cost_disappearance=100.0):
        """
        Initializes the tracking graph with configurable costs.
        """
        self.masks = masks
        self.max_distance = max_distance
        self.max_frame_skip = max_frame_skip
        self.cost_weight_transition = cost_weight_transition
        self.cost_weight_mitosis = cost_weight_mitosis
        self.cost_weight_fusion = cost_weight_fusion
        self.cost_appearance = cost_appearance
        self.cost_disappearance = cost_disappearance

        self.nodes = []
        self.edges = []
        self.node_features = {}
        self._extract_nodes_from_masks()

    def _extract_nodes_from_masks(self):
        """
        Extracts cell detections (nodes) from the segmentation masks.
        """
        logger.info("Extracting nodes from masks...")
        
        # Determine background label for this tracking run
        background_label = determine_background_label(self.masks)
        logger.info(f"Using background label: {background_label} for node extraction")
        
        node_counter = 0
        for frame_idx in range(self.masks.shape[2]):
            frame_mask = self.masks[:, :, frame_idx]
            props = regionprops(frame_mask, intensity_image=frame_mask)
            for prop in props:
                if prop.area == 0: continue
                if prop.label == background_label: continue  # Skip background regions
                node_id = node_counter
                self.nodes.append(node_id)
                self.node_features[node_id] = {
                    'frame_label_id': (frame_idx, prop.label),
                    'centroid': prop.centroid,
                    'area': prop.area,
                    'frame': frame_idx,
                    'label': prop.label,
                    'y': prop.centroid[0],
                    'x': prop.centroid[1],
                    'equivalent_diameter': prop.equivalent_diameter
                }
                node_counter += 1
        logger.info(f"Extracted {len(self.nodes)} nodes.")

    def build(self):
        """
        Builds the graph by creating edges between nodes based on tracking hypotheses.
        """
        logger.info("Building tracking graph with configurable costs...")
        self._add_transition_edges()
        self._add_mitosis_edges()
        self._add_fusion_edges()
        logger.info(f"Built graph with {len(self.edges)} potential edges (transitions, mitoses, and fusions).")

    def _add_transition_edges(self):
        """Adds transition edges (1-to-1) between subsequent frames."""
        logger.info("Adding transition edges...")
        nodes_by_frame = defaultdict(list)
        for node_id in self.nodes:
            nodes_by_frame[self.node_features[node_id]['frame']].append(node_id)

        for frame_idx in range(self.masks.shape[2] - 1):
            nodes_t = nodes_by_frame.get(frame_idx, [])
            nodes_t1 = nodes_by_frame.get(frame_idx + 1, [])

            if not nodes_t or not nodes_t1:
                continue

            centroids_t = np.array([self.node_features[n]['centroid'] for n in nodes_t])
            centroids_t1 = np.array([self.node_features[n]['centroid'] for n in nodes_t1])

            dist_matrix = cdist(centroids_t, centroids_t1)

            for i, node_t_id in enumerate(nodes_t):
                for j, node_t1_id in enumerate(nodes_t1):
                    if dist_matrix[i, j] < self.max_distance:
                        transition_cost = dist_matrix[i, j] * self.cost_weight_transition
                        self.edges.append({
                            'source': node_t_id,
                            'target': node_t1_id,
                            'type': 'transition',
                            'cost': transition_cost
                        })

    def _add_mitosis_edges(self):
        """Adds mitosis edges (1-to-2)."""
        logger.info("Adding mitosis edges...")
        nodes_by_frame = defaultdict(list)
        for node_id in self.nodes:
            nodes_by_frame[self.node_features[node_id]['frame']].append(node_id)

        for frame_idx in range(self.masks.shape[2] - 1):
            nodes_t = nodes_by_frame.get(frame_idx, [])
            nodes_t1 = nodes_by_frame.get(frame_idx + 1, [])

            if not nodes_t or len(nodes_t1) < 2:
                continue

            parent_centroids = np.array([self.node_features[n]['centroid'] for n in nodes_t])
            daughter_centroids = np.array([self.node_features[n]['centroid'] for n in nodes_t1])

            for i, parent_node_id in enumerate(nodes_t):
                parent_centroid = parent_centroids[i]
                dists = np.linalg.norm(daughter_centroids - parent_centroid, axis=1)
                close_daughters_indices = np.where(dists < self.max_distance)[0]

                if len(close_daughters_indices) < 2:
                    continue

                for d1_idx_ptr, d2_idx_ptr in combinations(close_daughters_indices, 2):
                    daughter1_node_id = nodes_t1[d1_idx_ptr]
                    daughter2_node_id = nodes_t1[d2_idx_ptr]

                    mitosis_cost = self.cost_weight_mitosis + (dists[d1_idx_ptr] + dists[d2_idx_ptr]) / 2
                    self.edges.append({
                        'source': parent_node_id,
                        'targets': [daughter1_node_id, daughter2_node_id],
                        'type': 'mitosis',
                        'cost': mitosis_cost
                    })

    def _add_fusion_edges(self):
        """Adds fusion edges (2-to-1)."""
        logger.info("Adding fusion edges...")
        nodes_by_frame = defaultdict(list)
        for node_id in self.nodes:
            nodes_by_frame[self.node_features[node_id]['frame']].append(node_id)

        for frame_idx in range(self.masks.shape[2] - 1):
            nodes_t = nodes_by_frame.get(frame_idx, [])
            nodes_t1 = nodes_by_frame.get(frame_idx + 1, [])

            if len(nodes_t) < 2 or not nodes_t1:
                continue

            parent_centroids = np.array([self.node_features[n]['centroid'] for n in nodes_t])
            child_centroids = np.array([self.node_features[n]['centroid'] for n in nodes_t1])

            for i, child_node_id in enumerate(nodes_t1):
                child_centroid = child_centroids[i]
                dists = np.linalg.norm(parent_centroids - child_centroid, axis=1)
                close_parents_indices = np.where(dists < self.max_distance)[0]

                if len(close_parents_indices) < 2:
                    continue

                for p1_idx_ptr, p2_idx_ptr in combinations(close_parents_indices, 2):
                    parent1_node_id = nodes_t[p1_idx_ptr]
                    parent2_node_id = nodes_t[p2_idx_ptr]

                    fusion_cost = self.cost_weight_fusion + (dists[p1_idx_ptr] + dists[p2_idx_ptr]) / 2
                    self.edges.append({
                        'sources': [parent1_node_id, parent2_node_id],
                        'target': child_node_id,
                        'type': 'fusion',
                        'cost': fusion_cost
                    })


def _solve_with_gurobi(graph, gurobi_params):
    """Solves the ILP using the Gurobi solver."""
    logger.info("Attempting to solve ILP with Gurobi...")
    try:
        # Set up Gurobi environment with WLS credentials
        with gp.Env(params=gurobi_params) as env:
            with gp.Model(env=env) as model:
                model.setParam('LogToConsole', 0)  # Suppress console output
                num_nodes = len(graph.nodes)
                num_edges = len(graph.edges)

                # Create variables
                edge_vars = model.addVars(num_edges, vtype=GRB.BINARY, name="edge")
                appear_vars = model.addVars(num_nodes, vtype=GRB.BINARY, name="appear")
                disappear_vars = model.addVars(num_nodes, vtype=GRB.BINARY, name="disappear")

                # Set objective function
                obj = gp.LinExpr()
                obj.add(edge_vars.prod([e['cost'] for e in graph.edges]))
                obj.add(appear_vars.prod([graph.cost_appearance] * num_nodes))
                obj.add(disappear_vars.prod([graph.cost_disappearance] * num_nodes))
                model.setObjective(obj, GRB.MINIMIZE)

                node_to_index = {node_id: i for i, node_id in enumerate(graph.nodes)}

                # Add flow conservation constraints
                for i in range(num_nodes):
                    in_flow = gp.LinExpr(appear_vars[i])
                    out_flow = gp.LinExpr(disappear_vars[i])

                    # Find all edges connected to this node
                    for edge_idx, edge in enumerate(graph.edges):
                        if edge['type'] in ('transition', 'fusion') and edge['target'] == graph.nodes[i]:
                            in_flow += edge_vars[edge_idx]
                        if edge['type'] == 'mitosis' and graph.nodes[i] in edge['targets']:
                            in_flow += edge_vars[edge_idx]
                        if edge['type'] in ('transition', 'mitosis') and edge['source'] == graph.nodes[i]:
                            out_flow += edge_vars[edge_idx]
                        if edge['type'] == 'fusion' and graph.nodes[i] in edge['sources']:
                            out_flow += edge_vars[edge_idx]

                    model.addConstr(in_flow == 1, name=f"in_flow_{i}")
                    model.addConstr(out_flow == 1, name=f"out_flow_{i}")

                # Optimize model
                model.optimize()

                if model.Status == GRB.OPTIMAL:
                    logger.info(f"Gurobi found an optimal solution. Objective value: {model.ObjVal:.2f}")
                    selected_edges_indices = [i for i, var in enumerate(edge_vars.values()) if var.X > 0.5]
                    return [graph.edges[i] for i in selected_edges_indices]
                else:
                    logger.warning(f"Gurobi did not find an optimal solution. Status code: {model.Status}")
                    return None

    except gp.GurobiError as e:
        logger.error(f"Gurobi error occurred: {e.message} (Error code: {e.errno})")
        logger.error("This may be due to incorrect license credentials or a network issue.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during Gurobi solving: {e}", exc_info=True)
        return None


def _solve_with_scipy(graph):
    """Solves the ILP using the SciPy solver."""
    logger.info("Solving ILP with SciPy...")
    num_nodes = len(graph.nodes)
    num_edges = len(graph.edges)
    num_vars = num_edges + 2 * num_nodes

    edge_costs = [edge['cost'] for edge in graph.edges]
    appearance_costs = [graph.cost_appearance] * num_nodes
    disappearance_costs = [graph.cost_disappearance] * num_nodes
    c = np.array(edge_costs + appearance_costs + disappearance_costs)

    node_to_index = {node_id: i for i, node_id in enumerate(graph.nodes)}

    A_eq = np.zeros((2 * num_nodes, num_vars))
    b_eq = np.ones(2 * num_nodes)

    for edge_idx, edge in enumerate(graph.edges):
        if edge['type'] in ('transition', 'mitosis'):
            source_node_idx = node_to_index[edge['source']]
            out_flow_row_idx = num_nodes + source_node_idx
            A_eq[out_flow_row_idx, edge_idx] = 1
        elif edge['type'] == 'fusion':
            for source_node in edge['sources']:
                source_node_idx = node_to_index[source_node]
                out_flow_row_idx = num_nodes + source_node_idx
                A_eq[out_flow_row_idx, edge_idx] = 1
        if edge['type'] in ('transition', 'fusion'):
            target_node_idx = node_to_index[edge['target']]
            in_flow_row_idx = target_node_idx
            A_eq[in_flow_row_idx, edge_idx] = 1
        elif edge['type'] == 'mitosis':
            for target_node in edge['targets']:
                target_node_idx = node_to_index[target_node]
                in_flow_row_idx = target_node_idx
                A_eq[in_flow_row_idx, edge_idx] = 1

    for i in range(num_nodes):
        appear_var_idx = num_edges + i
        disappear_var_idx = num_edges + num_nodes + i
        in_flow_row_idx = i
        A_eq[in_flow_row_idx, appear_var_idx] = 1
        out_flow_row_idx = num_nodes + i
        A_eq[out_flow_row_idx, disappear_var_idx] = 1

    constraints = LinearConstraint(A_eq, b_eq, b_eq)
    bounds = Bounds(lb=np.zeros(num_vars), ub=np.ones(num_vars))
    integrality = np.ones(num_vars)

    logger.info(f"Solving MILP with {num_nodes} nodes, {num_edges} edges ({num_vars} total variables)...")
    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

    if not res.success:
        logger.warning(f"SciPy ILP solver did not find an optimal solution. Status: {res.message}")
        return None

    logger.info(f"SciPy ILP solved successfully. Objective value: {res.fun:.2f}. Reconstructing tracks...")
    selected_edges_indices = np.where(np.round(res.x[:num_edges]) > 0.5)[0]
    return [graph.edges[i] for i in selected_edges_indices]


def solve_ilp(graph: TrackingGraph, solver='scipy', gurobi_params=None):
    """
    Solves the cell tracking problem using Integer Linear Programming (ILP).
    Dispatches to the selected solver (Gurobi or SciPy).
    """
    if not graph.nodes:
        logger.warning("No nodes in graph, cannot solve ILP.")
        return []

    if solver == 'gurobi' and GUROBI_AVAILABLE:
        if gurobi_params is None:
            logger.error("Gurobi solver selected, but no license parameters were provided.")
            logger.info("Falling back to SciPy solver.")
            return _solve_with_scipy(graph)

        result = _solve_with_gurobi(graph, gurobi_params)
        if result is not None:
            return result
        else:
            logger.warning("Gurobi solver failed. Falling back to SciPy solver.")
            return _solve_with_scipy(graph)
    else:
        if solver == 'gurobi' and not GUROBI_AVAILABLE:
            logger.warning("Gurobi solver selected but not available. Falling back to SciPy solver.")
        return _solve_with_scipy(graph)


def reconstruct_trajectories(graph, selected_edges):
    """
    Reconstructs trajectories from the selected edges.
    This version uses a logic similar to the original to ensure events are captured,
    but with deterministic parent assignment to prevent ID switches.
    """
    output_columns = ['frame', 'particle', 'parent_particle', 'y', 'x',
                      'original_mask_label', 'area', 'equivalent_diameter']

    if selected_edges is None:
        logger.error("Selected edges is None - solver likely failed")
        return pd.DataFrame(columns=output_columns), {}

    logger.info(f"Reconstructing trajectories from {len(selected_edges)} selected edges.")

    # Build relationship maps from selected edges
    parent_to_children = defaultdict(list)
    child_to_parents = defaultdict(list)
    for edge in selected_edges:
        if edge['type'] == 'transition':
            s, t = edge['source'], edge['target']
            parent_to_children[s].append(t)
            child_to_parents[t].append(s)
        elif edge['type'] == 'mitosis':
            s, (d1, d2) = edge['source'], edge['targets']
            parent_to_children[s].extend([d1, d2])
            child_to_parents[d1].append(s)
            child_to_parents[d2].append(s)
        elif edge['type'] == 'fusion':
            (p1, p2), t = edge['sources'], edge['target']
            parent_to_children[p1].append(t)
            parent_to_children[p2].append(t)
            child_to_parents[t].extend([p1, p2])

    # --- Step 1: Assign particle IDs to each node ---
    # This logic is based on the original version which correctly identified fusions.
    node_to_particle_id = {}
    particle_id_counter = 1
    nodes_by_frame = sorted(graph.nodes, key=lambda n: graph.node_features[n]['frame'])

    for node_id in nodes_by_frame:
        if node_id in node_to_particle_id:
            continue

        parents = child_to_parents.get(node_id, [])

        if not parents:
            # Node is a track start
            node_to_particle_id[node_id] = particle_id_counter
            particle_id_counter += 1
        else:
            # This node has one or more parents.
            # We need to decide if it continues an existing track or starts a new one.

            # Find the particle IDs of all parent nodes that have already been processed
            parent_pids = sorted([node_to_particle_id[p] for p in parents if p in node_to_particle_id])

            if not parent_pids:
                # This case should not happen if nodes are processed by frame.
                # As a fallback, start a new track.
                logger.warning(
                    f"Node {node_id} has parents that have not been assigned a particle ID yet. Starting new track.")
                node_to_particle_id[node_id] = particle_id_counter
                particle_id_counter += 1
                continue

            # Check for a simple 1-to-1 transition
            is_simple_transition = (len(parents) == 1 and
                                    len(parent_to_children.get(parents[0], [])) == 1)

            if is_simple_transition:
                # Continue the parent's track
                node_to_particle_id[node_id] = parent_pids[0]
            else:
                # This is a mitosis or fusion, so it starts a new track.
                node_to_particle_id[node_id] = particle_id_counter
                particle_id_counter += 1

    # --- Step 2: Assemble the final trajectory DataFrame ---
    trajectory_data = []
    lineage = defaultdict(list)
    for node_id, p_id in node_to_particle_id.items():
        features = graph.node_features[node_id]
        parent_nodes = child_to_parents.get(node_id, [])

        parent_particle_id = 0  # Default to 0 (no parent)
        if parent_nodes:
            # Get all unique parent particle IDs
            parent_pids = set(node_to_particle_id[p] for p in parent_nodes if p in node_to_particle_id)

            # The parent_particle field should be for a *different* track ID.
            # This handles lineage events (mitosis/fusion).
            lineage_parent_pids = parent_pids - {p_id}

            if lineage_parent_pids:
                # This is a lineage event. Choose the parent deterministically (e.g., the smallest ID).
                parent_particle_id = sorted(list(lineage_parent_pids))[0]
                for p_pid in lineage_parent_pids:
                    if p_id not in lineage[p_pid]:
                        lineage[p_pid].append(p_id)

        trajectory_data.append({
            'frame': features['frame'],
            'particle': p_id,
            'parent_particle': parent_particle_id,
            'y': features['y'],
            'x': features['x'],
            'original_mask_label': features['label'],
            'area': features['area'],
            'equivalent_diameter': features['equivalent_diameter']
        })

    if not trajectory_data:
        logger.warning("No trajectory data reconstructed")
        return pd.DataFrame(columns=output_columns), {}

    trj_df = pd.DataFrame(trajectory_data)

    # Post-process to ensure parent_particle is consistent for the whole track
    # The first frame of a track defines its parent
    parent_map = trj_df.sort_values('frame').groupby('particle')['parent_particle'].first()
    trj_df['parent_particle'] = trj_df['particle'].map(parent_map)

    trj_df = trj_df.sort_values(by=['particle', 'frame']).reset_index(drop=True)
    trj_df['parent_particle'] = trj_df['parent_particle'].fillna(0).astype(int)

    logger.info(f"Reconstructed {len(trj_df)} trajectory points for {trj_df['particle'].nunique()} tracks")
    logger.info(f"Lineage events: {len(lineage)} parents with children")
    if any(len(v) > 1 for v in child_to_parents.values()):
        logger.info("Fusion events were successfully reconstructed.")

    return trj_df, dict(lineage)
