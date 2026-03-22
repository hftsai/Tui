import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

# If available, import the lineage tree widget for advanced visualization
try:
    from lineage_tree import LineageTreeWidget
    HAS_LINEAGE_WIDGET = True
except ImportError:
    HAS_LINEAGE_WIDGET = False


def save_singular_tracks(trj, cell_lineage, ancestry, output_dir):
    """
    Save singular complete tracks (not involved in mitosis or fusion) to CSV and plot their trees.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Identify all mitosis and fusion related IDs
    mitosis_parents = {int(p_id) for p_id, d_ids in cell_lineage.items() if len(d_ids) >= 2}
    mitosis_daughters = {int(d_id) for d_ids in cell_lineage.values() if len(d_ids) >= 2 for d_id in d_ids}
    fusion_children = {int(c_id) for c_id, p_ids in ancestry.items() if len(p_ids) >= 2}
    fusion_parents = {int(p_id) for p_ids in ancestry.values() if len(p_ids) >= 2 for p_id in p_ids}
    event_related_ids = mitosis_parents | mitosis_daughters | fusion_children | fusion_parents
    # All track IDs
    all_ids = set(trj['particle'].unique())
    singular_ids = sorted(list(all_ids - event_related_ids))
    singular_trj = trj[trj['particle'].isin(singular_ids)]
    out_csv = os.path.join(output_dir, 'singular_tracks.csv')
    singular_trj.to_csv(out_csv, index=False)
    logging.info(f"Saved singular complete tracks to {out_csv}")
    # Plot each as a simple tree
    for tid in singular_ids:
        plot_single_track_tree(trj, tid, os.path.join(output_dir, f'singular_tree_{tid}.png'))

def save_mitosis_tracks(trj, cell_lineage, output_dir):
    """
    Save mitosis tracks (parents and daughters) to CSV and plot their trees.
    """
    os.makedirs(output_dir, exist_ok=True)
    mitosis_parents = {int(p_id) for p_id, d_ids in cell_lineage.items() if len(d_ids) >= 2}
    mitosis_daughters = {int(d_id) for d_ids in cell_lineage.values() if len(d_ids) >= 2 for d_id in d_ids}
    mitosis_ids = sorted(list(mitosis_parents | mitosis_daughters))
    mitosis_trj = trj[trj['particle'].isin(mitosis_ids)]
    out_csv = os.path.join(output_dir, 'mitosis_tracks.csv')
    mitosis_trj.to_csv(out_csv, index=False)
    logging.info(f"Saved mitosis tracks to {out_csv}")
    # Plot each mitosis event as a tree (parent and its daughters)
    for parent_id in mitosis_parents:
        plot_mitosis_tree(trj, cell_lineage, parent_id, os.path.join(output_dir, f'mitosis_tree_{parent_id}.png'))

def save_fusion_tracks(trj, ancestry, output_dir):
    """
    Save fusion tracks (parents and children) to CSV and plot their trees.
    """
    os.makedirs(output_dir, exist_ok=True)
    fusion_children = {int(c_id) for c_id, p_ids in ancestry.items() if len(p_ids) >= 2}
    fusion_parents = {int(p_id) for p_ids in ancestry.values() if len(p_ids) >= 2 for p_id in p_ids}
    fusion_ids = sorted(list(fusion_children | fusion_parents))
    fusion_trj = trj[trj['particle'].isin(fusion_ids)]
    out_csv = os.path.join(output_dir, 'fusion_tracks.csv')
    fusion_trj.to_csv(out_csv, index=False)
    logging.info(f"Saved fusion tracks to {out_csv}")
    # Plot each fusion event as a tree (child and its parents)
    for child_id in fusion_children:
        plot_fusion_tree(trj, ancestry, child_id, os.path.join(output_dir, f'fusion_tree_{child_id}.png'))

def plot_single_track_tree(trj, track_id, out_path):
    """
    Plot a simple tree for a singular track (just a line for its lifespan).
    """
    frames = trj[trj['particle'] == track_id]['frame']
    plt.figure(figsize=(3, 2))
    plt.plot(frames, [track_id]*len(frames), marker='o')
    plt.title(f'Singular Track {track_id}')
    plt.xlabel('Frame')
    plt.ylabel('Track ID')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_mitosis_tree(trj, cell_lineage, parent_id, out_path):
    """
    Plot a tree for a mitosis event: parent and its daughters.
    """
    daughters = cell_lineage.get(parent_id, [])
    parent_frames = trj[trj['particle'] == parent_id]['frame']
    plt.figure(figsize=(4, 3))
    # Parent line
    plt.plot(parent_frames, [parent_id]*len(parent_frames), label=f'Parent {parent_id}', color='blue')
    # Daughters
    colors = ['red', 'green', 'orange', 'purple']
    for i, d_id in enumerate(daughters):
        d_frames = trj[trj['particle'] == d_id]['frame']
        plt.plot(d_frames, [d_id]*len(d_frames), label=f'Daughter {d_id}', color=colors[i%len(colors)])
    plt.title(f'Mitosis Event: Parent {parent_id}')
    plt.xlabel('Frame')
    plt.ylabel('Track ID')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_fusion_tree(trj, ancestry, child_id, out_path):
    """
    Plot a tree for a fusion event: child and its parents.
    """
    parents = ancestry.get(child_id, [])
    child_frames = trj[trj['particle'] == child_id]['frame']
    plt.figure(figsize=(4, 3))
    # Child line
    plt.plot(child_frames, [child_id]*len(child_frames), label=f'Child {child_id}', color='red')
    # Parents
    colors = ['blue', 'green', 'orange', 'purple']
    for i, p_id in enumerate(parents):
        p_frames = trj[trj['particle'] == p_id]['frame']
        plt.plot(p_frames, [p_id]*len(p_frames), label=f'Parent {p_id}', color=colors[i%len(colors)])
    plt.title(f'Fusion Event: Child {child_id}')
    plt.xlabel('Frame')
    plt.ylabel('Track ID')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close() 