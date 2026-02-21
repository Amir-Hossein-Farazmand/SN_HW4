"""
Bonus Question 6: Operation "Dormant Cell" Detection
Custom algorithm for identifying suspicious hidden communities
"""

import os
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

def find_dormant_cell(G, partition):
    """
    Identify the most suspicious community based on dormant cell characteristics
    """
    # Convert partition to communities
    communities = defaultdict(set)
    for node, comm_id in partition.items():
        communities[comm_id].add(node)

    # Calculate average community size
    avg_size = np.mean([len(comm) for comm in communities.values()])

    # Calculate closeness centrality for the entire graph once (for peripherality metric)
    closeness = nx.closeness_centrality(G)

    # Weights
    w1, w2, w3, w4 = 0.35, 0.35, 0.15, 0.15

    best_community = None
    max_suspicion = -1
    all_scores = {}
    best_stats = {}
    best_reasoning = ""

    total_nodes = len(G.nodes())

    for comm_id, members in communities.items():
        if len(members) < 2:  # Skip single-node communities
            continue

        comm_size = len(members)
        
        # 1. Internal Density
        subgraph = G.subgraph(members)
        # nx.density calculates: actual edges / possible edges
        density_internal = nx.density(subgraph)

        # 2. External Connectivity
        # Count edges going from community members to outside nodes
        external_edges = sum(1 for u in members for v in G.neighbors(u) if v not in members)
        max_external_edges = comm_size * (total_nodes - comm_size)
        connectivity_external = external_edges / max_external_edges if max_external_edges > 0 else 0

        # 3. Size Score
        size_score = 1 - abs(comm_size - avg_size) / avg_size
        size_score = max(0, size_score) # Keep it non-negative

        # 4. Peripherality
        avg_closeness = np.mean([closeness[node] for node in members])
        peripherality = 1 - avg_closeness

        # Calculate combined suspicion score
        suspicion_score = (w1 * density_internal) + \
                          (w2 * (1 - connectivity_external)) + \
                          (w3 * size_score) + \
                          (w4 * peripherality)
                          
        all_scores[comm_id] = suspicion_score

        # Check if this is the most suspicious cell so far
        if suspicion_score > max_suspicion:
            max_suspicion = suspicion_score
            best_community = {
                'id': comm_id,
                'members': list(members)
            }
            best_stats = {
                'internal_density': density_internal,
                'external_connectivity': connectivity_external,
                'size': comm_size,
                'peripherality': peripherality
            }
            
            # Generate reasoning
            reasoning = (
                f"Community {comm_id} is flagged as the most suspicious dormant cell.\n"
                f"- It maintains a dense internal structure (Density: {density_internal:.2f}), allowing secure internal chatter.\n"
                f"- It is highly compartmentalized and isolated from the rest of the network (External Connectivity: {connectivity_external:.4f}).\n"
                f"- Its size ({comm_size} members) is optimal for avoiding detection while remaining operational.\n"
                f"- It operates at the network periphery (Peripherality: {peripherality:.2f}), staying away from central, highly-monitored hubs."
            )
            best_reasoning = reasoning

    # Store results
    result = {
        'suspected_community_id': best_community['id'],
        'suspicion_score': max_suspicion,
        'members': best_community['members'],
        'characteristics': best_stats,
        'reasoning': best_reasoning,
        'all_scores': all_scores # Added to pass data to the plotting function
    }

    return result

def visualize_dormant_cell(G, partition, dormant_result):
    """
    Generate visualizations for dormant cell detection
    """
    fig = plt.figure(figsize=(20, 6))
    
    dormant_members = set(dormant_result['members'])
    pos = nx.spring_layout(G, k=0.15, seed=42)

    # Plot 1: Full network with dormant cell highlighted
    ax1 = fig.add_subplot(1, 3, 1)
    node_colors = ['red' if node in dormant_members else 'lightgray' for node in G.nodes()]
    node_sizes = [200 if node in dormant_members else 50 for node in G.nodes()]
    
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax1)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax1)
    
    # Label only the dormant cell nodes in the full graph
    labels = {n: n for n in dormant_members}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax1)
    
    ax1.set_title("1. Network Topology (Dormant Cell Highlighted)")
    ax1.axis('off')

    # Plot 2: Dormant cell subgraph (internal structure)
    ax2 = fig.add_subplot(1, 3, 2)
    subgraph = G.subgraph(dormant_members)
    sub_pos = nx.spring_layout(subgraph, seed=42)
    
    nx.draw(subgraph, sub_pos, node_color='red', with_labels=True, 
            node_size=600, font_color='white', font_weight='bold', ax=ax2)
    ax2.set_title(f"2. Internal Structure of Cell {dormant_result['suspected_community_id']}")

    # Plot 3: Suspicion score comparison bar chart
    ax3 = fig.add_subplot(1, 3, 3)
    scores = dormant_result['all_scores']
    comm_ids = list(scores.keys())
    score_vals = list(scores.values())
    
    # Highlight the highest bar
    colors = ['red' if comm == dormant_result['suspected_community_id'] else 'teal' for comm in comm_ids]
    
    sns.barplot(x=comm_ids, y=score_vals, palette=colors, ax=ax3)
    ax3.set_title("3. Suspicion Score Comparison")
    ax3.set_xlabel("Community ID")
    ax3.set_ylabel("Suspicion Score")
    ax3.axhline(y=dormant_result['suspicion_score'], color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    try:
        plt.savefig('figures/q6_dormant_cell_detection.png', dpi=300, bbox_inches='tight')
    except FileNotFoundError:
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/q6_dormant_cell_detection.png', dpi=300, bbox_inches='tight')
        
    plt.show()

if __name__ == "__main__":
    # Load network
    G = nx.les_miserables_graph()

    print("=" * 80)
    print("OPERATION: DORMANT CELL DETECTION")
    print("=" * 80)

    # Load best partition from Question 5
    # For testing, you can use Louvain:
    import community as community_louvain

    partition = community_louvain.best_partition(G, random_state=42)

    print(f"Analyzing {len(set(partition.values()))} detected communities...")

    # Execute dormant cell detection
    dormant = find_dormant_cell(G, partition)

    # Display results
    print("\n" + "=" * 80)
    print("DORMANT CELL DETECTED")
    print("=" * 80)
    print(f"Community ID: {dormant['suspected_community_id']}")
    print(f"Suspicion Score: {dormant['suspicion_score']:.3f}/1.000")
    print(f"\nMembers ({len(dormant['members'])}):")
    print(f"  {', '.join(dormant['members'][:10])}")
    if len(dormant['members']) > 10:
        print(f"  ... and {len(dormant['members']) - 10} more")

    print(f"\nCharacteristics:")
    for key, value in dormant['characteristics'].items():
        print(f"  {key}: {value:.3f}")

    print(f"\nAnalysis:")
    print(dormant['reasoning'])

    # Generate visualizations
    visualize_dormant_cell(G, partition, dormant)

    print("\n✓ Mission Complete")