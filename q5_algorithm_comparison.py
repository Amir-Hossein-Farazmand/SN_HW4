"""
Question 5: Operation "Rapid Detection" - Algorithm Comparison
Multi-algorithm deployment on Les Misérables network
"""

import random
import time
from collections import Counter, defaultdict

import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import os

def label_propagation(G, max_iter=100, seed=None):
    """
    *** IMPLEMENT FROM SCRATCH - NO LIBRARY ALLOWED ***
    """
    if seed is not None:
        random.seed(seed)

    # Step 1: Initialize labels (each node gets a unique label)
    labels = {node: i for i, node in enumerate(G.nodes())}
    nodes = list(G.nodes())
    
    num_iterations = 0
    # Step 2: Iterate until convergence
    for iteration in range(max_iter):
        num_iterations += 1
        random.shuffle(nodes)
        converged = True
        
        for node in nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue
                
            # Count labels of neighbors
            neighbor_labels = [labels[nbr] for nbr in neighbors]
            counts = Counter(neighbor_labels)
            
            # Find the maximum frequency
            max_freq = max(counts.values())
            
            # Find all labels that have this maximum frequency
            most_frequent_labels = [lbl for lbl, freq in counts.items() if freq == max_freq]
            
            # Choose randomly among ties
            chosen_label = random.choice(most_frequent_labels)
            
            # Update label
            if labels[node] != chosen_label:
                labels[node] = chosen_label
                converged = False
                
        if converged:
            break

    return labels, num_iterations

def calculate_modularity_from_labels(G, labels):
    """
    Calculate modularity given node->community mapping
    """
    # Convert labels dict to list of sets
    communities_dict = defaultdict(set)
    for node, comm_id in labels.items():
        communities_dict[comm_id].add(node)
    
    communities = list(communities_dict.values())
    
    # Calculate modularity using NetworkX's implementation of Newman's formula
    # We use weight='weight' because Les Miserables graph has weighted edges
    return nx.community.modularity(G, communities, weight='weight')

def algorithm_comparison(G, num_runs=5):
    """
    Deploy three community detection algorithms and compare performance
    """
    results = []

    print("Deploying Algorithm 1: Louvain...")
    louvain_metrics = {'time': [], 'mod': [], 'num_comm': [], 'largest': [], 'smallest': []}
    for run in range(num_runs):
        start_time = time.time()
        partition = community_louvain.best_partition(G, random_state=run)
        end_time = time.time()
        
        mod = community_louvain.modularity(partition, G)
        comm_sizes = list(Counter(partition.values()).values())
        
        louvain_metrics['time'].append(end_time - start_time)
        louvain_metrics['mod'].append(mod)
        louvain_metrics['num_comm'].append(len(comm_sizes))
        louvain_metrics['largest'].append(max(comm_sizes))
        louvain_metrics['smallest'].append(min(comm_sizes))

    results.append({
        'algorithm': 'Louvain',
        'avg_time': np.mean(louvain_metrics['time']),
        'std_time': np.std(louvain_metrics['time']),
        'avg_num_communities': np.mean(louvain_metrics['num_comm']),
        'avg_modularity': np.mean(louvain_metrics['mod']),
        'std_modularity': np.std(louvain_metrics['mod']),
        'largest_community_size': np.mean(louvain_metrics['largest']),
        'smallest_community_size': np.mean(louvain_metrics['smallest'])
    })

    print("Deploying Algorithm 2: Label Propagation...")
    lp_metrics = {'time': [], 'mod': [], 'num_comm': [], 'largest': [], 'smallest': []}
    for run in range(num_runs):
        start_time = time.time()
        labels, _ = label_propagation(G, seed=run)
        end_time = time.time()
        
        mod = calculate_modularity_from_labels(G, labels)
        comm_sizes = list(Counter(labels.values()).values())
        
        lp_metrics['time'].append(end_time - start_time)
        lp_metrics['mod'].append(mod)
        lp_metrics['num_comm'].append(len(comm_sizes))
        lp_metrics['largest'].append(max(comm_sizes))
        lp_metrics['smallest'].append(min(comm_sizes))

    results.append({
        'algorithm': 'Label Propagation',
        'avg_time': np.mean(lp_metrics['time']),
        'std_time': np.std(lp_metrics['time']),
        'avg_num_communities': np.mean(lp_metrics['num_comm']),
        'avg_modularity': np.mean(lp_metrics['mod']),
        'std_modularity': np.std(lp_metrics['mod']),
        'largest_community_size': np.mean(lp_metrics['largest']),
        'smallest_community_size': np.mean(lp_metrics['smallest'])
    })

    print("Deploying Algorithm 3: Greedy Modularity...")
    greedy_metrics = {'time': [], 'mod': [], 'num_comm': [], 'largest': [], 'smallest': []}
    for run in range(num_runs):
        start_time = time.time()
        # Greedy is deterministic, but we run it multiple times for fair time averaging
        communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
        end_time = time.time()
        
        mod = nx.community.modularity(G, communities, weight='weight')
        comm_sizes = [len(c) for c in communities]
        
        greedy_metrics['time'].append(end_time - start_time)
        greedy_metrics['mod'].append(mod)
        greedy_metrics['num_comm'].append(len(comm_sizes))
        greedy_metrics['largest'].append(max(comm_sizes))
        greedy_metrics['smallest'].append(min(comm_sizes))

    results.append({
        'algorithm': 'Greedy Modularity',
        'avg_time': np.mean(greedy_metrics['time']),
        'std_time': np.std(greedy_metrics['time']),
        'avg_num_communities': np.mean(greedy_metrics['num_comm']),
        'avg_modularity': np.mean(greedy_metrics['mod']),
        'std_modularity': np.std(greedy_metrics['mod']),
        'largest_community_size': np.mean(greedy_metrics['largest']),
        'smallest_community_size': np.mean(greedy_metrics['smallest'])
    })

    df = pd.DataFrame(results)
    return df

def deep_analysis(G, best_partition):
    """
    Perform deep intelligence analysis on best community partition
    """
    analysis = {
        'bridges': [],
        'central_per_community': {},
        'largest_community': {},
        'edge_distribution': {'intra': 0, 'inter': 0, 'ratio': 0.0}
    }

    # 1. Find bridge characters (connected to >= 3 communities)
    for node in G.nodes():
        neighbor_communities = set(best_partition[nbr] for nbr in G.neighbors(node))
        if len(neighbor_communities) >= 3:
            analysis['bridges'].append({
                'node': node,
                'degree': G.degree(node),
                'communities_connected': len(neighbor_communities)
            })
            
    # Sort bridges by degree
    analysis['bridges'] = sorted(analysis['bridges'], key=lambda x: x['degree'], reverse=True)

    # Group nodes by community
    communities_dict = defaultdict(list)
    for node, comm_id in best_partition.items():
        communities_dict[comm_id].append(node)

    # 2. Find most central in each community & 3. Analyze largest community
    max_size = 0
    degree_centrality = nx.degree_centrality(G)
    
    for comm_id, members in communities_dict.items():
        # Central node
        central_node = max(members, key=lambda n: degree_centrality[n])
        analysis['central_per_community'][comm_id] = {
            'node': central_node,
            'centrality': degree_centrality[central_node],
            'size': len(members)
        }
        
        # Largest community tracking
        if len(members) > max_size:
            max_size = len(members)
            analysis['largest_community'] = {
                'id': comm_id,
                'size': len(members),
                'members': members,
                'leader': central_node
            }

    # 4. Count intra vs inter-community edges
    intra_edges = 0
    inter_edges = 0
    for u, v in G.edges():
        if best_partition[u] == best_partition[v]:
            intra_edges += 1
        else:
            inter_edges += 1
            
    analysis['edge_distribution']['intra'] = intra_edges
    analysis['edge_distribution']['inter'] = inter_edges
    if inter_edges > 0:
        analysis['edge_distribution']['ratio'] = intra_edges / inter_edges

    return analysis

def visualize_communities(G, partition, analysis):
    """
    Generate intelligence report visualizations
    """
    fig = plt.figure(figsize=(22, 14))
    
    # Generate consistent colors for communities
    unique_comms = list(set(partition.values()))
    palette = sns.color_palette("husl", len(unique_comms))
    color_map = {comm_id: palette[i] for i, comm_id in enumerate(unique_comms)}
    node_colors = [color_map[partition[node]] for node in G.nodes()]
    
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

    # Plot 1: Network with community colors
    ax1 = fig.add_subplot(2, 3, 1)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax1)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, ax=ax1)
    
    # Label top 10 highest degree nodes
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:10]
    labels = {n: n for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax1)
    ax1.set_title("1. Detected Cells Network Topology")
    ax1.axis('off')

    # Plot 2: Community size distribution
    ax2 = fig.add_subplot(2, 3, 2)
    comm_sizes = [info['size'] for info in analysis['central_per_community'].values()]
    sns.histplot(comm_sizes, bins=len(unique_comms), kde=True, color='purple', ax=ax2)
    ax2.set_title("2. Cell Size Distribution")
    ax2.set_xlabel("Number of Agents in Cell")
    ax2.set_ylabel("Frequency")

    # Plot 3: Inter-community connection heatmap
    ax3 = fig.add_subplot(2, 3, 3)
    adj_matrix = np.zeros((len(unique_comms), len(unique_comms)))
    comm_to_idx = {comm: i for i, comm in enumerate(unique_comms)}
    
    for u, v in G.edges():
        c_u = comm_to_idx[partition[u]]
        c_v = comm_to_idx[partition[v]]
        adj_matrix[c_u][c_v] += 1
        if c_u != c_v:
            adj_matrix[c_v][c_u] += 1
            
    sns.heatmap(adj_matrix, annot=False, cmap='YlOrRd', ax=ax3)
    ax3.set_title("3. Inter-Cell Communication Volume")
    ax3.set_xlabel("Cell Index")
    ax3.set_ylabel("Cell Index")

    # Plot 4: Bridge nodes highlighted
    ax4 = fig.add_subplot(2, 3, 4)
    bridge_nodes = [b['node'] for b in analysis['bridges']]
    bridge_colors = ['red' if node in bridge_nodes else 'lightgray' for node in G.nodes()]
    bridge_sizes = [300 if node in bridge_nodes else 50 for node in G.nodes()]
    
    nx.draw_networkx_edges(G, pos, alpha=0.1, ax=ax4)
    nx.draw_networkx_nodes(G, pos, node_color=bridge_colors, node_size=bridge_sizes, ax=ax4)
    bridge_labels = {n: n for n in bridge_nodes}
    nx.draw_networkx_labels(G, pos, bridge_labels, font_size=8, font_weight='bold', ax=ax4)
    ax4.set_title("4. Bridge Agents (Connected to 3+ Cells)")
    ax4.axis('off')

    # Plot 5: Degree distribution
    ax5 = fig.add_subplot(2, 3, 5)
    degree_seq = [d for n, d in G.degree()]
    sns.kdeplot(degree_seq, fill=True, color='teal', ax=ax5)
    ax5.set_title("5. Network Degree Distribution")
    ax5.set_xlabel("Degree (Number of Connections)")
    ax5.set_ylabel("Density")

    # Plot 6: Statistics summary text box
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = (
        f"INTELLIGENCE SUMMARY\n\n"
        f"Total Cells Detected: {len(unique_comms)}\n"
        f"Total Bridge Agents: {len(bridge_nodes)}\n"
        f"Intra-cell Edges: {analysis['edge_distribution']['intra']}\n"
        f"Inter-cell Edges: {analysis['edge_distribution']['inter']}\n"
        f"Isolation Ratio: {analysis['edge_distribution']['ratio']:.2f} (Internal/External)\n\n"
        f"LARGEST CELL (ID: {analysis['largest_community']['id']})\n"
        f"- Size: {analysis['largest_community']['size']} agents\n"
        f"- Central Figure: {analysis['largest_community']['leader']}\n\n"
        f"TOP BRIDGE AGENTS\n"
    )
    
    for i, b in enumerate(analysis['bridges'][:5]):
        stats_text += f"- {b['node']} (Connects {b['communities_connected']} cells)\n"

    ax6.text(0.1, 0.9, stats_text, fontsize=12, family='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
    ax6.set_title("6. Deep Analysis Metrics")

    plt.tight_layout()
    try:
        plt.savefig('figures/q5_les_miserables_analysis.png', dpi=300, bbox_inches='tight')
    except FileNotFoundError:
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/q5_les_miserables_analysis.png', dpi=300, bbox_inches='tight')
        
    plt.show()

if __name__ == "__main__":
    # Load Les Misérables network
    G = nx.les_miserables_graph()

    print("=" * 80)
    print("OPERATION: RAPID DETECTION")
    print("=" * 80)
    print(f"Target Network: Les Misérables")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")
    print()

    # Execute multi-algorithm comparison
    comparison_df = algorithm_comparison(G, num_runs=5)

    # Display comparison table
    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON RESULTS")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    # Save results
    try:
        comparison_df.to_csv('results/q5_algorithm_comparison.csv', index=False)
    except FileNotFoundError:
        os.makedirs('results', exist_ok=True)
        comparison_df.to_csv('results/q5_algorithm_comparison.csv', index=False)

    # Identify best algorithm
    best_idx = comparison_df['avg_modularity'].idxmax()
    best_algo = comparison_df.iloc[best_idx]

    print("\n" + "=" * 80)
    print(f"BEST ALGORITHM: {best_algo['algorithm']}")
    print("=" * 80)
    print(f"Modularity: {best_algo['avg_modularity']:.4f} (±{best_algo['std_modularity']:.4f})")
    print(f"Time: {best_algo['avg_time']:.4f}s (±{best_algo['std_time']:.4f}s)")
    print(f"Communities: {best_algo['avg_num_communities']:.1f}")

    # Deep Analysis Execution
    print("\nExecuting deep intelligence analysis using best partition...")
    
    # Rerun the best algorithm (Louvain is almost certainly the winner here)
    if best_algo['algorithm'] == 'Louvain':
        best_partition = community_louvain.best_partition(G, random_state=42)
    elif best_algo['algorithm'] == 'Greedy Modularity':
        communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
        best_partition = {node: i for i, comm in enumerate(communities) for node in comm}
    else:
        best_partition, _ = label_propagation(G, seed=42)

    analysis = deep_analysis(G, best_partition)
    
    print("\nGenerating visual intelligence reports...")
    visualize_communities(G, best_partition, analysis)

    print("\n✓ Mission Complete")