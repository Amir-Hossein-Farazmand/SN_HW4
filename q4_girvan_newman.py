"""
Question 4: Operation "Chain Breaker" - Girvan-Newman Algorithm
Community Detection Assignment - Cold War Intelligence Analysis
"""

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

def girvan_newman_analysis(G, target_communities=2):
    """
    Execute Operation Chain Breaker using Girvan-Newman algorithm
    """
    results = {
        'removed_edges': [],
        'modularity_at_each_step': [],
        'num_components': [],
        'final_communities': None,
        'critical_edge': None,
        'max_modularity': -10.0,
        'optimal_step': 0
    }

    # Create a working copy of the graph
    G_copy = G.copy()
    initial_components = len(list(nx.connected_components(G_copy)))
    step = 0

    # Continue until all edges are removed to find the true max modularity
    while G_copy.number_of_edges() > 0:
        # 1. Calculate edge betweenness
        eb = nx.edge_betweenness_centrality(G_copy)
        
        # 2. Find the edge with the highest betweenness
        max_edge = max(eb, key=eb.get)
        
        # Remove the edge
        G_copy.remove_edge(*max_edge)
        step += 1
        
        # 3. Check connected components
        comps = list(nx.connected_components(G_copy))
        num_components = len(comps)
        
        # 4. Calculate modularity (based on the original graph G)
        mod = calculate_modularity(G, comps)
        
        # Store step results
        results['removed_edges'].append((max_edge[0], max_edge[1], eb[max_edge]))
        results['num_components'].append(num_components)
        results['modularity_at_each_step'].append(mod)
        
        # Identify the critical edge (first edge to increase component count)
        if results['critical_edge'] is None and num_components > initial_components:
            results['critical_edge'] = max_edge
            
        # Capture the final communities when target is reached
        if num_components == target_communities and results['final_communities'] is None:
            results['final_communities'] = [set(c) for c in comps]
            
        # Track maximum modularity
        if mod > results['max_modularity']:
            results['max_modularity'] = mod
            results['optimal_step'] = step

    # Fallback in case the graph was already disconnected
    if results['final_communities'] is None:
        results['final_communities'] = [set(c) for c in nx.connected_components(G)]

    return results

def calculate_modularity(G, communities):
    """
    Calculate Newman's modularity Q for a given partition
    """
    m = G.number_of_edges()
    if m == 0:
        return 0.0
        
    Q = 0.0
    degrees = dict(G.degree())
    
    # Q = (1/2m) * sum_{ij} [A_ij - (k_i * k_j)/(2m)] * delta(c_i, c_j)
    # delta is 1 only when nodes are in the same community, so we just iterate over communities
    for community in communities:
        for i in community:
            for j in community:
                A_ij = 1 if G.has_edge(i, j) else 0
                expected_edges = (degrees[i] * degrees[j]) / (2 * m)
                Q += (A_ij - expected_edges)
                
    return Q / (2 * m)

def visualize_results(G, results, true_labels):
    """
    Generate intelligence report visualizations
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Modularity progression
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(range(1, len(results['modularity_at_each_step']) + 1), results['modularity_at_each_step'], color='purple', marker='o', markersize=3)
    ax1.set_title("1. Modularity vs Edges Removed")
    ax1.set_xlabel("Number of Edges Removed")
    ax1.set_ylabel("Modularity (Q)")
    ax1.axvline(x=results['optimal_step'], color='red', linestyle='--', label='Max Modularity')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Plot 2: Number of components
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(range(1, len(results['num_components']) + 1), results['num_components'], color='teal')
    ax2.set_title("2. Components vs Edges Removed")
    ax2.set_xlabel("Number of Edges Removed")
    ax2.set_ylabel("Number of Components")
    ax2.grid(True, linestyle=':', alpha=0.7)

    # Fixed layout for consistent node positioning
    pos = nx.spring_layout(G, seed=42)

    # Plot 3: Ground truth network
    ax3 = fig.add_subplot(2, 3, 3)
    labels_unique = list(set(true_labels.values()))
    color_map_true = ['#1f77b4' if true_labels[n] == labels_unique[0] else '#ff7f0e' for n in G.nodes()]
    nx.draw(G, pos, node_color=color_map_true, with_labels=True, ax=ax3, node_size=400, font_color='white', font_weight='bold')
    ax3.set_title("3. Ground Truth Network Factions")

    # Plot 4: Detected communities
    ax4 = fig.add_subplot(2, 3, 4)
    color_map_pred = []
    if results['final_communities'] and len(results['final_communities']) >= 2:
        c0 = results['final_communities'][0]
        for n in G.nodes():
            color_map_pred.append('#1f77b4' if n in c0 else '#ff7f0e')
    else:
        color_map_pred = 'gray'
        
    nx.draw(G, pos, node_color=color_map_pred, with_labels=True, ax=ax4, node_size=400, font_color='white', font_weight='bold')
    ax4.set_title("4. Detected Communities (Operation Target)")

    # Plot 5: Confusion matrix
    ax5 = fig.add_subplot(2, 3, 5)
    if results['final_communities'] and len(results['final_communities']) >= 2:
        c0, c1 = results['final_communities'][0], results['final_communities'][1]
        l0, l1 = labels_unique[0], labels_unique[1]
        
        # Try both alignments to find the best match
        correct_align_1 = sum(1 for n in c0 if true_labels[n] == l0) + sum(1 for n in c1 if true_labels[n] == l1)
        correct_align_2 = sum(1 for n in c0 if true_labels[n] == l1) + sum(1 for n in c1 if true_labels[n] == l0)
        
        pred_dict = {}
        if correct_align_1 > correct_align_2:
            for n in c0: pred_dict[n] = l0
            for n in c1: pred_dict[n] = l1
        else:
            for n in c0: pred_dict[n] = l1
            for n in c1: pred_dict[n] = l0
            
        y_true = [true_labels[n] for n in G.nodes()]
        y_pred = [pred_dict.get(n, l0) for n in G.nodes()]
        
        cm = confusion_matrix(y_true, y_pred, labels=labels_unique)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels_unique, yticklabels=labels_unique, cmap='Blues', ax=ax5, cbar=False)
        ax5.set_title("5. Classification Confusion Matrix")
        ax5.set_xlabel("Detected Labels")
        ax5.set_ylabel("True Intelligence Labels")

    plt.tight_layout()
    # Using try-except to handle folder path issues if the "figures" folder doesn't exist
    try:
        plt.savefig('figures/q4_girvan_newman_analysis.png', dpi=300, bbox_inches='tight')
    except FileNotFoundError:
        import os
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/q4_girvan_newman_analysis.png', dpi=300, bbox_inches='tight')
        
    plt.show()

def calculate_accuracy(detected_communities, true_labels):
    """
    Calculate classification accuracy compared to ground truth
    """
    if not detected_communities or len(detected_communities) < 2:
        return 0.0
        
    c0 = detected_communities[0]
    c1 = detected_communities[1]
    
    labels_unique = list(set(true_labels.values()))
    l0, l1 = labels_unique[0], labels_unique[1]
    
    # Alignment 1: c0->l0, c1->l1
    correct1 = sum(1 for n in c0 if true_labels[n] == l0) + sum(1 for n in c1 if true_labels.get(n) == l1)
    
    # Alignment 2: c0->l1, c1->l0
    correct2 = sum(1 for n in c0 if true_labels[n] == l1) + sum(1 for n in c1 if true_labels.get(n) == l0)
    
    return max(correct1, correct2) / len(true_labels)

if __name__ == "__main__":
    # Load Zachary's Karate Club network
    G = nx.karate_club_graph()

    # Extract ground truth labels
    true_labels = {}
    for node in G.nodes():
        true_labels[node] = G.nodes[node]['club']

    print("=" * 60)
    print("OPERATION: CHAIN BREAKER")
    print("=" * 60)
    print(f"Target Network: Zachary's Karate Club")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print()

    # Execute Girvan-Newman algorithm
    print("Executing Girvan-Newman algorithm...")
    results = girvan_newman_analysis(G, target_communities=2)

    # Display results
    print("\n" + "=" * 60)
    print("OPERATION RESULTS")
    print("=" * 60)
    print(f"Total edges removed: {len(results['removed_edges'])}")
    print(f"Maximum Modularity: {results['max_modularity']:.4f}")
    print(f"Optimal step: {results['optimal_step']}")
    print(f"Critical Edge: {results['critical_edge']}")
    print(f"Final community sizes: {[len(c) for c in results['final_communities']]}")

    # Calculate accuracy
    accuracy = calculate_accuracy(results['final_communities'], true_labels)
    print(f"Accuracy vs Ground Truth: {accuracy:.2%}")

    # Generate visualizations
    print("\nGenerating intelligence report visualizations...")
    visualize_results(G, results, true_labels)

    print("\n✓ Mission Complete")