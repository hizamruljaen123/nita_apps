# Accident-Prone Area Clustering using k-Medoids and Dijkstra Mapping

## Introduction
This workflow aims to identify **accident-prone areas** and optimize route planning:  
- **k-Medoids**: Clusters locations based on accident frequency, severity, or traffic density.  
- **Dijkstra Mapping**: Calculates shortest paths on road networks to understand connectivity and risk exposure.

### Use Case
- Identify hotspots for traffic accidents.  
- Plan safer routes for emergency response or public transportation.  
- Support city planners in traffic management.

---

## How It Works
1. **Collect accident data**: GPS coordinates, severity, timestamp, traffic volume.  
2. **Preprocess data**: Normalize coordinates, aggregate per area.  
3. **Apply k-Medoids clustering**:
   - Each cluster represents a group of accident-prone locations.  
   - Medoids are actual accident locations representing cluster centers.  
4. **Map shortest paths** using **Dijkstra’s algorithm**:
   - Represent roads as a weighted graph (distance or risk).  
   - Compute shortest paths between medoids or between key locations.  
5. **Visualization**: Plot clusters and shortest paths for analysis.

---

## Python Implementation (Simplified Example)
```python
import numpy as np
import networkx as nx
from pyclustering.cluster.kmedoids import kmedoids
import matplotlib.pyplot as plt

# Example accident data (latitude, longitude)
accidents = np.array([
    [1, 2],
    [2, 3],
    [3, 1],
    [8, 9],
    [9, 8],
    [8, 8]
])

# Initial medoid indices (choose k=2 clusters)
initial_medoids = [0, 3]

# Apply k-Medoids clustering
kmedoids_instance = kmedoids(accidents, initial_medoids)
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()

print("Clusters:", clusters)
print("Medoids:", medoids)

# Build road network as graph (example)
G = nx.Graph()
edges = [
    (0, 1, 1), (1, 2, 1.5), (2, 3, 2), (3, 4, 1), (4, 5, 1.2), (0, 5, 3)
]
G.add_weighted_edges_from(edges)

# Compute shortest path between two medoids using Dijkstra
source = medoids[0]
target = medoids[1]
path = nx.dijkstra_path(G, source=source, target=target, weight='weight')
distance = nx.dijkstra_path_length(G, source=source, target=target, weight='weight')

print("Shortest path between medoids:", path)
print("Distance:", distance)

# Visualization
colors = ['red', 'blue']
for idx, cluster in enumerate(clusters):
    plt.scatter(accidents[cluster,0], accidents[cluster,1], c=colors[idx])
plt.scatter(accidents[medoids,0], accidents[medoids,1], c='green', marker='x', s=100, label='Medoids')
plt.title("Accident Clusters and Medoids")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.show()
