# PathSnap Device 3: Task 1 — Road Topology and Directed Graph Construction

## A. Introduction

In distributed vehicle trajectory tracking and map-matching, noisy GPS observations deviate from physical road networks due to urban signal obstruction, multipath reflections, and sensor noise.

The primary responsibility of **Device 3 (Task 1)** is to construct an exact, topologically sound, and geometrically accurate directed graph representation of the road network from raw geographic spatial data. This graph forms the foundational state space for downstream map-matching, Hidden Markov Model (HMM) emission/transition calculations, and shortest-path Viterbi route decoding.

---

## B. Dataset Structure

Each trip in the PathSnap dataset is defined by two fundamental road network files:

1. **Nodes File (`.nodes`)**:
   - Tab/whitespace-separated without headers.
   - Column 1: Longitude ($\lambda$, degrees East, $[-180.0, 180.0]$).
   - Column 2: Latitude ($\phi$, degrees North, $[-90.0, 90.0]$).
   - **Node ID**: Implicit 0-indexed row position in the file (`0, 1, 2, ..., N-1`).

2. **Arcs File (`.arcs`)**:
   - Tab/whitespace-separated without headers.
   - Column 1: `from_node_id` (non-negative integer).
   - Column 2: `to_node_id` (non-negative integer).
   - **Arc ID**: Implicit 0-indexed row position in the file (`0, 1, 2, ..., M-1`).
   - Represents a directed road segment connecting `from_node_id` $\to$ `to_node_id`.

---

## C. Data Loading (`load_nodes` & `load_arcs`)

Implemented in `graph/graph_builder.py`:

- **`load_nodes(nodes_path)`**:
  - Reads `.nodes` using Pandas with `sep=r"\s+"` and `header=None`.
  - Verifies exactly 2 columns, finite values, and valid geographic ranges ($-180 \le \lambda \le 180$, $-90 \le \phi \le 90$).
  - Assigns explicit `node_id = np.arange(len(df), dtype=np.int64)` matching row order.
  - Returns a clean DataFrame: `['node_id', 'longitude', 'latitude']` (`int64`, `float64`, `float64`).

- **`load_arcs(arcs_path)`**:
  - Reads `.arcs` using Pandas with `sep=r"\s+"` and `header=None`.
  - Verifies exactly 2 columns, non-negative integer IDs, and absence of missing values.
  - Assigns explicit `arc_id = np.arange(len(df), dtype=np.int64)` matching row order.
  - Preserves row order and parallel edges without de-duplicating.
  - Returns a clean DataFrame: `['arc_id', 'from_node_id', 'to_node_id']` (`int64`, `int64`, `int64`).

---

## D. Data Validation (`validate_road_data`)

The function `validate_road_data(nodes_df, arcs_df, strict=False)` verifies integrity:
1. **Duplicate ID Checks**: Ensures 0 duplicates in `node_id` and `arc_id`.
2. **Referential Integrity**: Verifies that every `from_node_id` and `to_node_id` in `arcs_df` exists in `nodes_df`.
3. **Self-Loop Detection**: Identifies edges where $u = v$.
4. **Endpoint Multiplicity**: Distinguishes **duplicate endpoint groups** (distinct $(u, v)$ pairs appearing $>1$ times) from **extra parallel arcs** ($\sum (\text{count} - 1)$).
5. **Bidirectional Symmetry**: Identifies reverse-direction connections where both $(u, v)$ and $(v, u)$ exist.
6. **Isolated Nodes**: Identifies nodes with $\text{in-degree} = 0$ and $\text{out-degree} = 0$.
7. **Strict Enforcement**: When `strict=True`, raises `RoadDataValidationError` immediately upon detecting invalid references or malformed IDs.

---

## E. Distance Calculation (`haversine_distance`)

Because longitude and latitude are angular coordinates on a non-Euclidean sphere, ordinary Euclidean distance ($\sqrt{\Delta \lambda^2 + \Delta \phi^2}$) in degrees produces severe anisotropic distortion (at $50^\circ\text{ N}$, $1^\circ \text{ lon} \approx 71.5 \text{ km}$, while $1^\circ \text{ lat} \approx 111.2 \text{ km}$).

We calculate the physical metric distance using the spherical **Haversine formula**:

$$\Delta \phi = \text{lat}_2 - \text{lat}_1, \quad \Delta \lambda = \text{lon}_2 - \text{lon}_1 \quad (\text{in radians})$$

$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\text{lat}_1) \cdot \cos(\text{lat}_2) \cdot \sin^2\left(\frac{\Delta \lambda}{2}\right)$$

$$a_{\text{clamped}} = \max(0.0, \min(1.0, a))$$

$$d = 2 \cdot R \cdot \arcsin\left(\sqrt{a_{\text{clamped}}}\right)$$

Where $R = 6,371,000.0 \text{ meters}$ (Earth's mean spherical radius).

Clamping $a$ to $[0.0, 1.0]$ prevents floating-point inaccuracies from causing domain errors in $\sqrt{a}$ and $\arcsin$.

---

## F. Graph Construction (`build_road_graph`)

Implemented as `build_road_graph(nodes_path, arcs_path)`:
1. Loads and strictly validates nodes and arcs.
2. Initializes an empty `nx.MultiDiGraph()`.
3. Adds nodes with attributes:
   ```python
   road_graph.add_node(node_id, node_id=int(node_id), longitude=float(lon), latitude=float(lat))
   ```
4. Adds directed edges with `arc_id` as the edge key:
   ```python
   road_graph.add_edge(u, v, key=arc_id, arc_id=arc_id, length_m=dist, from_node_id=u, to_node_id=v)
   ```
5. **Why `MultiDiGraph` is required**: In real road networks, parallel road segments frequently exist between the same pair of nodes (e.g., dual carriageways, service roads). A standard `nx.DiGraph` would overwrite earlier edges with the same $(u, v)$, corrupting `arc_id` indexing. `nx.MultiDiGraph` maintains distinct edge instances via `key=arc_id`.

---

## G. Arc Lookup Dictionary

The construction returns `arc_lookup: Dict[int, Tuple[int, int]]` mapping:
$$\text{arc\_id} \longrightarrow (u, v)$$

This allows $O(1)$ constant-time resolution of any road segment's endpoint nodes during downstream route decoding and HMM emission probability projection.

---

## H. Graph Statistics (`get_graph_statistics`)

The function `get_graph_statistics(road_graph)` calculates:
- `total_nodes`: Number of vertices $|V|$.
- `total_directed_arcs`: Number of directed edges $|E|$.
- `isolated_nodes`: Number of vertices with degree 0.
- `weakly_connected_components` (WCC): Components in the underlying undirected infrastructure.
- `strongly_connected_components` (SCC): Subgraphs where mutual directed reachability holds.
- `duplicate_endpoint_groups`: Number of distinct directed $(u, v)$ pairs with $>1$ edge.
- `extra_parallel_arcs`: Total count of parallel edges beyond the first.
- `self_loops`: Number of edges where $u = v$.
- `total_directed_length_m`: Sum of all directed edge lengths in meters.
- `min_arc_length_m`, `max_arc_length_m`, `avg_arc_length_m`: Metric distribution of road segment lengths.

---

## I. Geographic Visualization (`visualize_road_graph`)

Implemented in `visualization/road_network_plot.py`:
- **Strict Geographic Coordinate Layout**: Uses true longitude ($x$) and latitude ($y$). No artificial spring layout or force-directed displacement.
- **Mercator Aspect Ratio Correction**: Scaled by $1 / \cos(\text{mid\_latitude})$ so horizontal and vertical metric scales match.
- **High-Performance `LineCollection`**: Consolidates 160,000+ line segments into a single GPU/C-level draw call, rendering large graphs in under 5 seconds.
- **Sampled Direction Arrows**: Uses `ax.quiver` on a sampled subset (200–500 edges) to clearly display traffic directionality without cluttering the map.
- **Local Bounding Box Support**: Allows rendering local urban intersections with high magnification.

---

## J. Actual Experimental Observations

> [!NOTE]
> The figures below represent empirical measurements from the two inspected benchmark trips. They do not constitute an aggregate calculation across all 90 trips.

### Benchmark Results Table:

| Metric | Training Trip `00000001` | Student Testing Trip `00000000` |
| :--- | :--- | :--- |
| **Geographic Region** | Darmstadt / Frankfurt, Germany | Bashkortostan, Russia |
| **Coordinate Extent** | Lon: `[8.5705, 8.6479]`, Lat: `[49.8441, 50.2119]` | Lon: `[54.2485, 54.9635]`, Lat: `[56.0806, 56.2546]` |
| **Total Nodes** | **88,809** | **8,542** |
| **Total Directed Arcs** | **165,790** | **18,196** |
| **Graph Construction Time** | **4.69 seconds** | **0.57 seconds** |
| **Arc Lookup Size** | **165,790** ($100\%$ preserved) | **18,196** ($100\%$ preserved) |
| **Isolated Nodes** | **0** | **0** |
| **Weakly Connected Components (WCC)**| **175** | **100** |
| **Strongly Connected Components (SCC)**| **1,942** | **102** |
| **Self-Loops** | **0** | **0** |
| **Duplicate Endpoint Groups** | **292** | **6** |
| **Extra Parallel Arcs** | **292** | **6** |
| **Bidirectional Reverse Connections**| **67,730 pairs** | **8,994 pairs** |
| **Total Directed Road Length** | **6,569,833.07 m** ($6,569.83 \text{ km}$) | **1,759,315.60 m** ($1,759.32 \text{ km}$) |
| **Min Arc Length** | **0.00 m** (8 zero-length multi-layer edges) | **0.62 m** |
| **Max Arc Length** | **1,106.46 m** | **7,007.16 m** |
| **Average Arc Length** | **39.63 m** | **96.69 m** |

### Generated Visualizations in `results/`:
1. `results/training_00000001_road_network.png` (952 KB): Regional overview of Darmstadt network.
2. `results/training_00000001_local_subgraph.png` (1,365 KB): Zoomed urban grid detail showing intersections and arrows.
3. `results/student_test_00000000_road_network.png` (661 KB): Rural regional network with longer highway links.

---

## K. Installation and Execution

### 1. Environment Setup (Windows PowerShell)
```powershell
# Create local virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Running Unit Tests
```powershell
# Run the complete test suite (41 tests)
.\.venv\Scripts\python.exe -m pytest -v
```

### 3. Example Execution Script
```powershell
.\.venv\Scripts\python.exe -c "
from pathlib import Path
from graph.graph_builder import build_road_graph
p = Path('Student folder/Student folder/student_test/00000000/00000000')
G, lookup, stats = build_road_graph(p.with_suffix('.nodes'), p.with_suffix('.arcs'))
print(f'Constructed graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.')
"
```

---

## L. Known Characteristics & Limitations

1. **Zero-Length Arcs**:
   - In Trip `00000001`, exactly 8 edges connect distinct node IDs that share identical geographic coordinates down to 6 decimal places.
   - These are physical transition nodes (bridges, tunnels, barrier breaks) present in raw road extracts.
   - They have length $0.0 \text{ m}$ and must be supported without crashing division or square-root routines.
2. **Parallel Connections**:
   - Multiple directed arcs exist between identical $(u, v)$ pairs (e.g., 292 parallel arcs in Trip 00000001). They must always be addressed by `(u, v, arc_id)`.
3. **Directed Asymmetry**:
   - Roads are directed. Never convert the graph to an undirected graph, as this would permit illegal wrong-way route matching on one-way streets.
4. **Endpoint Distance Approximation**:
   - Arc length is computed as the great-circle distance between start node $u$ and end node $v$. If an intermediate physical road curves between two nodes, the straight great-circle chord underestimates true road curveture slightly unless intermediate shape points exist.

---

## M. Device 4 Handoff Interface

Device 4 can consume the road graph directly through `build_road_graph()` without modifying any internal graph construction logic:

```python
from pathlib import Path
from typing import Dict, Tuple, Any
import networkx as nx
from graph.graph_builder import build_road_graph

def run_device_4_pipeline(trip_nodes_path: Path, trip_arcs_path: Path):
    # 1. Build road graph and lookup
    road_graph: nx.MultiDiGraph
    arc_lookup: Dict[int, Tuple[int, int]]
    stats: Dict[str, Any]
    
    road_graph, arc_lookup, stats = build_road_graph(trip_nodes_path, trip_arcs_path)
    
    # 2. Accessing node coordinates:
    node_u = 0
    lon_u = road_graph.nodes[node_u]["longitude"]
    lat_u = road_graph.nodes[node_u]["latitude"]
    
    # 3. Accessing edge attributes by arc_id:
    arc_id = 42
    u, v = arc_lookup[arc_id]
    edge_data = road_graph[u][v][arc_id]
    length_meters = edge_data["length_m"]
    
    # 4. Successors and outgoing transitions (for HMM transition matrix):
    outgoing_arcs = []
    for neighbor in road_graph.successors(v):
        for candidate_arc_id in road_graph[v][neighbor]:
            outgoing_arcs.append(candidate_arc_id)
            
    return road_graph, arc_lookup
```
