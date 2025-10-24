# GraphRTX

**GraphRTX** is a high-performance **RT-accelerated graph analytics framework** built on top of **CUDA** and **NVIDIA OptiX**.  
It enables acceleration of both traditional graph algorithms using **ray tracing hardware (RT cores)**.

GraphRTX provides:
- A **C++ core engine** with direct CUDA/OptiX integration.
- **Python bindings** for easy use with `.mtx` files or existing `networkx` graphs.

---

## Overview


GraphRTX loads standard graph datasets (`.mtx` Matrix Market files), builds GPU-optimized adjacency structures (UASP, triangles and AABBs), and executes a suite of common graph algorithms using either:
- pure **CUDA**, or
- **OptiX + CUDA hybrid** pipelines that exploit RT cores for traversal.
A GPU memory manager allows to analyze graphs larger than GPU memory efficiently.

---

## Supported Algorithms

| Algorithm | GPU (CUDA) | Hybrid (OptiX + CUDA) | Notes |
|------------|-------------|----------------------|-------|
| **BFS** (Breadth-First Search) | ✓ | ✓ | Parallel frontier expansion |
| **PR** (PageRank) | ✓ | ✓ | Damping factor configurable |
| **SSSP** (Single-Source Shortest Path) | ✓ | ✓ | Weighted relaxation |
| **BC** (Betweenness Centrality) | ✓ | ✓ | Approximate version supported |
| **TC** (Triangle Counting) | ✓ | ✓ | Geometric intersection-based variant |

All hybrid versions device workload between CUDA and OptiX to accelerate neighbor traversal and intersection operations.

---
## Example Python call with NetworkX

```python
import networkx as nx
from pygraph_rtx import Graph

G = nx.fast_gnp_random_graph(1000, 0.01)
grtx = Graph()
grtx.from_networkx(G)
grtx.prepare()
res = grtx.run_bfs(0)
print(res)
```

---

## Build Instructions

### Prerequisites

- **CUDA** ≥ 13.0  
- **OptiX SDK** ≥ 9.0  
- **CMake** ≥ 3.13  
- **GPU** with RT Cores (e.g. RTX 4070)

Tested on: RTX 4070, RTX 5090, RTX A6000 Pro

### Build

Set OPTIX_ROOT to OptiX SDK root.

```bash
https://github.com/anongraph/graphrtx.git
cd graphrtx
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```


### Evaluation Datasets

- roadUSA  
- livegraph  
- kron21  
- weibo  
- twitter10  
- orkut  
- sk-2005  
- friendster
