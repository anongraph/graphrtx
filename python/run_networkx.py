import networkx as nx
from pygraph_rtx import Graph

G = nx.fast_gnp_random_graph(1000, 0.01)
grtx = Graph()
grtx.from_networkx(G)
grtx.prepare()
res = grtx.run_bfs(0)
print(res)
