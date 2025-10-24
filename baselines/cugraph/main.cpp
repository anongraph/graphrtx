#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/edge_property.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>

#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <variant>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <optional>
#include <chrono>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// GPU timer utility
// -----------------------------------------------------------------------------
template <typename F>
float time_gpu_ms(raft::handle_t const& handle, F&& fn)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  auto stream = handle.get_stream();
  cudaEventRecord(start, stream);
  fn();
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms;
}

// -----------------------------------------------------------------------------
// Robust Matrix Market graph loader (with remapping + symmetrization)
// -----------------------------------------------------------------------------
int load_mtx_graph(
  const std::string& filename,
  std::vector<uint32_t>& row_ptr,
  std::vector<uint32_t>& nbrs,
  std::vector<float>& wts)
{
  std::cout << "Loading Matrix Market graph from " << filename << " (robustly with remapping)...\n";
  auto t_start = std::chrono::high_resolution_clock::now();

  std::ifstream infile(filename);
  if (!infile.is_open()) {
      std::cerr << "Error: Could not open graph file " << filename << "\n";
      exit(1);
  }

  std::string line;
  bool is_symmetric = false;

  // --- Step 1: Read banner and detect symmetric format ---
  while (std::getline(infile, line)) {
      if (line.empty() || line[0] != '%') break;
      if (line.find("MatrixMarket") != std::string::npos) {
          if (line.find("symmetric") != std::string::npos) {
              is_symmetric = true;
              std::cout << "  [load] Detected symmetric graph format.\n";
          }
          while (std::getline(infile, line)) {
              if (line.empty() || line[0] == '%') continue;
              break;
          }
          break;
      }
  }

  // --- Step 2: Dimensions ---
  uint32_t header_rows = 0, header_cols = 0, header_edges = 0;
  {
    std::stringstream ss(line);
    ss >> header_rows >> header_cols >> header_edges;
  }
  std::cout << "  [load] MTX Header: " << header_rows << " x " << header_cols
            << ", " << header_edges << " edges.\n";

  // --- Step 3: Read edges and remap IDs ---
  std::vector<std::tuple<uint32_t, uint32_t, float>> temp_edges;
  temp_edges.reserve(header_edges);
  std::unordered_map<uint32_t, uint32_t> id_map;
  uint32_t next_id = 0;

  while (std::getline(infile, line)) {
      if (line.empty() || line[0] == '%') continue;
      std::stringstream edge_ss(line);
      uint32_t u_orig, v_orig;
      float w = 1.0f;
      edge_ss >> u_orig >> v_orig >> w;

      if (id_map.find(u_orig) == id_map.end()) id_map[u_orig] = next_id++;
      if (id_map.find(v_orig) == id_map.end()) id_map[v_orig] = next_id++;

      temp_edges.emplace_back(u_orig, v_orig, w);
  }
  infile.close();

  const uint32_t n = next_id;
  std::cout << "  [load] Found " << n << " unique nodes.\n";

  // --- Step 4: Build adjacency list ---
  std::vector<std::vector<std::pair<uint32_t, float>>> adj(n);
  for (const auto& e : temp_edges) {
      uint32_t u_new = id_map.at(std::get<0>(e));
      uint32_t v_new = id_map.at(std::get<1>(e));
      float w = std::get<2>(e);
      adj[u_new].push_back({v_new, w});
      if (is_symmetric && u_new != v_new) adj[v_new].push_back({u_new, w});
  }

  // --- Step 5: Convert to CSR ---
  std::cout << "  [load] Converting to CSR format...\n";
  row_ptr.clear(); nbrs.clear(); wts.clear();
  row_ptr.reserve(n + 1);
  row_ptr.push_back(0);

  for (uint32_t i = 0; i < n; ++i) {
      std::sort(adj[i].begin(), adj[i].end());
      adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
      for (auto& e : adj[i]) {
          nbrs.push_back(e.first);
          wts.push_back(e.second);
      }
      row_ptr.push_back(static_cast<uint32_t>(nbrs.size()));
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();

  std::cout << "\n--- Graph Loading Complete ---\n";
  std::cout << "Final Nodes: " << n << "\n";
  std::cout << "Final Edges: " << nbrs.size() << "\n";
  std::cout << "Loading and remapping took " << elapsed << " s\n";
  std::cout << "----------------------------\n\n";

  return n;
}

// -----------------------------------------------------------------------------
// Main benchmark
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  raft::handle_t handle;
  auto stream = handle.get_stream();

  using vertex_t = int32_t;
  using edge_t   = int32_t;
  using weight_t = float;

  bool directed = false;

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <graph.mtx>\n";
    return 1;
  }

  std::string filename = argv[1];

  std::vector<uint32_t> row_ptr, nbrs;
  std::vector<float> wts;
  int num_vertices = load_mtx_graph(filename, row_ptr, nbrs, wts);
  size_t num_edges = nbrs.size();

  // --- Convert CSR → edge list (for cuGraph) ---
  std::vector<vertex_t> h_src, h_dst;
  std::vector<weight_t> h_wgt;
  h_src.reserve(num_edges);
  h_dst.reserve(num_edges);
  h_wgt.reserve(num_edges);

  for (int u = 0; u < num_vertices; ++u) {
      for (uint32_t j = row_ptr[u]; j < row_ptr[u + 1]; ++j) {
          h_src.push_back(u);
          h_dst.push_back(nbrs[j]);
          h_wgt.push_back(wts[j]);
      }
  }

  std::cout << "CSR converted: " << h_src.size() << " edges.\n";

  // --- Device copy ---
  rmm::device_uvector<vertex_t> d_src(num_edges, stream);
  rmm::device_uvector<vertex_t> d_dst(num_edges, stream);
  rmm::device_uvector<weight_t> d_wgt(num_edges, stream);
  raft::update_device(d_src.data(), h_src.data(), num_edges, stream);
  raft::update_device(d_dst.data(), h_dst.data(), num_edges, stream);
  raft::update_device(d_wgt.data(), h_wgt.data(), num_edges, stream);

  // --- Create CSR graph ---
  std::vector<rmm::device_uvector<vertex_t>> srcs;
  std::vector<rmm::device_uvector<vertex_t>> dsts;
  srcs.emplace_back(std::move(d_src));
  dsts.emplace_back(std::move(d_dst));

  std::vector<std::variant<std::monostate,
                           rmm::device_uvector<float>,
                           rmm::device_uvector<double>,
                           rmm::device_uvector<int>,
                           rmm::device_uvector<long>,
                           rmm::device_uvector<unsigned long>>> edge_props;
  edge_props.emplace_back(std::move(d_wgt));

  std::vector<std::vector<std::variant<std::monostate,
                                       rmm::device_uvector<float>,
                                       rmm::device_uvector<double>,
                                       rmm::device_uvector<int>,
                                       rmm::device_uvector<long>,
                                       rmm::device_uvector<unsigned long>>>> edge_prop_groups;
  edge_prop_groups.emplace_back(std::move(edge_props));

  auto [graph, edge_weight_props, renumber_map] =
      cugraph::create_graph_from_edgelist<vertex_t, edge_t, false, false>(
          handle, std::nullopt, std::move(srcs), std::move(dsts),
          std::move(edge_prop_groups),
          cugraph::graph_properties_t{/*is_symmetric=*/!directed, /*is_multigraph=*/false},
          true, std::nullopt, std::nullopt, false);

  auto view = graph.view();
  std::cout << "Graph created. Vertices: " << num_vertices
            << "  Edges: " << num_edges << "\n";

  auto& prop_var = edge_weight_props[0];
  auto& edge_prop = std::get<cugraph::edge_property_t<edge_t, weight_t>>(prop_var);
  auto edge_w_view = edge_prop.view();

  vertex_t src_vertex = 12;

  // --- SSSP ---
  {
    rmm::device_uvector<weight_t> distances(num_vertices, stream);
    rmm::device_uvector<vertex_t> predecessors(num_vertices, stream);
    float ms = time_gpu_ms(handle, [&]() {
      cugraph::sssp(handle, view, edge_w_view,
                    distances.data(), predecessors.data(),
                    src_vertex,
                    std::numeric_limits<weight_t>::max(),
                    false);
    });
    std::cout << "SSSP done in " << ms << " ms.\n";
  }

  // --- BFS ---
  {
    rmm::device_uvector<vertex_t> distances(num_vertices, stream);
    rmm::device_uvector<vertex_t> predecessors(num_vertices, stream);
    rmm::device_uvector<vertex_t> d_start(1, stream);
    vertex_t h_start = src_vertex;
    raft::update_device(d_start.data(), &h_start, 1, stream);
    float ms = time_gpu_ms(handle, [&]() {
      cugraph::bfs(handle, view, distances.data(), predecessors.data(),
                   d_start.data(), 1, false,
                   std::numeric_limits<vertex_t>::max(), false);
    });
    std::cout << "BFS done in " << ms << " ms.\n";
  }

  // --- PageRank (CSC graph required) ---
  {
    std::cout << "Building CSC graph for PageRank...\n";
    rmm::device_uvector<vertex_t> d_src2(num_edges, stream);
    rmm::device_uvector<vertex_t> d_dst2(num_edges, stream);
    rmm::device_uvector<weight_t> d_wgt2(num_edges, stream);
    raft::update_device(d_src2.data(), h_src.data(), num_edges, stream);
    raft::update_device(d_dst2.data(), h_dst.data(), num_edges, stream);
    raft::update_device(d_wgt2.data(), h_wgt.data(), num_edges, stream);

    std::vector<rmm::device_uvector<vertex_t>> srcs2;
    std::vector<rmm::device_uvector<vertex_t>> dsts2;
    srcs2.emplace_back(std::move(d_src2));
    dsts2.emplace_back(std::move(d_dst2));

    std::vector<std::variant<std::monostate,
                             rmm::device_uvector<float>,
                             rmm::device_uvector<double>,
                             rmm::device_uvector<int>,
                             rmm::device_uvector<long>,
                             rmm::device_uvector<unsigned long>>> edge_props2;
    edge_props2.emplace_back(std::move(d_wgt2));

    std::vector<std::vector<std::variant<std::monostate,
                                         rmm::device_uvector<float>,
                                         rmm::device_uvector<double>,
                                         rmm::device_uvector<int>,
                                         rmm::device_uvector<long>,
                                         rmm::device_uvector<unsigned long>>>> edge_prop_groups2;
    edge_prop_groups2.emplace_back(std::move(edge_props2));

    auto [graph_csc, edge_weight_props_csc, renumber_map2] =
        cugraph::create_graph_from_edgelist<vertex_t, edge_t, true, false>(
            handle, std::nullopt, std::move(srcs2), std::move(dsts2),
            std::move(edge_prop_groups2),
            cugraph::graph_properties_t{/*is_symmetric=*/!directed, /*is_multigraph=*/false},
            false, std::nullopt, std::nullopt, false);

    auto view_csc = graph_csc.view();
    auto& prop_var2 = edge_weight_props_csc[0];
    auto& edge_prop2 = std::get<cugraph::edge_property_t<edge_t, weight_t>>(prop_var2);
    auto edge_w_view2 = edge_prop2.view();

    float alpha = 0.85f, eps = 1e-6f;
    float ms = time_gpu_ms(handle, [&]() {
      auto [pr_values, meta] = cugraph::pagerank<vertex_t, edge_t, weight_t, float, false>(
          handle, view_csc, edge_w_view2,
          std::nullopt, std::nullopt, std::nullopt,
          alpha, eps, 100, false);
    });
    std::cout << "PageRank done in " << ms << " ms.\n";
  }

  // --- Triangle Count ---
  {
    auto n_local_vertices = static_cast<size_t>(view.number_of_vertices());
    rmm::device_uvector<edge_t> counts(n_local_vertices, stream);
    float ms = time_gpu_ms(handle, [&]() {
      auto span = raft::device_span<edge_t>(counts.data(), counts.size());
      using span_t = raft::span<const edge_t, true, std::numeric_limits<size_t>::max()>;
      std::optional<span_t> no_edges = std::nullopt;
      cugraph::triangle_count(handle, view, no_edges, span, false);
    });
    std::cout << "Triangle Count done in " << ms << " ms.\n";
  }

  // --- Approximate Betweenness Centrality ---
  {
    auto nv = static_cast<size_t>(view.number_of_vertices());
    int k = 16;
    std::vector<vertex_t> h_sources(k);
    std::mt19937 rng(42);
    std::uniform_int_distribution<vertex_t> dist(0, nv - 1);
    for (int i = 0; i < k; ++i) h_sources[i] = dist(rng);
    rmm::device_uvector<vertex_t> d_sources(k, stream);
    raft::update_device(d_sources.data(), h_sources.data(), k, stream);

    float ms = time_gpu_ms(handle, [&]() {
      auto bc_scores = cugraph::betweenness_centrality<vertex_t, edge_t, weight_t, false>(
          handle, view, std::nullopt,
          std::make_optional(raft::device_span<const vertex_t>(d_sources.data(), k)),
          true, false, false);
    });
    std::cout << "Approximate Betweenness Centrality done in " << ms << " ms.\n";
  }

  std::cout << "=== All algorithms complete ===\n";
  return 0;
}

