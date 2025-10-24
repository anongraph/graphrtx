#include "graph.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cfloat>
#include <random>

int graph_rtx::load_mtx_graph(
  const std::string& filename,
  int num_threads)
{
  std::cout << "Loading Matrix Market graph from " << filename
            << " using " << num_threads << " threads...\n";

  auto t_start = std::chrono::high_resolution_clock::now();

  std::ifstream infile(filename);
  if (!infile.is_open()) {
      std::cerr << "Error: Could not open graph file " << filename << "\n";
      exit(1);
  }

  std::string line;
  bool is_symmetric = false;

  while (std::getline(infile, line)) {
      if (line.empty() || line[0] != '%') break;
      if (line.find("symmetric") != std::string::npos) {
          is_symmetric = true;
          std::cout << "  [load] Detected symmetric format.\n";
      }
  }

  uint32_t header_rows = 0, header_cols = 0, header_edges = 0;
  {
      std::stringstream ss(line);
      ss >> header_rows >> header_cols >> header_edges;
  }

  const uint32_t n = header_rows; 

  std::cout << "  [load] Header: " << header_rows << " x " << header_cols
            << ", " << header_edges << " edges.\n";
  std::cout << "  [load] Assuming 1-based indices 1 to " << n << ".\n"; // Informative debug

  std::vector<std::string> lines;
  lines.reserve(header_edges * 1.1);
  std::string buf;
  while (std::getline(infile, buf)) {
      if (!buf.empty() && buf[0] != '%') lines.push_back(std::move(buf));
  }
  infile.close();

  const size_t M = lines.size();
  std::cout << "  [load] Read " << M << " edges.\n";

  std::vector<std::vector<std::tuple<uint32_t, uint32_t, float>>> local_edges(num_threads);

  auto parse_chunk = [&](int tid, size_t start, size_t end) {
      auto& edges = local_edges[tid];
      edges.reserve(end - start);
      for (size_t i = start; i < end; ++i) {
          std::stringstream ss(lines[i]);
          uint32_t u, v; float w = 1.0f;
          ss >> u >> v >> w;
          edges.emplace_back(u, v, w);
      }
  };

  std::vector<std::thread> threads;
  size_t chunk = (M + num_threads - 1) / num_threads;
  for (int t = 0; t < num_threads; ++t) {
      size_t start = t * chunk;
      size_t end   = std::min(M, start + chunk);
      if (start >= M) break;
      threads.emplace_back(parse_chunk, t, start, end);
  }
  for (auto& th : threads) th.join();

  std::cout << "  [load] Found " << n << " unique nodes.\n";

  std::vector<std::vector<std::pair<uint32_t, float>>> adj(n);
  std::mutex adj_mutex; 

  auto remap_chunk = [&](int tid) {
      const auto& edges = local_edges[tid];
      std::vector<std::pair<uint32_t, std::pair<uint32_t, float>>> local;
      local.reserve(edges.size() * (is_symmetric ? 2 : 1));

      for (auto& [u, v, w] : edges) {
          
          uint32_t u_new = u - 1; 
          uint32_t v_new = v - 1; 
          
          if (u == 0 || v == 0 || u > n || v > n) continue;

          local.emplace_back(u_new, std::make_pair(v_new, w));
          if (is_symmetric && u_new != v_new)
              local.emplace_back(v_new, std::make_pair(u_new, w));
      }

      {
          std::lock_guard<std::mutex> lock(adj_mutex);
          for (auto& e : local)
              adj[e.first].push_back(e.second);
      }
  };

  threads.clear();
  for (int t = 0; t < num_threads; ++t)
      threads.emplace_back(remap_chunk, t);
  for (auto& th : threads) th.join();

  std::cout << "  [load] Sorting & deduplicating adjacency lists...\n";

  auto sort_chunk = [&](int tid, size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
          auto& vec = adj[i];
          std::sort(vec.begin(), vec.end(),
                    [](auto& a, auto& b){ return a.first < b.first; });
          vec.erase(std::unique(vec.begin(), vec.end(),
                                [](auto& a, auto& b){ return a.first == b.first; }),
                    vec.end());
      }
  };

  threads.clear();
  chunk = (n + num_threads - 1) / num_threads;
  for (int t = 0; t < num_threads; ++t) {
      size_t start = t * chunk;
      size_t end   = std::min<size_t>(n, start + chunk);
      if (start >= n) break;
      threads.emplace_back(sort_chunk, t, start, end);
  }
  for (auto& th : threads) th.join();

  std::cout << "  [load] Converting to CSR format...\n";
  row_ptr_.clear(); nbrs_ptr_.clear(); wts_.clear();
  row_ptr_.reserve(n + 1);
  row_ptr_.push_back(0);

  size_t total_edges = 0;
  for (uint32_t i = 0; i < n; ++i) {
      for (auto& e : adj[i]) {
        nbrs_ptr_.push_back(e.first);
          wts_.push_back(e.second);
      }
      total_edges += adj[i].size();
      row_ptr_.push_back(static_cast<uint32_t>(nbrs_ptr_.size()));
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t_end - t_start).count();

  std::cout << "\n--- Graph Loading Complete ---\n";
  std::cout << "Nodes: " << n << "\n";
  std::cout << "Edges: " << total_edges << "\n";
  std::cout << "Threads: " << num_threads << "\n";
  std::cout << "Elapsed: " << elapsed << " s\n";
  std::cout << "-------------------------------\n\n";

  uasp_first_.resize(n);
  uasp_count_.resize(n);
  uasps_host_.reserve(2*n);
  return n;
}

void graph_rtx::build_aabbs() {
    const size_t n = uasps_host_.size();
    const int num_threads = std::max(1u, std::thread::hardware_concurrency());

    aabbs6_.resize(n * 6);

    parallel_for(0, n, num_threads, [&](size_t i) {
        const auto& uasp = uasps_host_[i];
        const uint32_t u_base = row_ptr_[uasp.owner];
        const float x0 = float(uasp.start - u_base);
        const float x1 = x0 + float(uasp.len);
        const float y0 = float(uasp.owner);
        const float y1 = y0 + 1.0f;

        size_t base = i * 6;
        aabbs6_[base + 0] = x0;
        aabbs6_[base + 1] = y0;
        aabbs6_[base + 2] = 0.0f;
        aabbs6_[base + 3] = x1;
        aabbs6_[base + 4] = y1;
        aabbs6_[base + 5] = 1.0f;
    });
}

void graph_rtx::build_uasps(uint32_t MAX_SEG_LEN) {
    const int N = static_cast<int>(row_ptr_.size() - 1);
    const int num_threads = std::max(1u, std::thread::hardware_concurrency());

    uasp_first_.resize(N);
    uasp_count_.resize(N);

    // === Pass 1: Count segments per vertex (parallel) ===
    parallel_for(0, N, num_threads, [&](int u) {
        const uint32_t start = row_ptr_[u];
        const uint32_t deg   = row_ptr_[u + 1] - start;
        uasp_count_[u] = (deg == 0) ? 1u : (deg + MAX_SEG_LEN - 1) / MAX_SEG_LEN;
    });

    // === Pass 2: Sequential prefix sum (exclusive scan) ===
    uasp_first_[0] = 0;
    for (int u = 1; u < N; ++u)
        uasp_first_[u] = uasp_first_[u - 1] + uasp_count_[u - 1];

    const uint32_t total = uasp_first_.back() + uasp_count_.back();
    uasps_host_.resize(total);

    // === Pass 3: Fill segments (parallel) ===
    parallel_for(0, N, num_threads, [&](int u) {
        const uint32_t start = row_ptr_[u];
        const uint32_t deg   = row_ptr_[u + 1] - start;
        const uint32_t num_segs = uasp_count_[u];
        const uint32_t base = uasp_first_[u];

        if (deg == 0) {
            uasps_host_[base] = { (uint32_t)u, 0u, start, 0u };
        } else {
            for (uint32_t s = 0; s < num_segs; ++s) {
                const uint32_t seg_start = start + s * MAX_SEG_LEN;
                const uint32_t seg_len   = std::min<uint32_t>(MAX_SEG_LEN, deg - s * MAX_SEG_LEN);
                uasps_host_[base + s] = { (uint32_t)u, s, seg_start, seg_len };
            }
        }
    });    
}

static inline void compute_scene_bounds(const std::vector<float>& aabbs6,
                                        float bmin[3], float bmax[3]) {
    bmin[0] = bmin[1] = bmin[2] =  FLT_MAX;
    bmax[0] = bmax[1] = bmax[2] = -FLT_MAX;
    for (size_t i = 0; i + 5 < aabbs6.size(); i += 6) {
        float x0 = aabbs6[i+0], y0 = aabbs6[i+1], z0 = aabbs6[i+2];
        float x1 = aabbs6[i+3], y1 = aabbs6[i+4], z1 = aabbs6[i+5];
        bmin[0] = std::min(bmin[0], std::min(x0, x1));
        bmin[1] = std::min(bmin[1], std::min(y0, y1));
        bmin[2] = std::min(bmin[2], std::min(z0, z1));
        bmax[0] = std::max(bmax[0], std::max(x0, x1));
        bmax[1] = std::max(bmax[1], std::max(y0, y1));
        bmax[2] = std::max(bmax[2], std::max(z0, z1));
    }
}

void graph_rtx::append_dummy_aabbs_tagged(uint32_t count, float far_multiplier, float tiny_extent) {
  const size_t old_count = aabbs6_.size() / 6;
  aabb_mask_.resize(old_count, 0u);  

  if (count == 0) return;

  // Compute bounds
  float bmin[3], bmax[3];
  if (!aabbs6_.empty()) {
      compute_scene_bounds(aabbs6_, bmin, bmax);
  } else {
      bmin[0]=bmin[1]=bmin[2]=0.f;
      bmax[0]=bmax[1]=bmax[2]=1.f;
  }

  const float dx = (bmax[0] - bmin[0]);
  const float dy = (bmax[1] - bmin[1]);
  const float dz = (bmax[2] - bmin[2]);

  const float baseX  = bmin[0] - std::max(1000.0f, dx * far_multiplier);
  const float baseY  = bmax[1] + std::max(100.0f, dy * 0.1f);
  const float baseZ0 = bmin[2] - std::max(10.0f, dz * 10.0f);
  const float baseZ1 = baseZ0 + 1.0f;

  const size_t total_floats = size_t(count) * 6;
  const size_t total_mask   = size_t(count);

  // Reserve and pre-size to avoid lock contention
  const size_t offset_aabbs = aabbs6_.size();
  const size_t offset_mask  = aabb_mask_.size();
  aabbs6_.resize(offset_aabbs + total_floats);
  aabb_mask_.resize(offset_mask + total_mask);

  // Parallel region
  const unsigned num_threads = std::thread::hardware_concurrency();
  const uint32_t chunk = (count + num_threads - 1) / num_threads;

  auto worker = [&](uint32_t tid) {
      std::mt19937 rng(12345u + tid);  
      std::uniform_real_distribution<float> jitter(-0.5f, 0.5f);

      const uint32_t begin = tid * chunk;
      const uint32_t end   = std::min(count, begin + chunk);

      float* aabbs_ptr = aabbs6_.data() + (offset_aabbs + size_t(begin) * 6);
      uint8_t* mask_ptr = aabb_mask_.data() + (offset_mask + size_t(begin));

      for (uint32_t i = begin; i < end; ++i) {
          const float jx = jitter(rng);
          const float jy = jitter(rng);
          const float jz = jitter(rng);

          const float x0 = baseX - (i * (tiny_extent + 2.0f)) + jx;
          const float y0 = baseY + jy;
          const float z0 = baseZ0 + jz;
          const float x1 = x0 + tiny_extent;
          const float y1 = y0 + tiny_extent;
          const float z1 = baseZ1 + tiny_extent;

          aabbs_ptr[0] = x0;
          aabbs_ptr[1] = y0;
          aabbs_ptr[2] = z0;
          aabbs_ptr[3] = x1;
          aabbs_ptr[4] = y1;
          aabbs_ptr[5] = z1;
          aabbs_ptr += 6;

          *mask_ptr++ = 1u;
      }
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (unsigned t = 0; t < num_threads; ++t)
      threads.emplace_back(worker, t);
  for (auto& th : threads)
      th.join();    
}

double graph_rtx::uasp_total_size() {
    constexpr double bytes_per_uasp = sizeof(UASP);
    constexpr double bytes_per_MB = 1024.0 * 1024.0;
    return (uasps_host_.size() * bytes_per_uasp) / bytes_per_MB;
}

double graph_rtx::aabbs6_total_size() {
    constexpr double bytes_per_float = sizeof(float);
    constexpr double bytes_per_MB = 1024.0 * 1024.0;
    return (aabbs6_.size() * bytes_per_float) / bytes_per_MB;
}

void graph_rtx::generate_random_graph(int n, float p) {
    std::cout << "Generating random graph...\n";
    auto t_start = std::chrono::high_resolution_clock::now();
    
    row_ptr_.clear();
    nbrs_ptr_.clear();
    wts_.clear();
    row_ptr_.reserve(n + 1);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> weight_dist(0.1f, 1.0f);
    
    row_ptr_.push_back(0);
    
    std::binomial_distribution<int> deg_dist(n - 1, p);
    size_t total_edges = 0;
    
    for (int src = 0; src < n; ++src) {
        int deg = deg_dist(rng);
        if (deg <= 0) {
            row_ptr_.push_back(static_cast<uint32_t>(nbrs_ptr_.size()));
            continue;
        }
        total_edges += deg;
        for (int e = 0; e < deg; ++e) {
            uint32_t dst = rng() % n;
            if (dst == (uint32_t)src) dst = (dst + 1) % n;
            nbrs_ptr_.push_back(dst);
            wts_.push_back(weight_dist(rng));
        }
        row_ptr_.push_back(static_cast<uint32_t>(nbrs_ptr_.size()));
        if (src % 10000 == 0 && src > 0)
            std::cout << "  [gen] " << src << "/" << n << " nodes processed...\n";
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    
    size_t bytes_rowptr = row_ptr_.size() * sizeof(uint32_t);
    size_t bytes_nbrs   = nbrs_ptr_.size() * sizeof(uint32_t);
    size_t bytes_wts    = wts_.size() * sizeof(float);
    size_t total_bytes  = bytes_rowptr + bytes_nbrs + bytes_wts;
    double total_MB     = total_bytes / (1024.0 * 1024.0);
    
    std::cout << "Nodes: " << n << "\n";
    std::cout << "Edges: " << total_edges << "\n";
    std::cout << "Graph size: " << total_MB << " MB ("
                << bytes_rowptr / (1024.0 * 1024.0) << " MB row_ptr, "
                << bytes_nbrs / (1024.0 * 1024.0)   << " MB nbrs, "
                << bytes_wts / (1024.0 * 1024.0)    << " MB weights)\n";
    std::cout << "Graph generation took " << elapsed << " s\n";
}
