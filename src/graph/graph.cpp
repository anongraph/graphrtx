#include "graph.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

struct MMapFile {
    int fd = -1;
    size_t size = 0;
    const char* data = nullptr;

    ~MMapFile() { close_map(); }

    void close_map() {
        if (data && data != MAP_FAILED) {
            ::munmap((void*)data, size);
            data = nullptr;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        size = 0;
    }

    bool open_map(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;

        struct stat st;
        if (::fstat(fd, &st) != 0) return false;

        size = (size_t)st.st_size;
        if (size == 0) return false;

        void* m = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) return false;

        data = (const char*)m;
        return true;
    }
};

static inline const char* skip_ws(const char* p, const char* end) {
    while (p < end) {
        unsigned char c = (unsigned char)*p;
        if (c != ' ' && c != '\t' && c != '\r') break;
        ++p;
    }
    return p;
}

static inline const char* skip_line(const char* p, const char* end) {
    while (p < end && *p != '\n') ++p;
    if (p < end && *p == '\n') ++p;
    return p;
}

static inline bool parse_u32(const char*& p, const char* end, uint32_t& out) {
    p = skip_ws(p, end);
    if (p >= end) return false;

    uint64_t v = 0;
    bool any = false;
    while (p < end) {
        unsigned char c = (unsigned char)*p;
        if (c < '0' || c > '9') break;
        any = true;
        v = v * 10 + (c - '0');
        if (v > std::numeric_limits<uint32_t>::max()) return false;
        ++p;
    }
    if (!any) return false;
    out = (uint32_t)v;
    return true;
}

static inline bool parse_u64(const char*& p, const char* end, uint64_t& out) {
    p = skip_ws(p, end);
    if (p >= end) return false;

    uint64_t v = 0;
    bool any = false;
    while (p < end) {
        unsigned char c = (unsigned char)*p;
        if (c < '0' || c > '9') break;
        any = true;
        v = v * 10 + (c - '0');
        ++p;
    }
    if (!any) return false;
    out = v;
    return true;
}

static inline bool parse_f32(const char*& p, const char* end, float& out) {
    p = skip_ws(p, end);
    if (p >= end) return false;

    const char* q = p;
    while (q < end) {
        unsigned char c = (unsigned char)*q;
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') break;
        ++q;
    }
    if (q == p) return false;

    char buf[64];
    size_t len = (size_t)(q - p);
    if (len >= sizeof(buf)) len = sizeof(buf) - 1;
    std::memcpy(buf, p, len);
    buf[len] = '\0';

    char* e = nullptr;
    out = std::strtof(buf, &e);
    if (e == buf) return false;

    p = q;
    return true;
}

static inline const char* align_to_next_line(const char* p, const char* end) {
    while (p < end && *p != '\n') ++p;
    if (p < end && *p == '\n') ++p;
    return p;
}

} 
int graph_rtx::load_mtx_graph(const std::string& filename, int num_threads)
{
    if (num_threads <= 0) num_threads = (int)std::max(1u, std::thread::hardware_concurrency());

    std::cout << "Loading Matrix Market graph from " << filename
              << " using " << num_threads << " threads...\n";

    auto t_start = std::chrono::high_resolution_clock::now();

   
    MMapFile mm;
    if (!mm.open_map(filename)) {
        std::cerr << "Error: Could not mmap graph file " << filename << "\n";
        std::exit(1);
    }

    const char* begin = mm.data;
    const char* end   = mm.data + mm.size;

    std::string line;
    bool is_symmetric = false;


    const char* p = begin;
    while (p < end) {
        if (*p == '%') {
            const char* ls = p;
            const char* le = p;
            while (le < end && *le != '\n') ++le;
            if (std::string_view(ls, (size_t)(le - ls)).find("symmetric") != std::string_view::npos) {
                is_symmetric = true;
            }
            p = skip_line(p, end);
            continue;
        }
        if (*p == '\n') { ++p; continue; }
        break;
    }

    uint32_t header_rows = 0, header_cols = 0;
    uint64_t header_edges = 0;

    {
        const char* q = p;
        if (!parse_u32(q, end, header_rows) || !parse_u32(q, end, header_cols) || !parse_u64(q, end, header_edges) ||
            header_rows == 0 || header_cols == 0) {

            const char* le = p;
            while (le < end && *le != '\n') ++le;
            std::string bad(p, le);
            std::cerr << "Error: Invalid MatrixMarket header line: '" << bad << "'\n";
            std::exit(1);
        }
        p = skip_line(p, end); 
    }

    const uint32_t n = header_rows;

    std::cout << "  [load] Header: " << header_rows << " x " << header_cols
              << ", " << header_edges << " edges.\n";
    std::cout << "  [load] Assuming 1-based indices 1 to " << n << ".\n";
    if (is_symmetric) {
        std::cout << "  [load] Detected symmetric format.\n";
    }

    const char* data_begin = p;

    std::vector<const char*> chunk_starts((size_t)num_threads + 1);
    chunk_starts[0] = data_begin;
    chunk_starts[(size_t)num_threads] = end;

    const size_t bytes = (size_t)(end - data_begin);
    for (int t = 1; t < num_threads; ++t) {
        const char* raw = data_begin + (bytes * (size_t)t) / (size_t)num_threads;
        chunk_starts[(size_t)t] = align_to_next_line(raw, end);
    }

    std::vector<std::vector<uint32_t>> deg_t((size_t)num_threads);
    for (int t = 0; t < num_threads; ++t) {
        deg_t[(size_t)t].assign((size_t)n, 0u);
    }

    std::atomic<uint64_t> skipped1{0};

    auto pass1_worker = [&](int tid) {
        const char* a = chunk_starts[(size_t)tid];
        const char* b = chunk_starts[(size_t)tid + 1];
        auto& deg = deg_t[(size_t)tid];

        const char* cur = a;
        while (cur < b) {
            if (*cur == '%') { cur = skip_line(cur, b); continue; }
            if (*cur == '\n') { ++cur; continue; }

            const char* line_end = cur;
            while (line_end < b && *line_end != '\n') ++line_end;

            uint32_t u = 0, v = 0;
            float w = 1.0f;
            const char* x = cur;

            if (!parse_u32(x, line_end, u) || !parse_u32(x, line_end, v)) {
                cur = (line_end < b ? line_end + 1 : line_end);
                continue;
            }
            (void)parse_f32(x, line_end, w); 

            if (u < 1 || v < 1 || u > n || v > n) {
                skipped1.fetch_add(1, std::memory_order_relaxed);
                cur = (line_end < b ? line_end + 1 : line_end);
                continue;
            }

            uint32_t uu = u - 1;
            uint32_t vv = v - 1;

            deg[(size_t)uu] += 1u;
            if (is_symmetric && uu != vv) deg[(size_t)vv] += 1u;

            cur = (line_end < b ? line_end + 1 : line_end);
        }
    };

    {
        std::vector<std::thread> threads;
        threads.reserve((size_t)num_threads);
        for (int t = 0; t < num_threads; ++t) threads.emplace_back(pass1_worker, t);
        for (auto& th : threads) th.join();
    }

    if (skipped1.load() != 0) {
        std::cout << "  [load] Warning: skipped " << skipped1.load()
                  << " edges with out-of-range indices.\n";
    }

    std::vector<uint32_t> deg((size_t)n, 0u);

    auto reduce_worker = [&](int tid) {
        const size_t start = ((size_t)n * (size_t)tid) / (size_t)num_threads;
        const size_t stop  = ((size_t)n * (size_t)(tid + 1)) / (size_t)num_threads;

        for (size_t i = start; i < stop; ++i) {
            uint64_t sum = 0;
            for (int t = 0; t < num_threads; ++t) sum += deg_t[(size_t)t][i];
            if (sum > std::numeric_limits<uint32_t>::max()) {
                std::cerr << "Error: degree overflow at vertex " << i << "\n";
                std::exit(1);
            }
            deg[i] = (uint32_t)sum;
        }
    };

    {
        std::vector<std::thread> threads;
        threads.reserve((size_t)num_threads);
        for (int t = 0; t < num_threads; ++t) threads.emplace_back(reduce_worker, t);
        for (auto& th : threads) th.join();
    }

    row_ptr_.clear();
    row_ptr_.resize((size_t)n + 1);
    row_ptr_[0] = 0;

    uint64_t total_edges_u64 = 0;
    for (uint32_t i = 0; i < n; ++i) {
        total_edges_u64 += (uint64_t)deg[(size_t)i];
        if (total_edges_u64 > (uint64_t)std::numeric_limits<uint32_t>::max()) {
            std::cerr << "Error: CSR offsets exceed uint32_t capacity (need uint64_t row_ptr_).\n";
            std::exit(1);
        }
        row_ptr_[(size_t)i + 1] = (uint32_t)total_edges_u64;
    }

    std::cout << "  [load] Read ~" << header_edges << " edges (header), CSR edges="
              << (uint64_t)row_ptr_.back() << (is_symmetric ? " (sym expanded)\n" : "\n");

    nbrs_ptr_.clear();
    wts_.clear();
    nbrs_ptr_.resize((size_t)row_ptr_.back());
    wts_.resize((size_t)row_ptr_.back());

    auto bases_worker = [&](int tid) {
        const size_t start = ((size_t)n * (size_t)tid) / (size_t)num_threads;
        const size_t stop  = ((size_t)n * (size_t)(tid + 1)) / (size_t)num_threads;

        for (size_t i = start; i < stop; ++i) {
            uint32_t base = row_ptr_[i];
            for (int t = 0; t < num_threads; ++t) {
                uint32_t cnt = deg_t[(size_t)t][i];
                deg_t[(size_t)t][i] = base; 
                base += cnt;
            }
        }
    };

    {
        std::vector<std::thread> threads;
        threads.reserve((size_t)num_threads);
        for (int t = 0; t < num_threads; ++t) threads.emplace_back(bases_worker, t);
        for (auto& th : threads) th.join();
    }

    std::atomic<uint64_t> skipped2{0};

    auto pass2_worker = [&](int tid) {
        const char* a = chunk_starts[(size_t)tid];
        const char* b = chunk_starts[(size_t)tid + 1];
        auto& cursor = deg_t[(size_t)tid]; 

        const char* cur = a;
        while (cur < b) {
            if (*cur == '%') { cur = skip_line(cur, b); continue; }
            if (*cur == '\n') { ++cur; continue; }

            const char* line_end = cur;
            while (line_end < b && *line_end != '\n') ++line_end;

            uint32_t u = 0, v = 0;
            float w = 1.0f;
            const char* x = cur;

            if (!parse_u32(x, line_end, u) || !parse_u32(x, line_end, v)) {
                cur = (line_end < b ? line_end + 1 : line_end);
                continue;
            }
            if (!(parse_f32(x, line_end, w))) w = 1.0f; 

            if (u < 1 || v < 1 || u > n || v > n) {
                skipped2.fetch_add(1, std::memory_order_relaxed);
                cur = (line_end < b ? line_end + 1 : line_end);
                continue;
            }

            uint32_t uu = u - 1;
            uint32_t vv = v - 1;

            {
                uint32_t idx = cursor[(size_t)uu]++;
                if (idx >= row_ptr_[(size_t)uu + 1]) {
                    std::cerr << "Error: CSR overflow fill (uu=" << uu << ")\n";
                    std::exit(1);
                }
                nbrs_ptr_[(size_t)idx] = vv;
                wts_[(size_t)idx] = w;
            }

            if (is_symmetric && uu != vv) {
                uint32_t idx = cursor[(size_t)vv]++;
                if (idx >= row_ptr_[(size_t)vv + 1]) {
                    std::cerr << "Error: CSR overflow fill (vv=" << vv << ")\n";
                    std::exit(1);
                }
                nbrs_ptr_[(size_t)idx] = uu;
                wts_[(size_t)idx] = w;
            }

            cur = (line_end < b ? line_end + 1 : line_end);
        }
    };

    {
        std::vector<std::thread> threads;
        threads.reserve((size_t)num_threads);
        for (int t = 0; t < num_threads; ++t) threads.emplace_back(pass2_worker, t);
        for (auto& th : threads) th.join();
    }

    if (skipped2.load() != 0) {
        std::cout << "  [load] Warning: skipped " << skipped2.load()
                  << " edges with out-of-range indices in pass2.\n";
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n--- Graph Loading Complete ---\n";
    std::cout << "Nodes: " << n << "\n";
    std::cout << "Edges: " << (uint64_t)row_ptr_.back() << "\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "Elapsed: " << elapsed << " s\n";
    std::cout << "-------------------------------\n\n";

    uasp_first_.resize(n);
    uasp_count_.resize(n);
    uasps_host_.clear();
    uasps_host_.reserve(static_cast<size_t>(2) * static_cast<size_t>(n));

    return static_cast<int>(n);
}

void graph_rtx::build_aabbs() {
    const size_t n_uasps = uasps_host_.size();
    const int num_threads = std::max(1u, std::thread::hardware_concurrency());

    aabbs6_.resize(n_uasps * 6);

    parallel_for(0, n_uasps, num_threads, [&](size_t i) {
        const auto& uasp = uasps_host_[i];

        const uint32_t num_vertices = static_cast<uint32_t>(row_ptr_.size() ? (row_ptr_.size() - 1) : 0);
        if (uasp.owner >= num_vertices) return;

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

    parallel_for(0, N, num_threads, [&](int u) {
        const uint32_t start = row_ptr_[u];
        const uint32_t deg   = row_ptr_[u + 1] - start;
        uasp_count_[u] = (deg == 0) ? 1u : (deg + MAX_SEG_LEN - 1) / MAX_SEG_LEN;
    });

    uasp_first_[0] = 0;
    for (int u = 1; u < N; ++u)
        uasp_first_[u] = uasp_first_[u - 1] + uasp_count_[u - 1];

    const uint32_t total = uasp_first_.back() + uasp_count_.back();
    uasps_host_.resize(total);

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

    const size_t offset_aabbs = aabbs6_.size();
    const size_t offset_mask  = aabb_mask_.size();
    aabbs6_.resize(offset_aabbs + total_floats);
    aabb_mask_.resize(offset_mask + total_mask);

    const unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());
    const uint32_t chunk = (count + num_threads - 1) / num_threads;

    auto worker = [&](uint32_t tid) {
        std::mt19937 rng(12345u + tid);
        std::uniform_real_distribution<float> jitter(-0.5f, 0.5f);

        const uint32_t begin = tid * chunk;
        const uint32_t end   = std::min(count, begin + chunk);

        float*   aabbs_ptr = aabbs6_.data() + (offset_aabbs + size_t(begin) * 6);
        uint8_t* mask_ptr  = aabb_mask_.data() + (offset_mask + size_t(begin));

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
    row_ptr_.reserve(static_cast<size_t>(n) + 1);

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
        total_edges += static_cast<size_t>(deg);
        for (int e = 0; e < deg; ++e) {
            uint32_t dst = static_cast<uint32_t>(rng() % static_cast<uint32_t>(n));
            if (dst == static_cast<uint32_t>(src)) dst = (dst + 1) % static_cast<uint32_t>(n);
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
              << bytes_nbrs   / (1024.0 * 1024.0) << " MB nbrs, "
              << bytes_wts    / (1024.0 * 1024.0) << " MB weights)\n";
    std::cout << "Graph generation took " << elapsed << " s\n";
}
