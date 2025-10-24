#pragma once
#include <optix.h>
#include <cstdint>

// -------- graph primitive (UASP = segment of a vertex's adjacency) ----------
struct UASP {
    uint32_t owner;      // vertex id
    uint32_t seg_idx;    // segment ordinal for this owner (0..k-1), 0 for 1-seg-per-vertex
    uint32_t start;      // offset into global nbrs[]
    uint32_t len;        // number of neighbors in this segment
};

enum QueryType : uint8_t {
    EXPAND = 0,
    JOIN = 1,
    PAGERANK = 2,
    SSSP = 3,
    BETWEENNESS = 4
};

// Job record
struct Job {
    QueryType qtype;
    uint32_t  src;        // vertex id
    uint32_t  partner;    // JOIN only
    uint32_t  primIndex;  // used by JOIN (segment index)
    uint32_t  depth;      // BFS depth
    uint32_t  max_depth;  // unused
    float     dist;       // optional
};

struct BloomLayout { uint32_t words_per_vertex{0}; };

struct PRWorkItem {
    uint32_t start;
    uint32_t len;
    float    contrib;
};

struct Params {
    // Scene
    OptixTraversableHandle tlas{};

    // Graph storage (CSR)
    const uint32_t* row_ptr{};   // N+1
    const uint32_t* nbrs{};      // M
    const float* weights{};   // M (edge weights for SSSP)

    // UASP segments
    const UASP* uasps{};     // P
    uint32_t        num_uasps{0};

    // Jobs
    const Job* jobs{};           // num_rays
    uint32_t   num_rays{0};

    const float* aabbs{};        // 6 floats per prim (min,max)
    uint32_t     grid_side{0};  

    uint32_t* job_counts{};     // num_rays
    uint64_t* tri_count{};

    // BFS / worklist
    uint32_t* next_frontier{};
    uint32_t* next_count{};
    uint32_t* visited_bitmap{};
    uint32_t* distances{};       // BFS distances (uint32_t)
    uint32_t  visited_words{0};
    uint32_t  num_vertices{0};
    uint32_t  next_capacity{0};
    uint32_t  num_rows{0};
    uint32_t  num_nbrs{0};
    const uint64_t* bloom_bits{};
    BloomLayout bloom{};
    const uint32_t* cluster_offsets{};
    uint32_t        num_clusters{0};

    // ---- PageRank buffers ----
    const float* pr_curr{};       // size N
    float* pr_next{};       // size N
    float        damping{0.85f};
    float        invN{0.0f};      // 1/N

    const float* inv_outdeg{};   
    PRWorkItem* pr_items{};      
    uint32_t* pr_item_count{};
    uint32_t     pr_item_cap{0};  

    // ---- SSSP buffers ----
    float* sssp_distances{}; // shortest-path distances (float)

    // ---- Betweenness Centrality buffers ----
    float* bc_values{};   
    uint32_t* sigma{};       
    float* delta{};       
    uint32_t     bc_phase{0};   

    // ---- Instance → global-primitive mapping for multi-GAS TLAS ----
    const uint32_t* instance_prim_bases{nullptr}; 
    uint32_t        num_instances{0};

    const uint32_t* uasp_first{nullptr};  // per-vertex: first segment index
    const uint32_t* uasp_count{nullptr};  // per-vertex: number of segments
    uint32_t        num_aabbs{0};

    const uint8_t* aabb_mask;
};