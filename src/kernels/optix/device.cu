#include <optix_device.h>
#include <math_constants.h>
#include "shared.h"
#include "rt_utils.cuh"

extern "C" __global__ void __raygen__graph()
{
    uint32_t jobIdx = optixGetLaunchIndex().x;
    if (jobIdx >= params.num_rays) return;

    const Job j = params.jobs[jobIdx];

    if (j.qtype == EXPAND || j.qtype == PAGERANK || j.qtype == SSSP || j.qtype == BETWEENNESS) {
        const float3 ro = make_float3(0.5f, (float)j.src + 0.5f, 0.5f);
        const float3 rd = make_float3(1.0f, 0.0f, 0.0f);
        optixTrace(params.tlas, ro, rd,
                   0.0f, 1e16f, 0.0f,
                   0xFF,
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                   0, 0, 0,
                   jobIdx);
        return;
    }

    if (j.qtype == JOIN) {
        const UASP segU = params.uasps[j.primIndex];
        if (segU.len == 0) {  
            if (params.job_counts) params.job_counts[jobIdx] = 0u;
            return;
        }

        // Compute ray parameters
        const uint32_t base = params.row_ptr[segU.owner];
        const float x0 = (float)(segU.start - base);
        const float x1 = x0 + (float)segU.len;
        const float3 ro = make_float3(0.5f * (x0 + x1), (float)segU.owner + 0.5f, 0.5f);
        const float3 rd = make_float3(1.0f, 0.0f, 0.0f);

        if (x1 > x0) {
            optixTrace(params.tlas, ro, rd,
                       0.0f, 1.0f, 0.0f,
                       0xFF,
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                       0, 0, 0,
                       jobIdx);
        }

        const uint32_t* __restrict__ seg = params.nbrs + segU.start;
        const uint32_t local =
            count_common_gt_partner(j.partner, seg, segU.len, params.row_ptr, params.nbrs);

        if (params.job_counts)
            params.job_counts[jobIdx] = local;

        return;
    }
}


extern "C" __global__ void __miss__noop() {}

extern "C" __global__ void __intersection__uasp()
{
    const uint32_t primID_global = global_prim_index();
    const UASP seg = params.uasps[primID_global];
    const unsigned int jobIdx = optixGetPayload_0();
    const Job job = params.jobs[jobIdx];
    if (seg.owner != job.src) return;
    optixReportIntersection(0.0f, 0);
}

extern "C" __global__ void __anyhit__uasp()
{
    const unsigned int jobIdx = optixGetPayload_0();
    const Job j = params.jobs[jobIdx];
    if (j.qtype == JOIN) {
        optixIgnoreIntersection();
        return;
    }

    const uint32_t primID_global = global_prim_index();
    const UASP seg = params.uasps[primID_global];
    const uint32_t start = seg.start;
    const uint32_t end   = start + seg.len;

    // --- BFS Frontier Expansion ---
    if (j.qtype == EXPAND) {
        const uint32_t u = j.src;
        const uint32_t du = j.depth;
    
        // Load segment (neighbors of u)
        const UASP seg = params.uasps[global_prim_index()];
        if (seg.owner != u) { optixIgnoreIntersection(); return; }
    
        const uint32_t start = seg.start;
        const uint32_t end   = start + seg.len;
    
        for (uint32_t off = start; off < end; ++off) {
            const uint32_t v = __ldg(&params.nbrs[off]);
            const uint32_t word = v >> 5;
            const uint32_t mask = 1u << (v & 31);
            const uint32_t old  = atomicOr(&params.visited_bitmap[word], mask);
    
            if ((old & mask) == 0u) {
                // first visit
                params.distances[v] = du + 1;
    
                uint32_t idx = atomicAdd(params.next_count, 1u);
                if (idx < params.next_capacity)
                    params.next_frontier[idx] = v;
                else
                    atomicSub(params.next_count, 1u); 
            }
        }
    
        optixIgnoreIntersection();
        return;
    }
    
    // --- PageRank Scatter ---
    if (j.qtype == PAGERANK) {
        const uint32_t u = j.src;
        const uint32_t deg_u = params.row_ptr[u + 1] - params.row_ptr[u];
        if (deg_u > 0) {
            const float contrib = params.damping * params.pr_curr[u] / (float)deg_u;
            for (uint32_t off = start; off < end; ++off) {
                const uint32_t v = params.nbrs[off];
                atomicAdd(&params.pr_next[v], contrib);
            }
        }
        optixIgnoreIntersection();
        return;
    }

    // --- SSSP Relaxation ---
    if (j.qtype == SSSP) {
        const float du = params.sssp_distances[j.src];
        if (isinf(du)) {
            optixIgnoreIntersection();
            return;
        }

        for (uint32_t off = start; off < end; ++off) {
            const uint32_t v = params.nbrs[off];
            const float w = params.weights[off];
            const float alt = du + w;
            const float old = atomicMinFloat(&params.sssp_distances[v], alt);
            if (alt < old) {
                const uint32_t idx = atomicAdd(params.next_count, 1u);
                if (idx < params.next_capacity)
                    params.next_frontier[idx] = v;
                else
                    atomicSub(params.next_count, 1u);
            }
        }
        optixIgnoreIntersection();
        return;
    }

    if (j.qtype == BETWEENNESS) {
        const uint32_t node = j.src;
        const uint32_t dist_node = params.distances[node];
        const uint32_t sigma_node = params.sigma[node];

        if (dist_node == 0xFFFFFFFFu) { optixIgnoreIntersection(); return; }

        // Forward
        if (params.bc_phase == 0u) {
            const uint32_t du = dist_node;
            const uint32_t su = sigma_node;

            for (uint32_t off = start; off < end; ++off) {
                const uint32_t v = params.nbrs[off];
                const uint32_t old_dist = atomicCAS(&params.distances[v], 0xFFFFFFFFu, du + 1u);

                if (old_dist == 0xFFFFFFFFu) {
                    uint32_t idx = atomicAdd(params.next_count, 1u);
                    if (idx < params.next_capacity)
                        params.next_frontier[idx] = v;
                    else
                        atomicSub(params.next_count, 1u);
                }

                if ((old_dist == 0xFFFFFFFFu || old_dist == du + 1u) && su > 0u)
                    atomicAdd(&params.sigma[v], su);
            }
            optixIgnoreIntersection();
            return;
        }

        // Backward
        else {
            // w is the node further from source (successor), u is the neighbor (predecessor)
            const uint32_t w  = node;
            const uint32_t dw = dist_node;
            const uint32_t sw = sigma_node;

            if (sw == 0u || dw == 0u) { optixIgnoreIntersection(); return; }

            for (uint32_t off = start; off < end; ++off) {
                const uint32_t u = params.nbrs[off];
                
                // Read predecessor distance u
                const uint32_t du = __ldg(&params.distances[u]); 

                // Check: u is predecessor of w (du + 1 = dw)
                if (du + 1u == dw) {
                    
                    // Read predecessor sigma u
                    const uint32_t su = __ldg(&params.sigma[u]); 
                    
                    if (su > 0u) {
                        // Read successor delta w
                        const float dep_w = __ldg(&params.delta[w]); 
                        
                        // Ratio is (sigma[u] / sigma[w]) * (1 + delta[w])
                        const float contrib = ((float)su / (float)sw) * (1.0f + dep_w);
                        
                        atomicAdd(&params.delta[u], contrib);
                    }
                }
            }
            optixIgnoreIntersection();
            return;
        }
    }
    
    optixIgnoreIntersection();
}
