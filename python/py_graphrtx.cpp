#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <optix_function_table_definition.h>

#include <memory>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "shared.h"
#include "memory/gpu_manager.hpp"
#include "memory/buffer.hpp"
#include "memory/register.hpp"
#include "common.hpp"
#include "graph/graph.hpp"
#include "algorithms/partition.hpp"
#include "algorithms/bfs.hpp"
#include "algorithms/pr.hpp"
#include "algorithms/bc.hpp"
#include "algorithms/sssp.hpp"
#include "algorithms/tc.hpp"
#include "kernels/cuda/algorithms.cuh"
#include "rt/rt_pipeline.hpp"

namespace py = pybind11;

class PyGraphRTX {
public:
    explicit PyGraphRTX(int device = 0) {
        CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
        CUDA_CHECK(cudaSetDevice(device));

        CUDA_CHECK(cudaStreamCreateWithFlags(&streamCompute_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&streamTransfer_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&h2dDone_, cudaEventDisableTiming));

        rt_pipe_ = std::make_shared<rt_pipeline>();
        ctx_     = rt_pipe_->get_context();
        module_  = rt_pipe_->get_module();
        pipeline_ = rt_pipe_->get_pipeline();
        sbt_      = rt_pipe_->get_sbt();

        graph_ = std::make_shared<graph_rtx>();
        std::cout << "Created GraphRTX instance on device " << device << std::endl;
    }

    ~PyGraphRTX() {
        if (!cleaned_) abandon();
    }

    int load_graph(const std::string& path) {
        int N = graph_->load_mtx_graph(path);
        if (N <= 0)
            throw std::runtime_error("Failed to load graph: " + path);
        N_ = N;
        return N;
    }

    void from_networkx(py::object nx_graph) {
        namespace py = pybind11;
        using namespace std;

        py::module nx = py::module::import("networkx");

        if (!py::isinstance(nx_graph, nx.attr("Graph")) &&
            !py::isinstance(nx_graph, nx.attr("DiGraph"))) {
            throw std::runtime_error("from_networkx: object is not a networkx Graph or DiGraph");
        }

        int num_nodes = py::cast<int>(nx_graph.attr("number_of_nodes")());
        int num_edges = py::cast<int>(nx_graph.attr("number_of_edges")());

        std::vector<std::pair<uint32_t, uint32_t>> edges;
        edges.reserve(num_edges);

        for (auto e : nx_graph.attr("edges")()) {
            auto tup = py::cast<py::tuple>(e);
            uint32_t u = py::cast<uint32_t>(tup[0]);
            uint32_t v = py::cast<uint32_t>(tup[1]);
            edges.emplace_back(u, v);
        }

        // Detect and extract edge weights if present
        bool has_weight = false;
        std::vector<float> weights;
        try {
            for (auto e : nx_graph.attr("edges")(py::arg("data")=true)) {
                auto tup = py::cast<py::tuple>(e);
                auto data = py::cast<py::dict>(tup[2]);
                if (data.contains("weight")) {
                    has_weight = true;
                    break;
                }
            }
        } catch (...) {}

        if (has_weight) {
            weights.reserve(edges.size());
            for (auto e : nx_graph.attr("edges")(py::arg("data")=true)) {
                auto tup = py::cast<py::tuple>(e);
                auto data = py::cast<py::dict>(tup[2]);
                float w = data.contains("weight") ? py::cast<float>(data["weight"]) : 1.0f;
                weights.push_back(w);
            }
        } else {
            weights.assign(edges.size(), 1.0f);
        }

        // Build CSR
        std::vector<std::vector<uint32_t>> adj(num_nodes);
        for (auto [u, v] : edges)
            adj[u].push_back(v);

        std::vector<uint32_t> row_ptr(num_nodes + 1, 0);
        for (int i = 0; i < num_nodes; ++i)
            row_ptr[i + 1] = row_ptr[i] + adj[i].size();

        std::vector<uint32_t> nbrs;
        nbrs.reserve(edges.size());
        for (int i = 0; i < num_nodes; ++i)
            nbrs.insert(nbrs.end(), adj[i].begin(), adj[i].end());

        graph_->set_graph_from_csr(row_ptr, nbrs, weights);
        N_ = num_nodes;

        std::cout << "Loaded NetworkX graph with " << num_nodes
                  << " nodes and " << num_edges << " edges." << std::endl;
    }

    py::object to_networkx() const {
        namespace py = pybind11;
        py::module nx = py::module::import("networkx");
        py::object G = nx.attr("DiGraph")();

        const auto& row_ptr = graph_->get_row_ptr();
        const auto& nbrs = graph_->get_nbrs_ptr();
        const auto& wts = graph_->get_wts();

        size_t num_nodes = row_ptr.size() - 1;
        for (size_t u = 0; u < num_nodes; ++u) {
            for (uint32_t i = row_ptr[u]; i < row_ptr[u + 1]; ++i) {
                uint32_t v = nbrs[i];
                float w = (i < wts.size()) ? wts[i] : 1.0f;
                G.attr("add_edge")(u, v, py::arg("weight") = w);
            }
        }
        return G;
    }

    py::dict prepare(uint32_t max_seg_len = 1024, int num_partitions = 0, int num_dummy_nodes = 0) {
        using namespace std::chrono;
        auto& row_ptr = graph_->get_row_ptr();
        auto& nbrs = graph_->get_nbrs_ptr();
        auto& wts = graph_->get_wts();

        size_t row_ptr_bytes = row_ptr.size() * sizeof(uint32_t);
        size_t nbrs_bytes    = nbrs.size() * sizeof(uint32_t);
        size_t wts_bytes     = wts.size()  * sizeof(float);

        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

        size_t graph_bytes = row_ptr_bytes + nbrs_bytes + wts_bytes;
        use_gpu_mem_ = (graph_bytes < static_cast<size_t>(free_mem * 0.8));

        if (use_gpu_mem_) {
            PinnedRegister pin_row(row_ptr.data(), row_ptr_bytes);
            PinnedRegister pin_nbr(nbrs.data(), nbrs_bytes);
            PinnedRegister pin_wts(wts.data(), wts_bytes);

            CUDA_CHECK(cudaMalloc(&d_row_ptr_, row_ptr_bytes));
            CUDA_CHECK(cudaMalloc(&d_nbrs_,    nbrs_bytes));
            CUDA_CHECK(cudaMalloc(&d_wts_,     wts_bytes));

            CUDA_CHECK(cudaMemcpyAsync(d_row_ptr_, row_ptr.data(), row_ptr_bytes,
                                       cudaMemcpyHostToDevice, streamTransfer_));
            CUDA_CHECK(cudaMemcpyAsync(d_nbrs_, nbrs.data(), nbrs_bytes,
                                       cudaMemcpyHostToDevice, streamTransfer_));
            CUDA_CHECK(cudaMemcpyAsync(d_wts_, wts.data(), wts_bytes,
                                       cudaMemcpyHostToDevice, streamTransfer_));
            CUDA_CHECK(cudaEventRecord(h2dDone_, streamTransfer_));
        } else {
            if (!row_ptr.empty()) CUDA_CHECK(cudaHostRegister(row_ptr.data(), row_ptr_bytes, cudaHostRegisterMapped));
            if (!nbrs.empty())    CUDA_CHECK(cudaHostRegister(nbrs.data(), nbrs_bytes, cudaHostRegisterMapped));
            if (!wts.empty())     CUDA_CHECK(cudaHostRegister(wts.data(), wts_bytes, cudaHostRegisterMapped));

            CUDA_CHECK(cudaHostGetDevicePointer(&d_row_ptr_, row_ptr.data(), 0));
            CUDA_CHECK(cudaHostGetDevicePointer(&d_nbrs_,    nbrs.data(),    0));
            CUDA_CHECK(cudaHostGetDevicePointer(&d_wts_,     wts.data(),     0));
            CUDA_CHECK(cudaEventRecord(h2dDone_, streamTransfer_));
        }

        auto t0 = high_resolution_clock::now();
        graph_->build_uasps(max_seg_len);
        auto t1 = high_resolution_clock::now();
        graph_->build_aabbs();
        auto t2 = high_resolution_clock::now();
        graph_->append_dummy_aabbs_tagged(num_dummy_nodes);
        auto t3 = high_resolution_clock::now();

        auto& uasp_first = graph_->get_uasp_first();
        auto& uasp_count = graph_->get_uasp_count();
        auto& uasps_host = graph_->get_uasp_host();
        auto& aabbs6     = graph_->get_aabb();
        auto& aabb_mask  = graph_->get_mask();

        d_aabb_mask_ = std::make_unique<DeviceBuffer<uint8_t>>(aabb_mask.size());
        d_aabb_mask_->uploadAsync(aabb_mask.data(), aabb_mask.size(), streamTransfer_);

        d_uasps_ = std::make_unique<DeviceBuffer<UASP>>(uasps_host.size());
        d_aabbs_ = std::make_unique<DeviceBuffer<float>>(aabbs6.size());
        d_uasp_first_ = std::make_unique<DeviceBuffer<uint32_t>>(uasp_first.size());
        d_uasp_count_ = std::make_unique<DeviceBuffer<uint32_t>>(uasp_count.size());

        d_uasps_->uploadAsync(uasps_host.data(), uasps_host.size(), streamTransfer_);
        d_aabbs_->uploadAsync(aabbs6.data(), aabbs6.size(), streamTransfer_);
        d_uasp_first_->uploadAsync(uasp_first.data(), uasp_first.size(), streamTransfer_);
        d_uasp_count_->uploadAsync(uasp_count.data(), uasp_count.size(), streamTransfer_);

        CUDA_CHECK(cudaStreamWaitEvent(streamCompute_, h2dDone_, 0));
        CUDA_CHECK(cudaEventDestroy(h2dDone_));
        h2dDone_ = nullptr;

        CUDA_CHECK(cudaMalloc((void**)&d_params_, sizeof(Params)));

        base_.num_vertices = static_cast<uint32_t>(N_);
        base_.row_ptr = d_row_ptr_;
        base_.nbrs = d_nbrs_;
        base_.weights = d_wts_;
        base_.aabb_mask = (const uint8_t*)d_aabb_mask_->ptr;
        base_.uasps = (const UASP*)d_uasps_->ptr;
        base_.aabbs = (const float*)d_aabbs_->ptr;
        base_.num_uasps = (uint32_t)uasps_host.size();
        base_.num_aabbs = (uint32_t)(aabbs6.size() / 6);
        base_.uasp_first = (const uint32_t*)d_uasp_first_->ptr;
        base_.uasp_count = (const uint32_t*)d_uasp_count_->ptr;

        mm_ = std::make_unique<GPUMemoryManager>(ctx_, uasps_host, aabbs6,
                                                 (uint32_t)uasps_host.size(), 0.85f,
                                                 num_partitions, &aabb_mask);

        py::dict stats;
        stats["nodes"] = N_;
        stats["edges"] = (int)nbrs.size();
        stats["gpu_memory_used_MB"] = toMB(row_ptr_bytes + nbrs_bytes + wts_bytes);
        stats["uasp_build_ms"] = duration<double, std::milli>(t1 - t0).count();
        stats["aabb_build_ms"] = duration<double, std::milli>(t2 - t1).count();
        stats["dummy_build_ms"] = duration<double, std::milli>(t3 - t2).count();
        return stats;
    }

    py::dict device_memory_info_gb() {
        size_t freeB = 0, totalB = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
        py::dict d;
        d["free_gb"] = toGB(freeB);
        d["total_gb"] = toGB(totalB);
        return d;
    }

    py::dict run_bfs(int src = 0, bool hybrid = false) {
        return run_one([&]{ graph_->bfs(rt_pipe_, *mm_, d_params_, base_, src, N_, streamCompute_, hybrid); }, "bfs");
    }
    py::dict run_pr(int iters = 20, float damp = 0.85f, bool hybrid = false) {
        return run_one([&]{ graph_->pr(rt_pipe_, *mm_, d_params_, base_, N_, iters, damp, streamCompute_, hybrid); }, "pr");
    }
    py::dict run_sssp(int src = 0, bool hybrid = false) {
        return run_one([&]{ graph_->sssp(rt_pipe_, *mm_, d_params_, base_, src, N_, streamCompute_, hybrid); }, "sssp");
    }
    py::dict run_bc(bool hybrid = false) {
        return run_one([&]{ graph_->bc(rt_pipe_, *mm_, d_params_, base_, N_, streamCompute_, hybrid); }, "bc");
    }
    py::dict run_tc(bool hybrid = false) {
        return run_one([&]{ graph_->tc(rt_pipe_, *mm_, d_params_, base_, N_, streamCompute_, hybrid); }, "tc");
    }

    py::dict run_all(int src = 0, int pr_iters = 20, float pr_damp = 0.85f) {
        std::vector<py::dict> results;
        results.push_back(run_bfs(src, false));
        results.push_back(run_pr(pr_iters, pr_damp, false));
        results.push_back(run_sssp(src, false));
        results.push_back(run_bc(false));
        results.push_back(run_tc(false));

        py::dict d;
        d["results"] = results;
        return d;
    }

    void close() {
        if (cleaned_) return;
        cleaned_ = true;
        try {
            if (streamCompute_) cudaStreamSynchronize(streamCompute_);
            if (streamTransfer_) cudaStreamSynchronize(streamTransfer_);

            d_aabb_mask_.reset();
            d_uasps_.reset();
            d_aabbs_.reset();
            d_uasp_first_.reset();
            d_uasp_count_.reset();

            if (d_params_) { cudaFree((void*)d_params_); d_params_ = 0; }

            if (use_gpu_mem_) {
                if (d_row_ptr_) cudaFree((void*)d_row_ptr_);
                if (d_nbrs_)    cudaFree((void*)d_nbrs_);
                if (d_wts_)     cudaFree((void*)d_wts_);
            } else if (graph_) {
                auto& row_ptr = graph_->get_row_ptr();
                auto& nbrs = graph_->get_nbrs_ptr();
                auto& wts  = graph_->get_wts();
                if (!row_ptr.empty()) cudaHostUnregister(row_ptr.data());
                if (!nbrs.empty())    cudaHostUnregister(nbrs.data());
                if (!wts.empty())     cudaHostUnregister(wts.data());
            }

            mm_.reset();
            if (pipeline_) { optixPipelineDestroy(pipeline_); pipeline_ = nullptr; }
            if (module_)   { optixModuleDestroy(module_);     module_   = nullptr; }
            if (ctx_)      { optixDeviceContextDestroy(ctx_); ctx_      = nullptr; }

            if (h2dDone_)       cudaEventDestroy(h2dDone_);
            if (streamCompute_) cudaStreamDestroy(streamCompute_);
            if (streamTransfer_)cudaStreamDestroy(streamTransfer_);
        } catch (...) {}
    }

private:
    void refresh_base_pointers() {
        base_.row_ptr = d_row_ptr_;
        base_.nbrs = d_nbrs_;
        base_.weights = d_wts_;
        base_.aabb_mask = (const uint8_t*)(d_aabb_mask_ ? d_aabb_mask_->ptr : nullptr);
        base_.uasps = (const UASP*)(d_uasps_ ? d_uasps_->ptr : nullptr);
        base_.aabbs = (const float*)(d_aabbs_ ? d_aabbs_->ptr : nullptr);
        base_.uasp_first = (const uint32_t*)(d_uasp_first_ ? d_uasp_first_->ptr : nullptr);
        base_.uasp_count = (const uint32_t*)(d_uasp_count_ ? d_uasp_count_->ptr : nullptr);
    }

    void push_params_async(cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync((void*)d_params_, &base_, sizeof(Params),
                                   cudaMemcpyHostToDevice, stream));
    }

    template <typename F>
    py::dict run_one(F&& f, const char* name) {
        using namespace std::chrono;
        refresh_base_pointers();
        push_params_async(streamCompute_);

        auto t0 = high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaStreamSynchronize(streamCompute_));
        auto t1 = high_resolution_clock::now();

        py::dict d;
        d["algo"] = name;
        d["ms"] = duration<double, std::milli>(t1 - t0).count();
        return d;
    }

    void abandon() noexcept {
        try {
            d_aabb_mask_.release();
            d_uasps_.release();
            d_aabbs_.release();
            d_uasp_first_.release();
            d_uasp_count_.release();
            mm_.release();

            d_params_ = 0;
            d_row_ptr_ = nullptr;
            d_nbrs_ = nullptr;
            d_wts_ = nullptr;
            streamCompute_ = nullptr;
            streamTransfer_ = nullptr;
            h2dDone_ = nullptr;
            pipeline_ = nullptr;
            module_   = nullptr;
            ctx_      = nullptr;
        } catch (...) {}
    }

private:
    std::shared_ptr<graph_rtx> graph_;
    std::shared_ptr<rt_pipeline> rt_pipe_;
    std::unique_ptr<GPUMemoryManager> mm_;

    CUdeviceptr d_params_ = 0;
    uint32_t *d_row_ptr_ = nullptr, *d_nbrs_ = nullptr;
    float *d_wts_ = nullptr;

    std::unique_ptr<DeviceBuffer<uint8_t>> d_aabb_mask_;
    std::unique_ptr<DeviceBuffer<UASP>> d_uasps_;
    std::unique_ptr<DeviceBuffer<float>> d_aabbs_;
    std::unique_ptr<DeviceBuffer<uint32_t>> d_uasp_first_, d_uasp_count_;

    cudaStream_t streamCompute_ = nullptr, streamTransfer_ = nullptr;
    cudaEvent_t h2dDone_ = nullptr;

    OptixDeviceContext ctx_ = nullptr;
    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixShaderBindingTable sbt_{};

    Params base_{};
    bool use_gpu_mem_ = false;
    int N_ = 0;
    bool cleaned_ = false;
};

PYBIND11_MODULE(pygraph_rtx, m) {
    py::class_<PyGraphRTX>(m, "Graph")
        .def(py::init<int>(), py::arg("device") = 0)
        .def("load_graph", &PyGraphRTX::load_graph)
        .def("from_networkx", &PyGraphRTX::from_networkx)
        .def("to_networkx", &PyGraphRTX::to_networkx)
        .def("prepare", &PyGraphRTX::prepare,
             py::arg("max_seg_len")=1024,
             py::arg("num_partitions")=0,
             py::arg("num_dummy_nodes")=0)
        .def("device_memory_info_gb", &PyGraphRTX::device_memory_info_gb)
        .def("run_bfs",  &PyGraphRTX::run_bfs,  py::arg("src")=0, py::arg("hybrid")=false)
        .def("run_pr",   &PyGraphRTX::run_pr,   py::arg("iters")=20, py::arg("damp")=0.85f, py::arg("hybrid")=false)
        .def("run_sssp", &PyGraphRTX::run_sssp, py::arg("src")=0, py::arg("hybrid")=false)
        .def("run_bc",   &PyGraphRTX::run_bc,   py::arg("hybrid")=false)
        .def("run_tc",   &PyGraphRTX::run_tc,   py::arg("hybrid")=false)
        .def("run_all",  &PyGraphRTX::run_all,
             py::arg("src")=0, py::arg("pr_iters")=20, py::arg("pr_damp")=0.85f)
        .def("close",    &PyGraphRTX::close)
        .def("__enter__", [](PyGraphRTX &self) -> PyGraphRTX& { return self; })
        .def("__exit__",  [](PyGraphRTX &self, py::object, py::object, py::object){ self.close(); });
}
