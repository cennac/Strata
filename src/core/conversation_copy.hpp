#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core::conversation_detail {
#if defined(STRATA_USE_HIP)
inline std::atomic<int>& pinned_buffers_live() { static std::atomic<int> count{0}; return count; }
#endif

/// Conversation snapshot / checkpoint copies go through a private non-blocking stream of the current device
/// instead of the legacy (null) stream.  A plain cudaMemcpy is ordered against every blocking stream of the
/// device, and while another thread captures a graph on one (a drafter or a stage worker), HIP refuses it with
/// "operation would make the legacy stream depend on a capturing blocking stream".  The caller's contract is
/// unchanged: the device was synchronized before the copy (the copy itself is complete on return).
inline cudaError_t copy_nonblocking(void* dst, const void* src, size_t bytes) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<cudaStream_t> streams;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);   // never destroyed: process lifetime
    if (!streams[(size_t) dev]) {
        e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
        if (e != cudaSuccess) { streams[(size_t) dev] = nullptr; return e; }
    }
    cudaStream_t s = streams[(size_t) dev];
    // Experimental HIP-only staging, enabled explicitly for the isolated candidate.
#if defined(STRATA_USE_HIP)
    static const bool pinned = [] { const char* v = std::getenv("STRATA_CONVERSATION_PINNED_COPY"); return v && std::strcmp(v, "1") == 0; }();
    if (pinned && bytes) {
        auto kind = [](const void* p) {
            hipPointerAttribute_t a{};
            auto err = hipPointerGetAttributes(&a, p);
            if (err == hipErrorInvalidValue) { (void)hipGetLastError(); return 0; }
            if (err != hipSuccess) return -1;
            if (a.type == hipMemoryTypeDevice || a.isManaged) return 1;
            return a.type == hipMemoryTypeHost ? 2 : 0;
        };
        const int dk = kind(dst), sk = kind(src);
        if (dk < 0 || sk < 0) return hipErrorInvalidValue;
        const bool to_device = dk == 1 && sk == 0;
        const bool to_host = dk == 0 && sk == 1;
        if (to_device || to_host) {
            constexpr size_t capacity = 4 * 1024 * 1024;
            struct OwnedScratch {
                std::vector<void*> buffers;
                ~OwnedScratch() {
                    for (void* p : buffers) if (p) {
                        const auto status = cudaFreeHost(p);
                        if (status == cudaSuccess) pinned_buffers_live().fetch_sub(1);
                        else std::fprintf(stderr, "[pinned-copy] release failed: %s\n", cudaGetErrorString(status));
                    }
                }
            };
            thread_local OwnedScratch owned;
            auto& scratch = owned.buffers;
            if (scratch.size() <= (size_t)dev) scratch.resize((size_t)dev + 1, nullptr);
            if (!scratch[dev]) {
                e = cudaMallocHost(&scratch[dev], capacity);
                if (e != cudaSuccess) return e;
                pinned_buffers_live().fetch_add(1);
                std::fprintf(stderr, "[pinned-copy] device=%d staging_bytes=%zu\n", dev, capacity);
            }
            for (size_t offset = 0; offset < bytes;) {
                const size_t count = std::min(capacity, bytes - offset);
                auto* d = static_cast<unsigned char*>(dst) + offset;
                auto* q = static_cast<const unsigned char*>(src) + offset;
                if (to_device) std::memcpy(scratch[dev], q, count);
                e = cudaMemcpyAsync(to_device ? static_cast<void*>(d) : scratch[dev],
                                    to_device ? scratch[dev] : static_cast<const void*>(q),
                                    count, to_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, s);
                if (e != cudaSuccess) return e;
                e = cudaStreamSynchronize(s);
                if (e != cudaSuccess) return e;
                if (to_host) std::memcpy(d, scratch[dev], count);
                offset += count;
            }
            return cudaSuccess;
        }
    }
#endif

    e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s);
    if (e != cudaSuccess) return e;
    return cudaStreamSynchronize(s);
}

} // namespace strata::core::conversation_detail
