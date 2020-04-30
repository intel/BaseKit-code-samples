//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <utility>

#include "CL/sycl.hpp"
#include "pipe_array_internal.hpp"

template <class Id, typename BaseTy, std::size_t depth, int... dims>
struct PipeArray {
  PipeArray() = delete;

  template <int... idxs>
  struct StructId;

  template <int... idxs>
  struct VerifyIndices {
    static_assert(sizeof...(idxs) == sizeof...(dims),
                  "Indexing into a PipeArray requires as many indices as "
                  "dimensions of the PipeArray.");
    static_assert(VerifierDimLayer<dims...>::template VerifierIdxLayer<
                      idxs...>::is_valid(),
                  "Index out of bounds");
    using VerifiedPipe =
        cl::sycl::intel::pipe<StructId<idxs...>, BaseTy, depth>;
  };

  template <int... idxs>
  using PipeAt = typename VerifyIndices<idxs...>::VerifiedPipe;
};
