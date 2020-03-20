//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl/intel/fpga_extensions.hpp>
#include <utility>

#include "CL/sycl.hpp"
#include "pipe_array_internal.h"

template <class Id, typename BaseTy, std::size_t Depth, int... Dims>
struct PipeArray {
  PipeArray() = delete;

  template <int... Idxs>
  struct StructId;

  template <int... Idxs>
  struct VerifyIndices {
    static_assert(sizeof...(Idxs) == sizeof...(Dims),
                  "Indexing into a PipeArray requires as many indices as "
                  "dimensions of the PipeArray.");
    static_assert(VerifierDimLayer<Dims...>::template VerifierIdxLayer<
                      Idxs...>::is_valid(),
                  "Index out of bounds");
    using VerifiedPipe = cl::sycl::pipe<StructId<Idxs...>, BaseTy, Depth>;
  };

  template <int... Idxs>
  using pipe_at = typename VerifyIndices<Idxs...>::VerifiedPipe;
};
