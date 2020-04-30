//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

namespace {
template <int dim1, int... dims>
struct VerifierDimLayer {
  template <int idx1, int... idxs>
  struct VerifierIdxLayer {
    static constexpr bool is_valid() {
      return idx1 < dim1 &&
             (VerifierDimLayer<dims...>::template VerifierIdxLayer<
                 idxs...>::is_valid());
    }
  };
};
template <int dim>
struct VerifierDimLayer<dim> {
  template <int idx>
  struct VerifierIdxLayer {
    static constexpr bool is_valid() { return idx < dim; }
  };
};
}  // namespace
