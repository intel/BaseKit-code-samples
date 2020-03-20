//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

namespace {
template <int Dim1, int... Dims>
struct VerifierDimLayer {
  template <int Idx1, int... Idxs>
  struct VerifierIdxLayer {
    static constexpr bool is_valid() {
      return Idx1 < Dim1 &&
             (VerifierDimLayer<Dims...>::template VerifierIdxLayer<
                 Idxs...>::is_valid());
    }
  };
};
template <int Dim>
struct VerifierDimLayer<Dim> {
  template <int Idx>
  struct VerifierIdxLayer {
    static constexpr bool is_valid() { return Idx < Dim; }
  };
};
}  // namespace
