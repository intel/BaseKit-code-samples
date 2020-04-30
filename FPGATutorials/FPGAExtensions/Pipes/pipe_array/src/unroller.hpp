//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
template <int it, int end> struct Unroller {
  template <typename Action> static void step(const Action &action) {
    action(std::integral_constant<int, it>());
    Unroller<it + 1, end>::step(action);
  }
};

template <int end> struct Unroller<end, end> {
  template <typename Action> static void step(const Action &) {}
};
