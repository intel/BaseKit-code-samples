//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
template <int It, int End> struct unroller {
  template <typename Action> static void step(const Action &action) {
    action(std::integral_constant<int, It>());
    unroller<It + 1, End>::step(action);
  }
};

template <int End> struct unroller<End, End> {
  template <typename Action> static void step(const Action &) {}
};
