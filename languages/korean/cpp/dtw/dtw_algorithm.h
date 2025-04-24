// src/cpp/dtw/dtw_algorithm.h
#pragma once

#include <vector>
#include <array>
#include <utility>

namespace realtime_engine_ko {
namespace dtw {

using VecD = std::vector<double>;
using MatD = std::vector<std::vector<double>>;
using PairVI = std::pair<std::vector<int>, std::vector<int>>;

// 유클리드 거리 계산
double euclidean(const VecD& a, const VecD& b);

// DTW 정렬 함수 (X[n×d], Y[m×d] 입력)
PairVI dtw_align(const std::vector<VecD>& X,
                 const std::vector<VecD>& Y);

} // namespace dtw
} // namespace realtime_engine_ko