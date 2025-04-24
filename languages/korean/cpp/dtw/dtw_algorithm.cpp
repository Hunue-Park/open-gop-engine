// src/cpp/dtw/dtw_algorithm.cpp
#include "dtw_algorithm.h"
#include <limits>
#include <cmath>
#include <algorithm>

namespace realtime_engine_ko {
namespace dtw {

// 유클리드 거리 계산
double euclidean(const VecD& a, const VecD& b) {
    double sum = 0;
    for (size_t k = 0; k < a.size(); ++k)
        sum += (a[k] - b[k]) * (a[k] - b[k]);
    return std::sqrt(sum);
}

// AsymmetricP1 단계 패턴: (di, dj, weight)
const std::vector<std::vector<std::array<double,3>>> pattern = {
    // 패턴 0
    { {-1,-2, NAN}, { 0,-1, 0.5}, { 0,0, 0.5} },
    // 패턴 1
    { {-1,-1, NAN}, { 0,0, 1.0} },
    // 패턴 2
    { {-2,-1, NAN}, {-1,0, 1.0}, { 0,0, 1.0} }
};

// DTW 정렬 함수 (X[n×d], Y[m×d] 입력)
PairVI dtw_align(const std::vector<VecD>& X,
                 const std::vector<VecD>& Y) {
    int n = X.size(), m = Y.size();
    // 1) 로컬 거리 행렬 계산
    MatD cost(n, VecD(m));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            cost[i][j] = euclidean(X[i], Y[j]);

    const double INF = std::numeric_limits<double>::infinity();
    // 2) 누적 비용 행렬 D와 방향 저장용 dir 행렬
    MatD D(n+1, VecD(m+1, INF));
    std::vector<std::vector<int>> dir(n+1, std::vector<int>(m+1, -1));
    D[0][0] = 0.0;

    // 3) DP 채우기
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            double best = INF;
            int best_p = -1;
            // 모든 패턴 시도
            for (int p = 0; p < (int)pattern.size(); ++p) {
                const auto& pat = pattern[p];
                // 시작 지점 (base offset)
                int bi = i + (int)pat[0][0];
                int bj = j + (int)pat[0][1];
                if (bi < 0 || bj < 0) continue;
                double c = D[bi][bj];
                // 패턴의 각 엣지 가중치 * 해당 셀 비용
                for (size_t s = 1; s < pat.size(); ++s) {
                    int ii = i + (int)pat[s][0];
                    int jj = j + (int)pat[s][1];
                    c += pat[s][2] * cost[ii-1][jj-1];
                }
                if (c < best) {
                    best = c;
                    best_p = p;
                }
            }
            D[i][j] = best;
            dir[i][j] = best_p;
        }
    }

    // 4) 역추적: (n,m) → (0,0)
    std::vector<int> idx1, idx2;
    int i = n, j = m;
    while (i > 0 && j > 0) {
        idx1.push_back(i-1);
        idx2.push_back(j-1);
        int p = dir[i][j];
        const auto& pat = pattern[p];
        // base offset로 이동
        i += (int)pat[0][0];
        j += (int)pat[0][1];
    }
    std::reverse(idx1.begin(), idx1.end());
    std::reverse(idx2.begin(), idx2.end());
    return {idx1, idx2};
}

} // namespace dtw
} // namespace realtime_engine_ko