#ifndef ALGORITHMS_ADAPTED_INIT_H
#define ALGORITHMS_ADAPTED_INIT_H

#include "../utils/graph.h"
#include "../utils/matrix.h"
#include "../utils/shortest_path.hpp"
#include "../layout/base_optimizer.hpp"
#include<vector>
#include<algorithm>
#include<set>
#include<cstdlib>
#include<ctime>
#include<utility>
#include<iostream>
#include<cmath>
#include<ctime>

namespace adapted_init{
    const float lr_max = 1.0;
    const float lr_min = 0.05;
    const float max_step = 10;
    const int max_effective_center_dist = 10;  // stress迭代更新时考虑的最远的center距离
}

#endif //ALGORITHMS_ADAPTED_INIT_H
