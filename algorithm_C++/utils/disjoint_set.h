//
// Created by chenst on 2020/3/25.
//

#ifndef MULTILEVEL_STRESS_C___DISJOINT_SET_H
#define MULTILEVEL_STRESS_C___DISJOINT_SET_H

#include<vector>
#include<algorithm>
#include<set>
#include<map>
#include<iostream>


class DisjointSet{
public:
    int n;
    std::vector<int> parent;
    std::set<int> roots;

    DisjointSet(int n=1);
    int get_root(int idx);
    int merge(int root1, int root2);
    int get_n_roots();  // 返回剩下的root个数
    int get_n_cluster_nodes(int root);  // 返回一个cluster中的元素个数
};


#endif //MULTILEVEL_STRESS_C___DISJOINT_SET_H
