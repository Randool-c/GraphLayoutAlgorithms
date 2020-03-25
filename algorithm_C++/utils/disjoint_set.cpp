//
// Created by chenst on 2020/3/25.
//

#include "disjoint_set.h"

DisjointSet::DisjointSet(int n) {
    parent.resize(n);
    std::fill(parent.begin(), parent.end(), -1);
    for (int i = 0; i < n; ++i){
        roots.insert(i);
    }
}

int DisjointSet::get_root(int idx) {
    if (parent[idx] < 0){
        return idx;
    }
    else return (parent[idx] = get_root(parent[idx]));
}

int DisjointSet::merge(int root1, int root2){
    /*
     * 返回合并后的根
     */
//    root1 = get_root(root1);
//    root2 = get_root(root2);
    if (root1 == root2) return root1;
    else{
        if (parent[root1] < parent[root2]){
            parent[root1] += parent[root2];
            parent[root2] = root1;
            roots.erase(root2);
            return root1;
        }
        else{
            parent[root2] += parent[root1];
            parent[root1] = root2;
            roots.erase(root1);
            return root2;
        }
    }
}

int DisjointSet::get_n_roots() {
    return roots.size();
}

int DisjointSet::get_n_cluster_nodes(int root) {
    return -parent[root];
}


//int main(){
//    DisjointSet x(10);
//    x.merge(1, 5);
//    x.merge(1, 9);
//    x.merge(2, 8);
//    x.merge(5, 8);
//    x.merge(3, 4);
//    x.merge(6, 4);
//    x.merge(4, 0);
//    std::cout << x.get_n_roots() << std::endl;
//    return 0;
//}
