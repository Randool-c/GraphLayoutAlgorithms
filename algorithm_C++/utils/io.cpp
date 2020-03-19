//
// Created by chenst on 2020/1/6.
//

#include "io.h"

void io::read_dataset(std::string dataset_name, Graph &graph){
    std::string path = ROOT + "/dataset/" + dataset_name + "/" + dataset_name + ".mtx";
    char buffer[256];
    std::cout << path << std::endl;
//    freopen((char*)path.data(), "r", stdin);
    std::ifstream fin((char*)path.data(), std::ios::in);

    fin.getline(buffer, 128);
    int n_nodes, n_edges;
    std::string src, dst;
    std::string value;
    fin >> n_nodes >> n_nodes >> n_edges;
    fin.get();
    graph.resize(n_nodes, n_edges);
    std::string line;
    for (int i = 0; i < n_edges; ++i){
        fin.getline(buffer, 128);
        std::istringstream is(buffer);
//        std::cout << buffer << std::endl;
        is >> src >> dst;
//        std::cout << "src dst: " << src << ' ' << dst << std::endl;
        graph.insert_edge(src, dst, 1);
//        std::cout << "slhfjoewf" << std::endl;
    }
    std::cout << "else" << std::endl;
//    for (int i = 0; i < n_edges; ++i){
//        std::cin >> src >> dst >> value;
//        graph.insert_edge(src, dst, 1);
//    }
}
