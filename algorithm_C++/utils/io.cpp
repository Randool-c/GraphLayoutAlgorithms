//
// Created by chenst on 2020/1/6.
//

#include "io.h"

void io::read_dataset(std::string dataset_name, Graph &graph){
    std::string path = ROOT + "/dataset/" + dataset_name + "/" + dataset_name + ".mtx";
    char buffer[256];
    std::cout << path << std::endl;
    freopen((char*)path.data(), "r", stdin);

    std::cin.getline(buffer, 80);
    int n_nodes, n_edges;
    std::string src, dst;
    std::string value;
    std::cin >> n_nodes >> n_nodes >> n_edges;
    graph.resize(n_nodes, n_edges);
    for (int i = 0; i < n_edges; ++i){
        std::cin >> src >> dst >> value;
        graph.insert_edge(src, dst, 1);
    }
    fclose(stdin);
}
