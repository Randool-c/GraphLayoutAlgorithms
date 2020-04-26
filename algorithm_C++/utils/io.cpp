//
// Created by chenst on 2020/1/6.
//

#include "io.h"

void io::read_dataset(std::string dataset_name, Graph &graph){
    std::string path = ROOT + "/dataset/" + dataset_name + "/" + dataset_name + ".mtx";
    char buffer[256];
    std::cout << path << std::endl;
    std::ifstream fin((char*)path.data(), std::ios::in);

    do {
        fin.getline(buffer, 128);
    } while(buffer[0] == '%');

    int n_nodes, n_edges;
    std::istringstream is(buffer);
    is >> n_nodes >> n_nodes >> n_edges;
    std::string src, dst;
    std::string value;

    std::cout << n_nodes << " " << n_edges << std::endl;

    graph.resize(n_nodes, n_edges);
    std::string line;
    for (int i = 0; i < n_edges; ++i){
        fin.getline(buffer, 128);
        std::istringstream is(buffer);
        is >> src >> dst;
        graph.insert_edge(src, dst, 1);
    }
}
