def read_data(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()
        f.close()
    data = [x.split() for x in data[1:]]
    n_nodes = int(data[0][0])
    n_edges = int(data[0][2])

    nodes = list(range(1, 1 + n_nodes))
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    edges = []
    for edge in data[1:]:
        src = int(edge[0])
        dst = int(edge[1])
        if src == dst:
            continue

        edges.append([int(node_to_idx[src]), int(node_to_idx[dst])])
    return nodes, edges
