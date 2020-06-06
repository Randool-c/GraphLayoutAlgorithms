import json


# def write_data(filepath, nodes, edges):
#     with open(filepath, 'w', encoding='utf-8') as f:
#         obj = {'nodes': nodes, 'edges': edges}
#         json.dump(obj, f, indent=4)
#         f.close()

def write_data(nodes, edges):
    with open('nodes.txt', 'w') as f:
        for node in nodes:
            f.write('{} {}\n'.format(float(node[0]), float(node[1])))
        f.close()

    with open('edges.txt', 'w') as f:
        for edge in edges:
            f.write('{} {}\n'.format(int(edge[0]), int(edge[1])))
        f.close()
