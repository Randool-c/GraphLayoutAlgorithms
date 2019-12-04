import json


def write_data(filepath, nodes, edges):
    with open(filepath, 'w', encoding='utf-8') as f:
        obj = {'nodes': nodes, 'edges': edges}
        json.dump(obj, f, indent=4)
        f.close()
