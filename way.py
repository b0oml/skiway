#!/usr/bin/env python3

import json
import networkx as nx
# import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt


with open('data.json') as f:
    data = json.load(f)

nodes_map = {node['id']: node for node in data['elements'] if node['type'] == 'node'}


def haversine(coord1, coord2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1 = map(radians, coord1)
    lon2, lat2 = map(radians, coord2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km


def construct_graph(data):
    graph = nx.DiGraph()

    # First pass to add nodes/edges
    for element in data['elements']:
        if element['type'] == 'way' and 'nodes' in element:
            if element.get('tags', {}).get('piste:type'):
                edge_data = {
                    'type': 'piste',
                    'name': element['tags'].get('name', 'unknown'),
                    'difficulty': element['tags'].get('piste:difficulty', 'unknown'),
                }
                for i in range(len(element['nodes']) - 1):
                    start_node = element['nodes'][i]
                    end_node = element['nodes'][i + 1]
                    graph.add_edge(start_node, end_node, **edge_data)

            elif element.get('tags', {}).get('aerialway'):
                edge_data = {
                    'type': 'aerialway',
                    'name': element['tags'].get('name', 'unknown'),
                }
                start_node = element['nodes'][0]
                end_node = element['nodes'][-1]
                graph.add_edge(start_node, end_node, **edge_data)

    # Add piste near the aerialway
    edges_to_add = []

    for edge in graph.edges(data=True):
        start_node, end_node, edge_data = edge
        if edge_data.get('type') != 'aerialway':
            continue

        # If a node is within 100 meters of start or end node, add an edge
        for node in graph.nodes(data=True):
            node_id = node[0]
            if node_id in [start_node, end_node]:
                continue

            node_data = nodes_map[node_id]

            # Calculate distance
            start_coords = (nodes_map[start_node]['lon'], nodes_map[start_node]['lat'])
            node_coords = (node_data['lon'], node_data['lat'])
            end_coords = (nodes_map[end_node]['lon'], nodes_map[end_node]['lat'])
            if haversine(start_coords, node_coords) < 0.1:
                edges_to_add.append((node_id, start_node, {
                    'type': 'piste',
                    'name': f'Piste near {edge_data["name"]}',
                    'difficulty': 'unknown',
                }))
            if haversine(end_coords, node_coords) < 0.1:
                edges_to_add.append((end_node, node_id, {
                    'type': 'piste',
                    'name': f'Piste near {edge_data["name"]}',
                    'difficulty': 'unknown',
                }))

    for edge in edges_to_add:
        graph.add_edge(edge[0], edge[1], **edge[2])

    return graph


def display_graph(graph):
    # Prepare colors based on difficulty levels
    difficulty_colors = {
        'easy': 'green',
        'intermediate': 'blue',
        'advanced': 'red',
        'expert': 'black',
        'unknown': 'gray'  # Default color for undefined difficulty
    }

    # Prepare for visualization
    edge_colors = [difficulty_colors.get(graph[u][v].get('difficulty'), 'gray') for u, v in graph.edges()]
    edge_labels = {(u, v): graph[u][v]['name'] for u, v in graph.edges()}

    # Generate a layout for our nodes
    pos = nx.spring_layout(graph)

    # Draw the graph
    nx.draw(graph, pos, edge_color=edge_colors, with_labels=False, node_size=0)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)

    plt.show()


def deduplicate_nodes_by_edge_name(graph):
    # Create a copy of the graph to modify it while iterating
    new_graph = graph.copy()

    # We'll check for removable nodes based on the in-degree and out-degree being 1
    removable_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 1 and graph.out_degree(node) == 1]

    for node in removable_nodes:
        # Get incoming and outgoing edges
        in_edges = list(graph.in_edges(node, data=True))
        out_edges = list(graph.out_edges(node, data=True))

        # We should only have one in-edge and one out-edge since the in-degree and out-degree are 1
        if len(in_edges) == 1 and len(out_edges) == 1:
            in_edge = in_edges[0]
            out_edge = out_edges[0]

            # Check if the edges have the same name and thus can be merged
            if in_edge[2]['name'] == out_edge[2]['name']:
                # Remove the node and add a direct edge from the predecessor to the successor
                new_graph.remove_node(node)
                new_graph.add_edge(in_edge[0], out_edge[1], **in_edge[2])

    return new_graph


def find_shortest_path_by_name(graph, start_name, end_name):
    # Filter edges by name
    start_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('name') == start_name]
    end_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('name') == end_name]

    # Find the shortest path between any of the start edges and end edges
    shortest_path = None
    shortest_length = float('inf')
    for s_u, s_v in start_edges:
        for e_u, e_v in end_edges:
            try:
                path = nx.shortest_path(graph, source=s_u, target=e_u)
                length = sum(nx.shortest_path_length(graph, source=n, target=n_next)
                             for n, n_next in zip(path[:-1], path[1:]))
                if length < shortest_length:
                    shortest_path = path
                    shortest_length = length
            except nx.NetworkXNoPath:
                continue  # No path found for this combination, move to next

    return shortest_path, shortest_length


def get_edge_name(graph, u, v):
    data = graph.get_edge_data(u, v)
    return data['name'] if data else 'Unknown'


def get_path_edges(graph, path):
    edges = []
    for i in range(len(path) - 1):
        edge = graph.get_edge_data(path[i], path[i + 1])
        if (len(edges) == 0 or edge['name'] != edges[-1]['name']) and not edge['name'].startswith('Piste near'):
            edges.append(edge)
    return edges


graph = construct_graph(data)


if __name__ == '__main__':
    # simplified = deduplicate_nodes_by_edge_name(graph)
    # display_graph(graph)
    # display_graph(simplified)

    # Find shortest path from edge with name 'Llop' to edge with name 'TSD4 Tarter'
    import sys

    # start_edge_name = 'Llop'
    # start_edge_name = 'Esquirol'
    # end_edge_name = 'TSD4 Tarter'

    start_edge_name = sys.argv[1]
    end_edge_name = sys.argv[2]

    path, length = find_shortest_path_by_name(graph, start_edge_name, end_edge_name)
    print(get_path_edges(graph, path))

    # construct a json with all the edges (unique by name)
    edges = []
    for edge in graph.edges(data=True):
        if edge[2]['name'] not in [e['name'] for e in edges]:
            edges.append(edge[2])
    
    with open('edges.json', 'w') as f:
        json.dump(edges, f)
