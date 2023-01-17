import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
import pandas as pd
from queue import PriorityQueue
from math import inf

import timeit
from statistics import mean
import numpy as np

import osmnx as ox
ox.settings.use_cache=True


def prepare_test_environment():


    print("Setup beginnt...")
    # Get Base Graph
    adress = "Weingartener Straße 2, Stutensee, Deutschland"
    base_graph = ox.graph_from_address(adress, dist=20000, dist_type='bbox', network_type='drive') # 10000, 20000, 50000

    # Get Source Node
    orig_cords = ox.geocode("Weingartener Straße 2, Stutensee, Deutschland")
    orig_node = ox.nearest_nodes(base_graph, orig_cords[1], orig_cords[0])

    # Get Destination Node
    dest_cords = ox.geocode("Moltkestraße 30, Karlsruhe, Deutschland")
    dest_node = ox.nearest_nodes(base_graph, dest_cords[1], dest_cords[0])

    ### --- Get Sparse Base Graph  --- ###
    adress = "Weingartener Straße 2, Stutensee, Deutschland"
    detail_source_graph = ox.graph_from_address(adress, dist=5000, dist_type='bbox', network_type='drive') # 10000, 20000, 50000
    adress = "Moltkestraße 30, Karlsruhe, Deutschland"
    detail_target_graph = ox.graph_from_address(adress, dist=5000, dist_type='bbox', network_type='drive') # 10000, 20000, 50000
    # Dict with Road-Classification
    cf =  '["highway"~"motorway|trunk|primary|secondary|tertiary|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]'
    adress = "Weingartener Straße 2, Stutensee, Deutschland"
    base_graph_sparse = ox.graph_from_address(adress, dist=20000, dist_type='bbox', network_type='drive', custom_filter=cf) # 10000, 20000, 50000
    base_nodes, base_edges = ox.graph_to_gdfs(base_graph_sparse, nodes=True, edges=True, node_geometry=True)
    source_nodes, source_edges = ox.graph_to_gdfs(detail_source_graph, nodes=True, edges=True, node_geometry=True) #
    target_nodes, target_edges = ox.graph_to_gdfs(detail_target_graph, nodes=True, edges=True, node_geometry=True) #
    # Join all Geo Dataframes
    leveling_edges = pd.concat([base_edges, source_edges, target_edges])
    leveling_nodes = pd.concat([base_nodes, source_nodes, target_nodes])
    # Remove duplicate nodes and edges
    leveling_nodes = leveling_nodes[~leveling_nodes.index.duplicated(keep='first')]
    leveling_edges = leveling_edges[~leveling_edges.index.duplicated(keep='first')]
    leveling_graph = ox.graph_from_gdfs(leveling_nodes, leveling_edges, graph_attrs=None)

    print("Setup abgeschlossen...")

    return base_graph, orig_node, dest_node, leveling_graph



def route_dijkstra(base_graph, orig_node, dest_node):

    di_graph = ox.utils_graph.get_digraph(base_graph, weight="length")
    route, dist = dijkstra(di_graph, orig_node, dest_node)

def route_bidir_dijkstra(base_graph, orig_node, dest_node):

    di_graph = ox.utils_graph.get_digraph(base_graph, weight="length")
    route, dist = bidirectional_dijkstra(di_graph, orig_node, dest_node)

def route_a_star_dijkstra(base_graph, orig_node, dest_node):

    di_graph = ox.utils_graph.get_digraph(base_graph, weight="length")
    route, dist = a_star_dijkstra(di_graph, orig_node, dest_node)

def route_bidir_a_star_dijkstra(base_graph, orig_node, dest_node):

    di_graph = ox.utils_graph.get_digraph(base_graph, weight="length")
    route, dist = bidirectional_a_star_dijkstra(di_graph, orig_node, dest_node)


def route_level_dijkstra(leveling_graph, orig_node, dest_node):

    di_graph = ox.utils_graph.get_digraph(leveling_graph, weight="length")
    route, dist = dijkstra(di_graph, orig_node, dest_node)

def route_level_bidir_dijkstra(leveling_graph, orig_node, dest_node):

    di_graph = ox.utils_graph.get_digraph(leveling_graph, weight="length")
    route, dist = bidirectional_dijkstra(di_graph, orig_node, dest_node)

def route_level_a_star_dijkstra(leveling_graph, orig_node, dest_node):

    di_graph = ox.utils_graph.get_digraph(leveling_graph, weight="length")
    route, dist = a_star_dijkstra(di_graph, orig_node, dest_node)
    
def route_level_bidir_a_star_dijkstra(leveling_graph, orig_node, dest_node):

    di_graph = ox.utils_graph.get_digraph(leveling_graph, weight="length")
    route, dist = bidirectional_a_star_dijkstra(di_graph, orig_node, dest_node)

def backtrace(prev, start, end):
    node = end
    path = []
    while node != start:
        path.append(node)
        node = prev[node]
    path.append(node) 
    path.reverse()
    return path

# Basic Dijkstra 
def dijkstra(graph, start, end):

    #Initialisierung
    # Vorherige Node mit bisher kürzestem Weg zu aktueller Node 
    pred_node = {} 
    # Distanzen initialisieren für alle Nodes inf (unendlich) setzen, außer für unsere Start-Node hier wird 0 gesetzt
    dist = {v: inf for v in list(nx.nodes(graph))} 
    dist[start] = 0
    
    # Bereits besuchte Nodes als Set
    visited = set() 
    
    # prioritize nodes from start -> node with the shortest distance!
    ## elements stored as tuples (distance, node) 
    priority_queue = PriorityQueue()  
    priority_queue.put((dist[start], start))
    
    current_node = start

    # Solange
    while priority_queue.qsize() != 0: # current_node != end:  
        current_node_cost, current_node = priority_queue.get()
        
        # Aktuell besuchte Node zu den besuchten hinzufügen
        visited.add(current_node)
        # Nachfolger der aktuellen Node abrufen und für jeden Nachfolger die Distanz berechnen
        for neighbor in dict(graph.adjacency()).get(current_node):
            path = dist[current_node] + graph.get_edge_data(current_node,neighbor).get('length')

            # Wenn die neue gefundene Distanz über den aktuellen Knoten kürzer ist als die bisher kürzeste Distanz
            if path < dist[neighbor]:
                # Die neue gefundene Distanz als kürzeste Distanz speichern
                dist[neighbor] = path
                # Die aktuelle Node als vorherige Node mit dem kürzesten Weg zur neuen Node setzen 
                pred_node[neighbor] = current_node
                # if we haven't visited the neighbor
                if neighbor not in visited:
                    # insert into priority queue
                    priority_queue.put((dist[neighbor], neighbor))
                # otherwise update the entry in the priority queue
                else:
                    # remove old
                    _ = priority_queue.get((dist[neighbor], neighbor))
                    # insert new
                    priority_queue.put((dist[neighbor], neighbor))
    print(f"Distance: {dist[end]}" )
    print(f"Visited: {len(visited)}")
    return backtrace(pred_node, start, end), dist[end]

def bidirectional_backtrace(pred_node_source, pred_node_target, start, intersection , end):
    
    # From intersection to start
    node = intersection
    path = []
    while node != start:
        path.append(node)
        node = pred_node_source[node]
    path.append(node) 
    path.reverse()

    # From intersection to end
    node = intersection
    while node != end:
        if node != intersection:
            path.append(node)
        node = pred_node_target[node]
    path.append(node)

    return path

# Bidirectional Dijkstra 
def bidirectional_dijkstra(graph, start, end):

    ## Initialisierung
    # Vorherige Node mit bisher kürzestem Weg zu aktueller Node 
    pred_node_source = {}
    pred_node_target = {}
    # Distanzen initialisieren für alle Nodes inf (unendlich) setzen, außer für unsere Start-Node hier wird 0 gesetzt
    dist_source = {v: inf for v in list(nx.nodes(graph))} 
    dist_source[start] = 0
    dist_target = {v: inf for v in list(nx.nodes(graph))} 
    dist_target[end] = 0
    
    # Bereits besuchte Nodes als Set
    visited_source = set()
    visited_target = set() 
    
    # prioritize nodes from start -> node with the shortest distance!
    ## elements stored as tuples (distance, node) 
    priority_queue_source = PriorityQueue()  
    priority_queue_source.put((dist_source[start], start))

    priority_queue_target = PriorityQueue()  
    priority_queue_target.put((dist_target[end], end))
    
    # Solange
    while len(visited_source.intersection(visited_target)) == 0:
        
        current_node_source_cost, current_source_node = priority_queue_source.get()
        current_node_target_cost, current_target_node = priority_queue_target.get()
        
        # Aktuell besuchte Node zu den besuchten hinzufügen
        visited_source.add(current_source_node)
        visited_target.add(current_target_node)

        # Source: Nachfolger der aktuellen Node abrufen und für jeden Nachfolger die Distanz berechnen
        for neighbor in dict(graph.adjacency()).get(current_source_node):
            path = dist_source[current_source_node] + graph.get_edge_data(current_source_node, neighbor).get('length')

            # Wenn die neue gefundene Distanz über den aktuellen Knoten kürzer ist als die bisher kürzeste Distanz
            if path < dist_source[neighbor]:
                # Die neue gefundene Distanz als kürzeste Distanz speichern
                dist_source[neighbor] = path
                # Die aktuelle Node als vorherige Node mit dem kürzesten Weg zur neuen Node setzen 
                pred_node_source[neighbor] = current_source_node
                # if we haven't visited the neighbor
                if neighbor not in visited_source:
                    # insert into priority queue
                    priority_queue_source.put((dist_source[neighbor], neighbor))
                # otherwise update the entry in the priority queue
                else:
                    # remove old
                    _ = priority_queue_source.get((dist_source[neighbor], neighbor))
                    # insert new
                    priority_queue_source.put((dist_source[neighbor], neighbor))


        # Target: Nachfolger der aktuellen Node abrufen und für jeden Nachfolger die Distanz berechnen
        for neighbor in graph.predecessors(current_target_node):
            path = dist_target[current_target_node] + graph.get_edge_data(neighbor, current_target_node).get('length')

            # Wenn die neue gefundene Distanz über den aktuellen Knoten kürzer ist als die bisher kürzeste Distanz
            if path < dist_target[neighbor]:
                # Die neue gefundene Distanz als kürzeste Distanz speichern
                dist_target[neighbor] = path
                # Die aktuelle Node als vorherige Node mit dem kürzesten Weg zur neuen Node setzen 
                pred_node_target[neighbor] = current_target_node
                # if we haven't visited the neighbor
                if neighbor not in visited_target:
                    # insert into priority queue 
                    priority_queue_target.put((dist_target[neighbor], neighbor))
                # otherwise update the entry in the priority queue
                else:
                    # remove old
                    _ = priority_queue_target.get((dist_target[neighbor], neighbor))
                    # insert new
                    priority_queue_target.put((dist_target[neighbor], neighbor))
    
    print(f"Visited Forward Search: {len(visited_source)}")
    print(f"Visited Backwards Search: {len(visited_target)}")
    intersection = visited_source.intersection(visited_target).pop()
    print(f"Intersection: {intersection}")
    print(f"Distance: {dist_target[intersection] + dist_source[intersection]}")
    return bidirectional_backtrace(pred_node_source, pred_node_target, start, intersection, end), dist_target[intersection] + dist_source[intersection]

# A-Star Dijkstra 
def a_star_dijkstra(graph, start, end):

    #Initialisierung
    dest_point = np.array((graph.nodes[end]['y'], graph.nodes[end]['x']))

    pred_node = {} 
    dist = {v: inf for v in list(nx.nodes(graph))}
    dist[start] = 0
    visited = set() 
    priority_queue = PriorityQueue()  
    priority_queue.put((dist[start], start))
    current_node = start

    while current_node != end: 

        current_node_cost, current_node = priority_queue.get()
        visited.add(current_node)
        for neighbor in dict(graph.adjacency()).get(current_node):

            # Näherung der Luftlinie bis zu Zielknoten berechnen 
            neighbor_point = np.array((graph.nodes[neighbor]['y'], graph.nodes[neighbor]['x']))
            dist_to_end = np.linalg.norm(neighbor_point - dest_point)  * 100000

            # Näherung der Luftlinie bei Distanzberechnung einbeziehen
            # Knoten, die von dem Zielknoten weiter weg liegen, werden damit bestraft
            path = dist[current_node] + graph.get_edge_data(current_node,neighbor).get('length') + dist_to_end

            if path < dist[neighbor]:
                dist[neighbor] = path
                pred_node[neighbor] = current_node
                if neighbor not in visited:
                    priority_queue.put((dist[neighbor], neighbor))
                else:
                    remove = priority_queue.get((dist[neighbor], neighbor))
                    priority_queue.put((dist[neighbor], neighbor))

                    
    print(f"Distance: {dist[end]}" )
    print(f"Visited: {len(visited)}")
    return backtrace(pred_node, start, end), dist[end]

# Bidirectional A-Star Dijkstra 
def bidirectional_a_star_dijkstra(graph, start, end):

    ## Initialisierung
    # Koordinaten des Start- und Endpunktes speichern für die Berechnung der Luftlinie
    dest_point = np.array((graph.nodes[end]['y'], graph.nodes[end]['x']))
    orig_point = np.array((graph.nodes[start]['y'], graph.nodes[start]['x']))

    pred_node_source = {}
    dist_source = {v: inf for v in list(nx.nodes(graph))} 
    dist_source[start] = 0
    visited_source = set() 
    priority_queue_source = PriorityQueue()  
    priority_queue_source.put((dist_source[start], start))

    pred_node_target = {}
    dist_target = {v: inf for v in list(nx.nodes(graph))} 
    dist_target[end] = 0
    visited_target = set() 
    priority_queue_target = PriorityQueue()  
    priority_queue_target.put((dist_target[end], end))

    while len(visited_source.intersection(visited_target)) == 0:
        
        # Vorwärtssuche: Nachfolger der aktuellen Node abrufen und für jeden Nachfolger die Distanz berechnen
        current_node_source_cost, current_source_node = priority_queue_source.get()
        visited_source.add(current_source_node)
        for neighbor in dict(graph.adjacency()).get(current_source_node):
            
            neighbor_point = np.array((graph.nodes[neighbor]['y'], graph.nodes[neighbor]['x']))
            dist_to_end = np.linalg.norm(neighbor_point - dest_point)  * 100000
            
            path = dist_source[current_source_node] + graph.get_edge_data(current_source_node, neighbor).get('length') + dist_to_end

            if path < dist_source[neighbor]:
                dist_source[neighbor] = path
                pred_node_source[neighbor] = current_source_node
                if neighbor not in visited_source:
                    priority_queue_source.put((dist_source[neighbor], neighbor))
                else:
                    _ = priority_queue_source.get((dist_source[neighbor], neighbor))
                    priority_queue_source.put((dist_source[neighbor], neighbor))

        # Rückwärtssuche: Vorgänger der aktuellen Node abrufen und für jeden Nachfolger die Distanz berechnen
        current_node_target_cost, current_target_node = priority_queue_target.get()
        visited_target.add(current_target_node)
        for neighbor in graph.predecessors(current_target_node):
            neighbor_point = np.array((graph.nodes[neighbor]['y'], graph.nodes[neighbor]['x']))
            dist_to_end = np.linalg.norm(neighbor_point - orig_point)  * 100000
            
            path = dist_target[current_target_node] + graph.get_edge_data(neighbor, current_target_node).get('length')

            if path < dist_target[neighbor]:
                dist_target[neighbor] = path
                pred_node_target[neighbor] = current_target_node
                if neighbor not in visited_target:
                    priority_queue_target.put((dist_target[neighbor], neighbor))
                else:
                    _ = priority_queue_target.get((dist_target[neighbor], neighbor))
                    priority_queue_target.put((dist_target[neighbor], neighbor))

    
    print(f"Visited Forward Search: {len(visited_source)}")
    print(f"Visited Backwards Search: {len(visited_target)}")
    intersection = visited_source.intersection(visited_target).pop()
    print(f"Intersection: {intersection}")
    print(f"Distance: {dist_target[intersection] + dist_source[intersection]}")
    return bidirectional_backtrace(pred_node_source, pred_node_target, start, intersection, end), dist_target[intersection] + dist_source[intersection]

if __name__ == "__main__":

    base_graph, orig_node, dest_node, leveling_graph = prepare_test_environment()

    print("Time Dijkstra ...")
    time_route_dijkstra = timeit.repeat("route_dijkstra(base_graph, orig_node, dest_node)", repeat=5, number=1, globals=locals())
    print(f"Time dijkstra: {mean(time_route_dijkstra)} s, standard deviation {np.std(time_route_dijkstra)}")
    
    print("Time bidirectional Dijkstra ...")
    time_route_bidir_dijkstra = timeit.repeat("route_bidir_dijkstra(base_graph, orig_node, dest_node)", repeat=5, number=1, globals=locals())
    print(f"Time bidirectional dijkstra: {mean(time_route_bidir_dijkstra)} s, standard deviation {np.std(time_route_bidir_dijkstra)}")

    print("Time a-star Dijkstra ...")
    time_route_a_star_dijkstra = timeit.repeat("route_a_star_dijkstra(base_graph, orig_node, dest_node)", repeat=5, number=1, globals=locals())
    print(f"Time a-star dijkstra: {mean(time_route_a_star_dijkstra)} s, standard deviation {np.std(time_route_a_star_dijkstra)}")

    print("Time a-star bidirectional Dijkstra ...")
    time_route_bidir_a_star_dijkstra = timeit.repeat("route_bidir_a_star_dijkstra(base_graph, orig_node, dest_node)", repeat=5, number=1, globals=locals())
    print(f"Time a-star bidirectional dijkstra: {mean(time_route_bidir_a_star_dijkstra)} s, standard deviation {np.std(time_route_bidir_a_star_dijkstra)}")

    print("Time leveling Dijkstra ...")
    time_route_level_dijkstra = timeit.repeat("route_level_dijkstra(leveling_graph, orig_node, dest_node)", repeat=5, number=1, globals=locals())
    print(f"Time leveling dijkstra: {mean(time_route_level_dijkstra)} s, standard deviation {np.std(time_route_level_dijkstra)}")

    print("Time leveling bidirectional Dijkstra ...")
    time_route_level_bidir_dijkstra = timeit.repeat("route_level_bidir_dijkstra(leveling_graph, orig_node, dest_node)", repeat=5, number=1, globals=locals())
    print(f"Time leveling bidirectional dijkstra: {mean(time_route_level_bidir_dijkstra)} s, standard deviation {np.std(time_route_level_bidir_dijkstra)}")

    print("Time leveling a-star Dijkstra ...")
    time_route_level_a_star_dijkstra = timeit.repeat("route_level_a_star_dijkstra(leveling_graph, orig_node, dest_node)", repeat=5, number=1, globals=locals())
    print(f"Time leveling a-star dijkstra: {mean(time_route_level_a_star_dijkstra)} s, standard deviation {np.std(time_route_level_a_star_dijkstra)}")

    print("Time leveling a-star bidirectional Dijkstra ...")
    time_route_level_bidir_a_star_dijkstra = timeit.repeat("route_level_bidir_a_star_dijkstra(leveling_graph, orig_node, dest_node)", repeat=5, number=1, globals=locals())
    print(f"Time leveling a-star bidirectional dijkstra: {mean(time_route_level_bidir_a_star_dijkstra)} s, standard deviation {np.std(time_route_level_bidir_a_star_dijkstra)}")
