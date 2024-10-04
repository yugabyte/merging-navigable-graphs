#!python3
import numpy as np
import random
from math import log2
from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
import json

def l2_distance(a, b):
    return np.linalg.norm(a - b)

def heuristic(candidates, curr, k, distance_func, data):
    '''
    candidates - list of candidate vertexes with the distances to the curr [(v_1, d_1), (v_2, d_2),..., (v_n, d_n)].
    k – maximum size of the neighborhood 
    distance_func – distance function for the distance caluclation between vertexes
    returns the neighborhood for the vertex curr [(neighbour_1, dist_1), (neighbour_2, dist_2), ..., (neighbour_k, dist_k)]
    data – the object for obtaining vertex content by its id.  it can be an array or a dictonary
    '''
    candidates = sorted(candidates, key=lambda a: a[1])
    result_indx_set = {candidates[0][0]}
    result = [candidates[0]]
    added_data = [ data[candidates[0][0]] ]
    for c, curr_dist in candidates[1:]:
        c_data = data[c]       
        if curr_dist < min(map(lambda a: distance_func(c_data, a), added_data)):
            result.append( (c, curr_dist))
            result_indx_set.add(c)
            added_data.append(c_data)
    for c, curr_dist in candidates: # optional. uncomment to build neighborhood exactly with k elements.
        if len(result) < k and (c not in result_indx_set):
            result.append( (c, curr_dist) )
    
    return result
def k_closest(candidates: list, curr, k, distance_func, data):
    return sorted(candidates, key=lambda a: a[1])[:k]
    
class HNSW:
    # self._graphs[level][i] contains a {j: dist} dictionary,
    # where j is a neighbor of i and dist is distance
    
    def vectorized_distance(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def __init__(self, distance_func, m=5, ef=10, ef_construction=30, m0=None, neighborhood_construction=heuristic):
        self.data = {}
        self.distance_func = distance_func
        self.neighborhood_construction = neighborhood_construction

        self._m = m
        self._ef = ef
        self._ef_construction = ef_construction
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / log2(m)
        self._graphs = []
        self._enter_point = None

    def add(self, key, elem):
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m

        # level at which the element will be inserted
        level = int(-log2(random.random()) * self._level_mult) + 1
        # print("level: %d" % level)

        # elem will be at data[idx]
        idx = key
        data[key] = elem

        if point is not None:  # the HNSW is not empty, we have an entry point
            dist = self.distance_func(elem, data[point])
            # for all levels in which we dont have to insert elem,
            # we search for the closest neighbor
            for layer in reversed(graphs[level:]):
                point, dist = self.beam_search(graph=layer, q=elem, k=1, eps=[point], ef=1)[0]
            # at these levels we have to insert elem; ep is a heap of entry points.

            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0
                # navigate the graph and update ep with the closest
                # nodes we find
                # ep = self._search_graph(elem, ep, layer, ef)
                candidates = self.beam_search(graph=layer, q=elem, k=level_m*2, eps=[point], ef=self._ef_construction)
                point = candidates[0][0]
                
                # insert in g[idx] the best neighbors
                # layer[idx] = layer_idx = {}
                # self._select(layer_idx, ep, level_m, layer, heap=True)

                neighbors = self.neighborhood_construction(candidates=candidates, curr=idx, k=level_m, distance_func=self.distance_func, data=self.data)
                layer[idx] = neighbors
                # insert backlinks to the new node
                for j, dist in neighbors:
                    candidates_j = layer[j] + [(idx, dist)]
                    neighbors_j = self.neighborhood_construction(candidates=candidates_j, curr=j, k=level_m, distance_func=self.distance_func, data=self.data)
                    layer[j] = neighbors_j
                    
                
        for i in range(len(graphs), level):
            # for all new levels, we create an empty graph
            graphs.append({idx: []})
            self._enter_point = idx
            
    # can be used for search after jump        
    def search(self, q, k=1, ef=10, level=0, return_observed=True):
        graphs = self._graphs
        point = self._enter_point
        for layer in reversed(graphs[level:]):
            point, dist = self.beam_search(layer, q=q, k=1, eps=[point], ef=1)[0]

        return self.beam_search(graph=graphs[level], q=q, k=k, eps=[point], ef=ef, return_observed=return_observed)

    def beam_search(self, graph, q, k, eps, ef, return_observed=False):
        '''
        graph – the layer where the search is performed
        q - query
        k - number of closest neighbors to return
        eps – entry points [vertex_id, ..., vertex_id]
        ef – size of the beam
        observed – if True returns the full of elements for which the distance were calculated
        returns – a list of tuples [(vertex_id, distance), ... , ]
        '''
        # Priority queue: (negative distance, vertex_id)
        candidates = []
        visited = set()  # set of vertex used for extending the set of candidates
        observed = dict() # dict: vertex_id -> float – set of vertexes for which the distance were calculated

        # Initialize the queue with the entry points
        if type(eps[0]) is tuple:
            for ep in eps:
                heappush(candidates, (ep[1], ep[0]))
                observed[ep[0]] = ep[1]   
        else:
            for ep in eps:
                dist = self.distance_func(q, self.data[ep])
                heappush(candidates, (dist, ep))    
                observed[ep] = dist

        while candidates:
            # Get the closest vertex (furthest in the max-heap sense)
            dist, current_vertex = heappop(candidates)

            # check stop conditions #####
            observed_sorted = sorted( observed.items(), key=lambda a: a[1] ) # TODO fix checking the stoping condition. Just find the ef_largest element
            # print(observed_sorted)
            ef_largets = observed_sorted[ min(len(observed)-1, ef-1 ) ]
            # print(ef_largets[0], '<->', -dist)
            if ef_largets[1] < dist:
                break
            #############################

            # Add current_vertex to the visited set
            visited.add(current_vertex)

            # Check neighbors of current vertex
            for neighbor, _ in graph[current_vertex]:
                if neighbor not in observed:
                    dist = self.distance_func(q, self.data[neighbor])                    
                    # if neighbor not in visited:
                    heappush(candidates, (dist, neighbor))
                    observed[neighbor] = dist                    
                    
        # Sort the results by distance and return top-k
        if return_observed:
            observed_sorted = sorted( observed.items(), key=lambda a: a[1] )
            return observed_sorted
        # TODO: Replace sorting by supporing sorted order only for k(ef)-first elements with the smallest distance
        observed_sorted =sorted( observed.items(), key=lambda a: a[1] )
        return observed_sorted[:k]
    
    def save_graph_plane(self, file_path):
        with open(file_path, "w") as f:
            f.write(f'{len(self.data)}\n')

            for x in self.data:
                s = ' '.join([a.astype('str') for a in x ])
                f.write(f'{s}\n')

            for graph in self._graphs:
                for src, neighborhood in graph.items():
                    for dst, dist in neighborhood: 
                        f.write(f'{src} {dst}\n')

    def save(self, file_path):
        def convert_np_float32(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            return obj
        with open(file_path, "w") as f:
            f.write(json.dumps( {key: x.tolist() for key, x in self.data.items() }   )) 
            f.write('\n')
            f.write(json.dumps(self._graphs, default=convert_np_float32))

    def load(self, file_path):
        with open(file_path, "r") as f:
            # First part is hnsw.data
            hnsw_data_str = f.readline().strip()
            hnsw_data = json.loads(hnsw_data_str)
            self.data = {int(key): np.array(value) for key, value in hnsw_data.items()}

            # Second part is hnsw._graphs
            hnsw_graphs_str = f.readline().strip()
            hnsw_graphs = json.loads(hnsw_graphs_str)
            self._graphs =  [  {int(v):neighbor for v, neighbor in graph.items() }  for graph in hnsw_graphs]
            # self._graphs = ra for graph in hnsw_graphs]
            self._enter_point = list(self._graphs[-1].keys())[0]
                