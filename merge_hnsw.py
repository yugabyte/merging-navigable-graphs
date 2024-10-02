#!python
# -*- coding: utf-8 -*-
from tqdm import tqdm
from hnsw import HNSW
import random

def hnsw_general_merge(hnsw_a, hnsw_b, merged_data, layer_merge_func):
    hnsw_merged = HNSW(distance_func=hnsw_a.distance_func, m=hnsw_a._m, m0=hnsw_a._m0, ef=hnsw_a._ef, ef_construction=hnsw_a._ef_construction, neighborhood_construction = hnsw_a.neighborhood_construction)
    hnsw_merged.data = merged_data
    hnsw_merged._graphs = []
    levels_merged = max(len(hnsw_a._graphs), len(hnsw_b._graphs))
    levels_merged_min = min(len(hnsw_a._graphs), len(hnsw_b._graphs))

    if len(hnsw_a._graphs) >= len(hnsw_b._graphs):
        hnsw_merged._enter_point = hnsw_a._enter_point
    else:
        hnsw_merged._enter_point = hnsw_b._enter_point

    for level in range(levels_merged_min): 
        print('Merging level:', level)
        # hnsw_merged._graphs.append(  merge_naive( hnsw_merged.distance_func, hnsw_a, hnsw_b, level, search_ef=20) ) 
        hnsw_merged._graphs.append(  layer_merge_func(hnsw_a, hnsw_b, merged_data, level) ) 

    #TODO 

    return hnsw_merged


def merge_naive(hnsw_a, hnsw_b, merged_data, level, search_ef=5):
    '''
    hnsw_a    – the first hnsw graph 
    hnsw_b    – the second hnsw graph
    level     – mering level number
    search_ef – ef parameter for searching candidates in the second graph
                  
    '''
    m = hnsw_a._m0 if level == 0 else hnsw_a._m
    merged_edges = {}
    for curr_idx in tqdm(hnsw_a._graphs[level].keys()): 
        observed = hnsw_b.search(q=hnsw_a.data[curr_idx], k=m, ef=search_ef, level=level, return_observed=True) #return_observed=True
        # candidates_b = observed[:k]
        candidates_b = observed
        # == build neighborhood for curr_idx and save to externalset of edges  ==
        candidates = [ (idx_b, dist) for idx_b, dist in candidates_b] + [ (idx, dist) for idx, dist in hnsw_a._graphs[level][curr_idx]]
        # merged_edges[curr_idx] = sorted ([ (idx_b + len(kga.data), dist) for idx_b, dist in candidates_b] + [ (idx, dist) for idx, dist in kga.edges[curr_idx]], key=lambda a: a[1])[:k]
        merged_edges[curr_idx] = hnsw_a.neighborhood_construction(candidates, hnsw_a.data[curr_idx], m, hnsw_a.distance_func, merged_data)    
        # == == == == == == == == == == == == == == == == == == == == == == == ==

    for curr_idx in tqdm(hnsw_b._graphs[level].keys()): 
        observed = hnsw_a.search(q=hnsw_b.data[curr_idx], k=m, ef=search_ef, level=level, return_observed=True)
        # candidates_a = observed[:k]
        candidates_a = observed
        # == build neighborhood for curr_idx and save to externalset of edges  ==
        candidates = [(idx_a, dist) for idx_a, dist in candidates_a] + [(idx, dist) for idx, dist in hnsw_b._graphs[level][curr_idx]]
        merged_edges[curr_idx] = hnsw_b.neighborhood_construction(candidates, hnsw_b.data[curr_idx], m, hnsw_a.distance_func, merged_data)
        # == == == == == == == == == == == == == == == == == == == == == == == ==

    return merged_edges


def merge1(hnsw_a, hnsw_b, merged_data, level, jump_ef=1, local_ef=5, next_step_k=1, next_step_ef=5, M = 5):
    '''
    hnsw_a          – first graph 
    hnsw_b          – second graph
    search_ef    - ef parameter for searching candidates in the second graph
    next_step_k  - at each iteration we look for the next element around the current vertex in the first graph. 
                   However it can be surrounded by the "done" vertex, so we have to walk away. 
                   Thus this parameter controls how far from the current vertex we can go.
    next_step_ef – a purpose of this parameter is similar {next_step_k}
    M            – number of point returned by the jump-search                 
    '''
    merged_edges = {}
    not_done = set(hnsw_a._graphs[level].keys())
    m = hnsw_a._m0 if level == 0 else hnsw_a._m
    
    # tqdm progress bar based on the initial size of the `not_done` set
    progress_bar = tqdm(total=len(not_done), desc="Merging progress")

    while not_done:
        # Start with a vertex from `not_done`
        curr_idx = not_done.pop()

        # Perform jump search on graph B
        observed = hnsw_b.search(q=hnsw_a.data[curr_idx], k=M, ef=jump_ef, level=level, return_observed=True)
        staring_points = [idx for idx, dist in observed[:M]]
        
        while True:
            # Perform local search at graph B
            observed.extend(hnsw_b.beam_search(graph=hnsw_b._graphs[level], q=hnsw_a.data[curr_idx], k=m, eps=staring_points, ef=local_ef, return_observed=True))
            candidates_b = observed[:m]

            # Build neighborhood for curr_idx and save to external set of edges
            candidates = [(idx_b, dist) for idx_b, dist in candidates_b] + [(idx, dist) for idx, dist in hnsw_a._graphs[level][curr_idx]]
            merged_edges[curr_idx] = hnsw_a.neighborhood_construction(candidates, merged_data[curr_idx], m, hnsw_a.distance_func, merged_data)

            # Determine new set of entry points for search in hnsw_b
            staring_points = [idx for idx, dist in candidates_b[:m]]

            # Perform local search at graph A to find next candidate
            candidates_a = hnsw_a.beam_search(graph=hnsw_a._graphs[level], q=hnsw_a.data[curr_idx], k=next_step_k, eps=[curr_idx], ef=next_step_ef, return_observed=True)
            candidates_a = [c[0] for c in candidates_a[:next_step_k] if c[0] in not_done]

            if not candidates_a:
                break

            # Move to the next candidate and remove it from `not_done`
            curr_idx = candidates_a[0]
            not_done.remove(curr_idx)
            progress_bar.update(1)
            observed = []

        # Update the progress bar
        progress_bar.update(1)

    progress_bar.close()
    return merged_edges




def layer_merge1_func(hnsw_a, hnsw_b, merged_data, level) :
    merged_edges = {} 
    # phase 1) 
    merged_edges.update(merge1(hnsw_a, hnsw_b, merged_data, level=level, jump_ef=20, local_ef=5, next_step_k=5, next_step_ef=3, M = 5))
    # phase 2)
    merged_edges.update(merge1(hnsw_b, hnsw_a, merged_data,  level=level, jump_ef=20, local_ef=5, next_step_k=5, next_step_ef=3, M = 5))
    return merged_edges


def merge_alg2(hnsw_a, hnsw_b, merged_data, level, jump_ef = 20, local_ef=5, next_step_k=3, M = 3):
    '''
    hnsw_a       – first graph 
    hnsw_b       – second graph
    search_ef    - ef parameter for searching candidates in the second graph
    next_step_k  - at each iteration we look for the next element around the current vertex in the first graph. 
                   However it can be surrounded by the "done" vertex, so we have to walk away. 
                   Thus this parameter controlls how far from the current vertex we can go.
    M            – number of starting random enter points                 
    '''
    merged_edges = {}

    m = hnsw_a._m0 if level == 0 else hnsw_a._m
    
    not_done_a = set( hnsw_a._graphs[level].keys())
    not_done_b = set( hnsw_b._graphs[level].keys())
    not_done = not_done_a.union( [i  for i in not_done_b] )

    progress_bar = tqdm(total=len(not_done), desc="Merging progress")
    
    while not_done:
        curr_idx = random.choice(list(not_done))
    
        # do jump search   
        observed_jump_a = hnsw_a.search(q=merged_data[curr_idx], k=M, ef=jump_ef, level=level, return_observed=True) #return_observed=True
        observed_jump_b = hnsw_b.search(q=merged_data[curr_idx], k=M, ef=jump_ef, level=level, return_observed=True) #return_observed=True

        enter_points_a = [idx for idx, dist in observed_jump_a[:M]]
        enter_points_b = [idx for idx, dist in observed_jump_b[:M]]
        while True:
            not_done.remove(curr_idx) # remove from not_done
            progress_bar.update(1)
            # searching for a new current

            # do local serach at graph A. # decrease k to traverse closer to curr vertex
            observed_a = hnsw_a.beam_search(graph=hnsw_a._graphs[level], q=merged_data[curr_idx], k=m, eps=enter_points_a, ef=local_ef, return_observed=True) 
            # do local serach at graph A. # decrease k to traverse closer to curr vertex
            observed_b = hnsw_b.beam_search(graph=hnsw_b._graphs[level], q=merged_data[curr_idx], k=m, eps=enter_points_b, ef=local_ef, return_observed=True)
            
            candidates_a = observed_a[:m]
            candidates_b = observed_b[:m]

            # --== build neighborhood for new_curr_idx and save to externalset of edges  ==--
            if curr_idx < len(hnsw_a.data):                
                candidates  = hnsw_a._graphs[level][curr_idx] + [ (idx_b, dist) for idx_b, dist in candidates_b]
            else:                
                candidates =  candidates_a + [(idx_b, dist) for idx_b, dist in hnsw_b._graphs[level][curr_idx]]                     
            merged_edges[curr_idx] = hnsw_a.neighborhood_construction(candidates, merged_data[curr_idx], m, hnsw_a.distance_func, merged_data)                
            # --== build neighborhood for new_curr_idx and save to externalset of edges  ==--
                  
            candidates_a_not_done = [ (idx, dist) for idx, dist in observed_a if idx in not_done]
            candidates_b_not_done = [ (idx , dist) for idx, dist in observed_b if idx in not_done]
            
            candidates_not_done = [candidates_a_not_done[0]] if len(candidates_a_not_done) > 0 else [] + [candidates_b_not_done[0]] if len(candidates_b_not_done) > 0 else []

            if not candidates_not_done: 
                break #jump to the random point

            new_curr = min(candidates_not_done, key=lambda a: a[1])
            new_curr_idx = new_curr[0]
                        
            curr_idx = new_curr_idx
            enter_points_a = [idx for idx, dist in candidates_a]
            enter_points_b = [idx for idx, dist in candidates_b]
            observed_jump_a = []
            observed_jump_b = []
    return merged_edges
                           

def layer_merge2_func(hnsw_a, hnsw_b, merged_data, level) :
    return merge_alg2(hnsw_a=hnsw_a, hnsw_b=hnsw_b, merged_data=merged_data, level=level, jump_ef=20, local_ef=5, next_step_k=5, M = 5)