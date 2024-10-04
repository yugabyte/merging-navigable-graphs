#!python
# -*- coding: utf-8 -*-

from hnsw import HNSW
from hnsw import heuristic
import numpy as np
from datasets import load_sift_dataset, calculate_recall
from merge_hnsw import merge2
import os.path
import pandas as pd 

distance_count = 0
def l2_distance(a, b):
    global distance_count
    distance_count+=1
    return np.linalg.norm(a - b)

result_file = 'result.csv'

k=5
efs=[32,40,50,64,72]
exp = {}

merge_params_list = [        
    {'jump_ef':20, 'local_ef':5, 'next_step_k':3, 'M':1},
    {'jump_ef':20, 'local_ef':5, 'next_step_k':3, 'M':2},
    {'jump_ef':20, 'local_ef':5, 'next_step_k':3, 'M':3},
    {'jump_ef':20, 'local_ef':5, 'next_step_k':3, 'M':20},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'M':1},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'M':2},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'M':3},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'M':5},
    {'jump_ef':64, 'local_ef':5, 'next_step_k':3, 'M':10},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'M':1},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'M':2},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'M':3},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'M':5},
    {'jump_ef':64, 'local_ef':10, 'next_step_k':3, 'M':10},
]


hnsw_a = HNSW( distance_func=l2_distance, m=5, m0=7, ef=10, ef_construction=30,  neighborhood_construction = heuristic)
hnsw_b = HNSW( distance_func=l2_distance, m=5, m0=7, ef=10, ef_construction=30,  neighborhood_construction = heuristic)

hnsw_a.load('save/sift1m/hnsw_a.txt')
hnsw_b.load('save/sift1m/hnsw_b.txt')

merged_data = hnsw_a.data.copy()
merged_data.update(hnsw_b.data)

_, test_data, groundtruth_data = load_sift_dataset(train_file = None,
                                                      test_file='datasets/sift1m-128d/sift_query.fvecs',
                                                      groundtruth_file='datasets/sift1m-128d/sift_groundtruth.ivecs')

for merge_params in merge_params_list:
    exp['params'] = merge_params
    distance_count = 0
    hnsw_merged2 = merge2(hnsw_a, hnsw_b, merged_data, 
                          jump_ef=merge_params['jump_ef'], 
                          local_ef=merge_params['local_ef'], 
                          next_step_k=merge_params['next_step_k'],
                          M=merge_params['M'])
    exp['merge distance count'] = distance_count

    for ef in efs:
        distance_count = 0
        recall, _ = calculate_recall(hnsw_merged2, test_data, groundtruth=groundtruth_data, k=5, ef=ef)
        exp[f'{k}@recall ef={ef}'] = recall
        exp[f'dist count ef={ef}'] = distance_count/len(test_data)

    df = pd.DataFrame([exp])
    if os.path.isfile(result_file):
        df.to_csv(result_file, mode='a', index=False, header=False)
    else:
        df.to_csv(result_file, mode='w', index=False, header=True)