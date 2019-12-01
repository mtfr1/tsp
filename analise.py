from tsp import *
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

import time
from memory_profiler import memory_usage

print("1: Christofides x TATT")
print("2: Christofides x TATT x Branch and Bound")
if int(input()) == 1:
    with open('data.csv', 'w') as f:
        f.write('tsp_func,num_verts,length,time,memory,dist_type\n')

    powers = [n for n in range(4, 11)]
    dists = [euclid_dist, manhattan_dist]
    algorithms = [twice_around, christofides]
    algo_str = ["Twice around the tree", "Christofides"]
    dist_str = ["Euclides", "Manhattan"]

    for power in powers:
        points = gen_points_power(power)

        for i in range(len(algorithms)):
            for j in range(len(dists)):
                G = create_graph(points, dists[j])
                
                start = time.time()
                _, value = algorithms[i](G)
                end = time.time()
                total_time = (end - start) / 60
                
                mem = max(memory_usage((algorithms[i], (G,))))
                ans = algo_str[i] + "," + str((2**power)) + "," + str(value) + "," + str(total_time) + "," + str(mem) + "," + dist_str[j] + "\n"
                
                with open('data.csv', 'a') as f:
                    f.write(ans)
else:
    with open('data_full.csv', 'w') as f:
        f.write('tsp_func,num_verts,length,time,memory,dist_type\n')

    n_vertex = [n for n in range(5, 16)]
    dists = [euclid_dist, manhattan_dist]
    algorithms = [twice_around, christofides, branch_and_bound]
    algo_str = ["Twice around the tree", "Christofides", "Branch and Bound"]
    dist_str = ["Euclides", "Manhattan"]
    for n in n_vertex:
        points = gen_points(n)

        for i in range(len(algorithms)):
            for j in range(len(dists)):
                G = create_graph(points, dists[j])
                
                start = time.time()
                _, value = algorithms[i](G)
                end = time.time()
                total_time = (end - start) / 60
                
                mem = max(memory_usage((algorithms[i], (G,))))
                ans = algo_str[i] + "," + str(n) + "," + str(value) + "," + str(total_time) + "," + str(mem) + "," + dist_str[j] + "\n"
                
                with open('data_full.csv', 'a') as f:
                    f.write(ans)