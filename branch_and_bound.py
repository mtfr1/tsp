import numpy as np

def first_min(adj, i):
	minimo = np.inf
	for k in range(len(adj)):
		if(adj[i][k] < minimo and i != k):
			minimo = adj[i][k]
	return minimo

def second_min(adj, i):
	first, second = np.inf, np.inf
	for j in range(len(adj)):
		if i == j:
			continue
		if adj[i][j] <= first:
			second = first
			first = adj[i][j]
		elif adj[i][j] <= second and adj[i][j] != first:
			second = adj[i][j]
	return second

def tsp_rec(adj, curr_bound, curr_weight, level, curr_path, visited, final_res, final_path):
	N = len(adj)

	if level == N:

		if adj[curr_path[level-1]][curr_path[0]] != 0:
			
			curr_res = curr_weight + adj[curr_path[level-1]][curr_path[0]]

			if curr_res < final_res:
				final_path = curr_path
				final_res = curr_res
		print(final_res)
		print(final_path)
		return
	for i in range(N):

		if adj[curr_path[level-1]][i] != 0 and visited[i] == 0:

			temp = curr_bound
			curr_weight += adj[curr_path[level-1]][i]

			if (level == 1):
				curr_bound -= ((first_min(adj, curr_path[level-1]) + first_min(adj, i))/2)

			else:
				curr_bound -= ((second_min(adj, curr_path[level-1]) + first_min(adj, i))/2)

			if curr_bound + curr_weight < final_res:
				curr_path[level] = i; 
				visited[i] = 1;
				tsp_rec(adj, curr_bound, curr_weight, level+1, curr_path, visited, final_res, final_path);  

			curr_weight -= adj[curr_path[level-1]][i]
			curr_bound = temp

			visited = np.zeros(N+1)
			for j in range(level-1):
				visited[curr_path[j]] = 1

def TSP(adj):
	final_res = np.inf
	final_path = []
	N = len(adj)
	curr_path = np.zeros(N+1)
	for i in range(len(curr_path)):
		curr_path[i] = -1

	curr_path = curr_path.tolist()
	visited = np.zeros(N+1)
	visited = visited.tolist()
	curr_bound = 0

	for i in range(N):
		curr_bound = (first_min(adj, i) + second_min(adj, i))
	
	print(curr_bound & 1)

	curr_bound = curr_bound//2 + 1 if curr_bound&1 else curr_bound/2
	visited[0] = 1
	curr_path[0] = 0
	
	tsp_rec(adj, curr_bound, 0, 1, curr_path, visited, final_res, final_path)

adj = [[0, 10, 15, 20], 
       [10, 0, 35, 25], 
       [15, 35, 0, 30], 
       [20, 25, 30, 0]]

TSP(adj)