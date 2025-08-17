import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cProfile

def graph_to_mat(G, path = 'test_graphs/default2.mat'):
    # get the adjancency matrix of the graph
    adj = nx.adjacency_matrix(G).todense()
    adj = np.asarray(adj, dtype=np.int32)
    scipy.io.savemat(path, {'M': adj})   # M for matrix, to stay consistent with the .mat files we already have
    return

def mat_to_graph(path):
    mat = scipy.io.loadmat(path)
    G = nx.from_numpy_array(mat['M'])
    return G

def gen_permutation_matrix(phi):
    n = phi.shape[0]
    order = np.argsort(phi)
    P = np.zeros((n, n))
    for i in range(n):
        P[i,order[i]] = 1  # orignally P[i,order[i]] = 1
    return P, order

def get_chr(P, A, order):
    n = A.shape[0]
    PAPT = (P @ A) @ P.T
    # perm = np.argmax(P, axis=1)
    # PAPT = A[perm][:, perm]  # equivalent to (P @ A) @ P.T
    block_list = []
    color_blocks = []
    i = 0
    while (i<n):
        condition = 0
        block = 0
        color_block = []
        while (condition==0 and (i+block)<n):
            #if PAPT[i:i+block+1,i:i+block+1] == np.zeros((block+1,block+1)):
            if np.sum(np.abs(PAPT[i:i+block+1,i:i+block+1])) == 0:
                color_block.append(order[i+block])
                block = block+1
            else:
                condition = 1
        block_list.append(block)
        color_blocks.append(color_block)
        i = i + block
    return len(block_list), color_blocks

def get_color_indice_array(color_blocks, adj):
    color_indice_array = np.zeros(adj.shape[0], dtype=np.int32)
    temp_color_index = 0
    for color_block in color_blocks:
        for node in color_block:
            color_indice_array[node] = temp_color_index
        temp_color_index += 1
    num_colors = temp_color_index
    return color_indice_array, num_colors

def density(adj_matrix):
    return np.sum(adj_matrix) / (adj_matrix.shape[0] * (adj_matrix.shape[1]-1))

def get_color_adj_graph(color_indice_array, adj, num_colors):
    color_adj = np.zeros((num_colors,num_colors), dtype=np.int32)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if adj[i][j] == 1:
                color_adj[color_indice_array[i]][color_indice_array[j]] = 1
    return color_adj

def loss(phi_history,adj):
    num_nodes = adj.shape[0]
    loss_list = []
    for i in range (len(phi_history)):
        curr_phi = phi_history[i]
        loss = 0
        for j in range (num_nodes):
            for k in range (j):
                if(adj[j][k] == 1):
                    loss -= 1 - np.cos(curr_phi[j] - curr_phi[k])
        loss_list.append(loss)
    return loss_list

def gen_cyclic_graph(n):
    B = np.zeros((n, n))
    for i in range(n):
        B[i,i-1] = 1.0
        B[i,(i+1)%n] = 1.0
    return B

def floyd_warshall_networkx(adj_matrix):
    G = nx.from_numpy_array(adj_matrix)

    length = dict(nx.all_pairs_shortest_path_length(G))
    
    n = adj_matrix.shape[0]
    d = np.full((n, n), np.inf)
    
    for i in range(n):
        for j, dist in length[i].items():
            d[i][j] = dist
    
    return d

def get_attraction_coefficient(adj_matrix):
    dist_temp = floyd_warshall_networkx(adj_matrix)
    dist_coeff = dist_temp-1
    dist_coeff[dist_coeff < 0] = 0
    dist_coeff[dist_temp == np.inf] = 0
    return dist_coeff

def get_attraction_coefficient_binary(adj_matrix):
    dist_temp = floyd_warshall_networkx(adj_matrix)
    dist_coeff = dist_temp-1
    dist_coeff[dist_coeff < 0] = 0
    dist_coeff[dist_temp == np.inf] = 0
    dist_coeff[dist_coeff > 1] = 1
    return dist_coeff