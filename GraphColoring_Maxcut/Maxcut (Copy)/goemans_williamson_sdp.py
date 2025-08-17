# The polynomial-time approximation algorithm for Max-Cut with the best known approximation ratio 
# is a method by Goemans and Williamson using semidefinite programming 
# We shall implement this algorithm using the cvxpy library
import cvxpy as cp
import numpy as np

def goemans_williamson_max_cut(G):
    # Number of nodes
    n = G.number_of_nodes()

    # Create a semidefinite variable for SDP relaxation
    # V will be the matrix of dot products, i.e., V[i, j] = v_i^T v_j
    V = cp.Variable((n, n), symmetric=True)
    
    # Objective: maximize sum of edge weights * (1 - <v_i, v_j>) / 2
    objective = cp.Maximize(
        cp.sum([1 * (1 - V[u, v]) / 2 for u, v in G.edges()])  # Initial 1 is the assumed weight of each edge
    )

    # Constraints: V should be positive semidefinite and diagonal elements must be 1
    constraints = [V >> 0]  # Positive semidefinite constraint
    constraints += [V[i, i] == 1 for i in range(n)]  # Diagonal elements must be 1 as v_i^T v_i = 1 (dot product of a vector with itself)

    # Define and solve the SDP
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Extract the matrix V
    V_value = V.value

    # Step 2: Randomly choose a hyperplane to perform rounding
    random_vector = np.random.randn(n)

    # Perform rounding
    colors = [0 if np.dot(V_value[i, :], random_vector) >= 0 else 1 for i in range(n)]
    # Note: Cholesky Decomposition is not necessary. We can use rows of V directly with the random vector
    # V_ij = u_i.u_j
    # u_i = {u_i1, u_i2, ..., u_in}
    # V_ij = u_i1.u_j1 + u_i2.u_j2 + ... + u_in.u_jn
    # random_vector = r = {r_1, r_2, ..., r_n}
    # V_[i,:].r =  u_i1.u_11.r_1 + u_i2.u_12.r_1 + ... + u_in.u_1n.r_1
    #            + u_i1.u_21.r_2 + u_i2.u_22.r_2 + ... + u_in.u_2n.r_2
    #            + u_i1.u_31.r_3 + u_i2.u_32.r_3 + ... + u_in.u_3n.r_3
    #            + ...
    #            + u_i1.u_n1.r_n + u_i2.u_n2.r_n + ... + u_in.u_nn.r_n
    #            = u_i1.(u_11.r_1 + u_21.r_2 + ... + u_n1.r_n) + u_i2.(u_12.r_1 + u_22.r_2 + ... + u_n2.r_n) + ... + u_in.(u_1n.r_1 + u_2n.r_2 + ... + u_nn.r_n)
    #           = u_i1.r'_1 + u_i2.r'_2 + ... + u_in.r'_n
    #           = u_i.r'
    # where r' = {r'_1, r'_2, ..., r'_n} and r'_i = u_1i.r_1 + u_2i.r_2 + ... + u_ni.r_n


    # Compute the value of the cut
    cut_value = sum([1 for u, v in G.edges() if colors[u] != colors[v]])  # 1 is the assumed weight of each edge

    # Return the cut, which is a partition of the vertices
    return colors, cut_value


