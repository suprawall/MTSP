import time
import networkx as nx
import matplotlib.pyplot as plt
from graph_init import blocks_graph, add_weight_edge
from execute_epsilon import execute_epsilon, solveur, get_weight_path
from execute_lagrangienne import get_min_path, execute_lagrangienne

NB_BLOCKS = 50
NB_MAX_COUT = 20
NB_MAX_DUREE = 20
COMPTEUR_FONCTIONS = 0
PARETO_LAGRANGE = 0
PARETO_EPSILON = 0

BLOCK_SPACING = 1


G = blocks_graph(NB_BLOCKS, BLOCK_SPACING)
weights = add_weight_edge(G, NB_MAX_COUT, NB_MAX_DUREE)
edge_labels = {edge: weights[idx] for idx, edge in enumerate(G.edges())}


sc, _ = solveur(G, weights, 0)
pc = get_weight_path(G, sc, weights)

sd, _ = solveur(G, weights, 1)
pd = get_weight_path(G, sd, weights) 

max_duree = int((pc[1] + pd[1]) / 2)
    

plt_epsilon, pareto_epsilon = execute_epsilon(G, weights, pc, pd)
plt_lagrange, pareto_lagrange = execute_lagrangienne(G, weights, max_duree, pc, pd)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  

epsilon_couts = [point[0] for point in pareto_epsilon]
epsilon_durees = [point[1] for point in pareto_epsilon]
ax1.scatter(epsilon_couts, epsilon_durees, color='blue')
ax1.set_title("Frontière de Pareto Epsilon")
ax1.set_xlabel("Coût")
ax1.set_ylabel("Durée")
ax1.grid(True)

lagrange_couts = [point[0] for point in pareto_lagrange]
lagrange_durees = [point[1] for point in pareto_lagrange]
ax2.scatter(lagrange_couts, lagrange_durees, color='red')
ax2.set_title("Frontière de Pareto Lagrange")
ax2.set_xlabel("Coût")
ax2.set_ylabel("Durée")
ax2.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
epsilon_couts = [point[0] for point in pareto_epsilon]
epsilon_durees = [point[1] for point in pareto_epsilon]
plt.scatter(epsilon_couts, epsilon_durees, color='blue', label='Pareto Epsilon')

lagrange_couts = [point[0] for point in pareto_lagrange]
lagrange_durees = [point[1] for point in pareto_lagrange]
plt.scatter(lagrange_couts, lagrange_durees, color='red', label='Pareto Lagrange')

plt.title("Frontières de Pareto")
plt.xlabel("Coût")
plt.ylabel("Durée")
plt.legend()
plt.grid(True)
plt.show()

print("Pareto epsilon: "+str(pareto_epsilon))
print("Pareto lagrange: "+str(pareto_lagrange))
