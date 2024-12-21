import time
import networkx as nx
import matplotlib.pyplot as plt
from graph_init import blocks_graph, add_weight_edge
from execute_brutforce import execute_brut, solveur, get_weight_path
from execute_lagrangienne import get_min_path, execute_lagrangienne

NB_BLOCKS = 60
NB_MAX_COUT = 20
NB_MAX_DUREE = 20
COMPTEUR_FONCTIONS = 0
PARETO_LAGRANGE = 0
PARETO_EPSILON = 0

BLOCK_SPACING = 1

len_pareto_lagrange = []
len_pareto_brut = []

G = blocks_graph(NB_BLOCKS, BLOCK_SPACING)
weights = add_weight_edge(G, NB_MAX_COUT, NB_MAX_DUREE)
edge_labels = {edge: weights[idx] for idx, edge in enumerate(G.edges())}


sc, _ = solveur(G, weights, 0)
pc = get_weight_path(G, sc, weights)

sd, _ = solveur(G, weights, 1)
pd = get_weight_path(G, sd, weights) 

strat_contrainte = int(pc[1])

plt_brut, pareto_brut = execute_brut(G, weights, pc, pd)
plt_lagrange, pareto_lagrange = execute_lagrangienne(G, weights, strat_contrainte, pc, pd)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  

brut_couts = [point[0] for point in pareto_brut]
brut_durees = [point[1] for point in pareto_brut]
ax1.scatter(brut_couts, brut_durees, color='blue')
ax1.set_title("Frontière de Pareto Brute force")
ax1.set_xlabel("Coût")
ax1.set_ylabel("Durée")
ax1.grid(True)
nb_sol_brut = len(pareto_brut)
ax1.text(
    min(brut_couts),  # Position en x
    max(brut_durees),  # Position en y
    f"{nb_sol_brut} solutions",
    fontsize=10,
    color="black",
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')  # Encadré
)

lagrange_couts = [point[0] for point in pareto_lagrange]
lagrange_durees = [point[1] for point in pareto_lagrange]
ax2.scatter(lagrange_couts, lagrange_durees, color='red')
ax2.set_title("Frontière de Pareto Lagrange")
ax2.set_xlabel("Coût")
ax2.set_ylabel("Durée")
ax2.grid(True)
nb_solutions_lagrange = len(pareto_lagrange)
ax2.text(
    min(lagrange_couts),  # Position en x
    max(lagrange_durees),  # Position en y
    f"{nb_solutions_lagrange} solutions",
    fontsize=10,
    color="black",
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')  # Encadré
)

plt.tight_layout()
plt.show()




