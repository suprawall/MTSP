import time
import networkx as nx
from graph_init import blocks_graph, add_weight_edge
from execute_epsilon import execute_epsilon, solveur, get_weight_path
from execute_lagrangienne import get_min_path, execute_lagrangienne

NB_BLOCKS = 40
NB_MAX_COUT = 20
NB_MAX_DUREE = 20
COMPTEUR_FONCTIONS = 0
PARETO_LAGRANGE = 0
PARETO_EPSILON = 0

BLOCK_SPACING = 1


G = blocks_graph(NB_BLOCKS, BLOCK_SPACING)

weights = add_weight_edge(G, NB_MAX_COUT, NB_MAX_DUREE)

edge_labels = {edge: weights[idx] for idx, edge in enumerate(G.edges())}

#paths = list(nx.all_simple_paths(G, source=1, target=max(G.nodes()) - 1))



sc, _ = solveur(G, weights, 0)
pc = get_weight_path(G, sc, weights)

sd, _ = solveur(G, weights, 1)
pd = get_weight_path(G, sd, weights) 

max_duree = int((pc[1] + pd[1]) / 2)
    
plt_epsilon, pareto_epsilon = execute_epsilon(G, weights, pc, pd)
plt_lagrange, pareto_lagrange = execute_lagrangienne(G, weights, max_duree, pc, pd)

plt_epsilon.show()
plt_lagrange.show()

print("Pareto epsilon: "+str(pareto_epsilon))
"""print("creation du graph: "+str(be - bs))
print("ajout des poids: "+str(we - ws))
print("ajout des labels: "+str(els - ele))
print("définition de tous les chemins: "+str(pe - ps))"""
print("Pareto lagrange: "+str(pareto_lagrange))