import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value, PULP_CBC_CMD
from execute_brutforce import get_weight_path

from execute_lagrangienne import add_path_function


def get_random_path(G, weights):
    current_node = 1
    path = [current_node]
    path_weight = [0, 0]
    edges = list(G.edges())
    
    while current_node != (max(G.nodes()) - 1):
        neighbors = list(G.successors(current_node))
        next_node = random.choice(neighbors)
        
        edge = (current_node, next_node)
        path_weight[0] += weights[edges.index(edge)][0]  # Coût
        path_weight[1] += weights[edges.index(edge)][1]  # Durée
        
        current_node = next_node
        path.append(current_node)
    
    return (path, path_weight)

def execute_random_path(G, weights, nb):
    for i in range(nb):
        random_path = get_random_path(G, weights)
        cout, sign, duree = add_path_function(random_path)
        
        print("Chemin aléatoire n°"+str(i+1)+" : "+str(random_path[0]))
        print("Coût : "+str(cout)+", Durée : "+str(duree))
        
        # Tracer la fonction affine
        alpha = np.linspace(0, 10, 100)  # Génère 100 valeurs pour alpha
        if sign == 1:
            y = cout + alpha * duree
            label = f"{cout} + α * {duree}"
        else:
            y = cout - alpha * duree
            label = f"{cout} - α * {duree}"
        
        plt.plot(alpha, y, label=label)
        
def crossing_point(tab):
    f1 = tab[0]
    f2 = tab[1]
    a1, b1 = f1
    a2, b2 = f2
    print(f1)
    print(f2)
    return (b2 - b1) / (a1 - a2)