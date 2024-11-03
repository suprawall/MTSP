import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value
from execute_epsilon import get_weight_path



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

def add_path_function(cout, duree, alpha, compt, max_duree):
    d = duree - max_duree
    y = cout + alpha * d
    
    label = f"{compt}: {cout} + α * {d}"
    #return (cout, +1, duree)  # Représente "cout + α * duree"
        
    plt.plot(alpha, y, label=label)
    return d, cout

def crossing_point(tab):
    f1 = tab[0]
    f2 = tab[1]
    a1, b1 = f1
    a2, b2 = f2
    print(f1)
    print(f2)
    return (b2 - b1) / (a1 - a2)

def get_min_path(G, paths, weights,  nv_weights = None, criteria=None):
    """
    Renvoie le chemin minimisant le critère donné.
    criteria = None pour l'étape où on cherche le prochain x
    criteria = 0 pour minimiser le coût
    criteria = 1 pour minimiser la durée
    """

    min_path = None
    min_value = float('inf')

    for path in paths:
        path_value = 0
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if criteria is not None:
                path_value += weights[list(G.edges()).index(edge)][criteria]
            else:
                path_value += nv_weights[list(G.edges()).index(edge)]
        
        if path_value < min_value:
            min_value = path_value
            min_path = path
            
    #calculer la valeur cout,duree du min-path
    copy = min_path.copy()
    tab = []
    w = [0, 0]
    edges = list(G.edges())
    i = 0
    
    while i < len(copy) - 1:
        tab.append((copy[i], copy[i+1]))
        i+=1
        
    for edge in tab:
        w[0] += weights[edges.index(edge)][0]
        w[1] += weights[edges.index(edge)][1]

    print("Sur le critère "+str(criteria)+", le meilleur chemin est "+str(min_path)+" et il est de poid "+str(w))
    
    return min_path, w

def find_max_of_min(fonctions, alpha_values):
    min_values = np.full_like(alpha_values, float('inf'))
    
    for (a, b) in fonctions:
        values = a * alpha_values + b
        min_values = np.minimum(min_values, values)
        
    next_alpha = alpha_values[np.argmax(min_values)]
    
    return next_alpha

def solver_tsp(G, weights):
    edges = list(G.edges())
        
    prob = LpProblem("Minimize_Objectif", LpMinimize)
    var_edges = LpVariable.dicts("Edges", edges, 0, 1, cat='Binary')

    prob += lpSum([var_edges[edge] * weights[i] for i, edge in enumerate(edges)])
    
    source, target = 1, max(G.nodes()) - 1

    for node in G.nodes():
        if node == source:  
            prob += lpSum([var_edges[edge] for edge in G.out_edges(node)]) == 1
        elif node == target:  
            prob += lpSum([var_edges[edge] for edge in G.in_edges(node)]) == 1
        else: 
            prob += lpSum([var_edges[edge] for edge in G.in_edges(node)]) == lpSum([var_edges[edge] for edge in G.out_edges(node)])
    
    prob.solve()
    
    s = [var_edges[edge].varValue for edge in edges]
    solution = []
    for i, e in enumerate(s):
        if(e == 1.0):
            solution.append(edges[i])
    objective_value = value(prob.objective)

    return solution, objective_value
    
    

def execute_lagrangienne(G, weights, max_duree, pc, pd):
    
    plt.figure(figsize=(8, 6))
    
    frontiere_pareto = []
    frontiere_pareto.append(pc)
    frontiere_pareto.append(pd)
    
    parametre_fonctions = []                       #(a1, b1), (a2, b2)....
    alpha_values = np.linspace(-10, 10, 200)
    compteur_fonctions = 1

    parametre_fonctions.append(add_path_function(pc[0], pc[1], alpha_values, compteur_fonctions, max_duree))
    compteur_fonctions += 1
    parametre_fonctions.append(add_path_function(pd[0], pd[1], alpha_values, compteur_fonctions, max_duree))
    compteur_fonctions += 1


    while(True):
        nv_alpha2 = find_max_of_min(parametre_fonctions, alpha_values)
        nv_weights = []
        for cout, duree in weights:
            nv_weights.append(cout + nv_alpha2*duree)
        nv_s, _ = solver_tsp(G, nv_weights)
        nv_s_p = get_weight_path(G, nv_s, weights)
        parametre_fonctions.append(add_path_function(nv_s_p[0], nv_s_p[1], alpha_values, compteur_fonctions, max_duree))
        frontiere_pareto.append(nv_s_p)
        """nv_path = get_min_path(G, paths, weights, nv_weights=nv_weights, criteria=None)
        parametre_fonctions.append(add_path_function(nv_path[1][0], nv_path[1][1], alpha_values, compteur_fonctions, max_duree))
        frontiere_pareto.append(nv_path[1])"""
        compteur_fonctions += 1
        if(parametre_fonctions[-2] == parametre_fonctions[-1]):
            break

    plt.title("Représentation des fonctions affines")
    plt.xlabel("α")
    plt.ylabel("Valeur de la fonction")
    plt.legend()
    plt.grid(True)

    """plt.figure(figsize=(8, 6))
    pareto_couts = [cout for cout, duree in frontiere_pareto]
    pareto_durees = [duree for cout, duree in frontiere_pareto]
    plt.scatter(pareto_couts, pareto_durees, color='red', label='Frontière de Pareto')
    plt.title("Frontière de Pareto Lagrange")
    plt.xlabel("Coût")
    plt.ylabel("Durée")
    plt.grid(True)
    plt.legend()"""

    return plt, frontiere_pareto


    """# Afficher le graphe
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Afficher le graphe
    plt.show()"""
