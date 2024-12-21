import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value, PULP_CBC_CMD
from execute_brutforce import get_weight_path



def add_path_function(cout, duree, alpha, compt, max_duree):
    d = duree - max_duree
    y = cout + alpha * d
    
    label = f"{compt}: {cout} + α * {d}"
    #return (cout, +1, duree)  # Représente "cout + α * duree"
        
    #plt.plot(alpha, y, label=label)
    return d, cout


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
    
    solver = PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    
    s = [var_edges[edge].varValue for edge in edges]
    solution = []
    for i, e in enumerate(s):
        if(e == 1.0):
            solution.append(edges[i])
    objective_value = value(prob.objective)

    return solution, objective_value

def get_nv_solution(list, contrainte):
    respecte = [l for l in list if l[1] <= contrainte]
    if(not respecte):
        return None
    sol = min(respecte, key=lambda l: l[0])
    return sol



def execute_lagrangienne(G, weights, start_contrainte, pc, pd):
    print("----------------LAGRANGE-----------------")
    
    
    frontiere_pareto = []
    frontiere_pareto.append(pc)
    frontiere_pareto.append(pd)
    next_contrainte = start_contrainte
    i = 0                     
    alpha_values = np.linspace(-10, 10, 200)
   
    
    while(True):
        i+=1
        print("itérations: "+str(i))
        solution_itération = []
        parametre_fonctions = []                #(a1, b1), (a2, b2)....
        compteur_fonctions = 1
        for sol in frontiere_pareto:
            parametre_fonctions.append(add_path_function(sol[0], sol[1], alpha_values, compteur_fonctions, next_contrainte))
            compteur_fonctions += 1

        while(True):
            nv_alpha2 = find_max_of_min(parametre_fonctions, alpha_values)
            nv_weights = []
            for cout, duree in weights:
                nv_weights.append(cout + nv_alpha2*duree)
            nv_s, _ = solver_tsp(G, nv_weights)
            nv_s_p = get_weight_path(G, nv_s, weights)
            parametre_fonctions.append(add_path_function(nv_s_p[0], nv_s_p[1], alpha_values, compteur_fonctions, next_contrainte))
            solution_itération.append(nv_s_p)
            compteur_fonctions += 1
            if(parametre_fonctions[-2] == parametre_fonctions[-1]):
                break
        
        #quelle solution on garde
        sol = get_nv_solution(solution_itération, next_contrainte)
        if(not sol):
            break
        frontiere_pareto.append(sol)
        next_contrainte = sol[1] - 1
        
        

    return _, frontiere_pareto

