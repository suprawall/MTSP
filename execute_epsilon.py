import networkx as nx
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value
from graph_init import blocks_graph, add_weight_edge
import matplotlib.pyplot as plt


def est_domine(s, frontiere):
    """s = [cout_1, duree_1] <? si = [cout_i, duree_i] in frontiere
    """
    for sol in frontiere:
        if( (s[0] > sol[0]) and (s[1] > sol[1]) and s != sol):
            return sol
    return None

def domine(s, frontiere):
    """l'inverse de est_domine()
    """
    for sol in frontiere:
        if( (s[0] < sol[0]) and (s[1] < sol[1]) and s != sol):
            return sol
    return None

def get_weight_path(G, path, weights):
    """en entré un chemin défini et retourne juste son poid
    
    """
    
    edges = list(G.edges())
    cout = 0
    duree = 0
    
    for i, edge in enumerate(edges):
        if(edge in path):
            cout += weights[i][0]
            duree += weights[i][1]
            
    return cout, duree


def solveur(G, weights, critere):
    """solveur avec Pulp: minimise le critere(0: cout, 1: duree)
    """
    edges = list(G.edges())
        
    prob = LpProblem("Minimize_Objectif", LpMinimize)
    var_edges = LpVariable.dicts("Edges", edges, 0, 1, cat='Binary')
    
    prob += lpSum([var_edges[edge] * weights[i][critere] for i, edge in enumerate(edges)])
    
    source, target = 1, max(G.nodes()) - 1
    
    for node in G.nodes():
        if node == source:  # Le nœud source doit avoir un flux sortant = 1
            prob += lpSum([var_edges[edge] for edge in G.out_edges(node)]) == 1
        elif node == target:  # Le nœud cible doit avoir un flux entrant = 1
            prob += lpSum([var_edges[edge] for edge in G.in_edges(node)]) == 1
        else:  # Tous les autres nœuds doivent avoir des flux entrants et sortants égaux (conservation du flux)
            prob += lpSum([var_edges[edge] for edge in G.in_edges(node)]) == lpSum([var_edges[edge] for edge in G.out_edges(node)])
    
    prob.solve()

    s = [var_edges[edge].varValue for edge in edges]
    solution = []
    for i, e in enumerate(s):
        if(e == 1.0):
            solution.append(edges[i])
    objective_value = value(prob.objective)

    return solution, objective_value
    

def solveur_containte(G, weights, a_minimiser, epsilon):
    """solveur avec pulp, Min(coût) avec durée < epsilon

    Args:
        G (graph)
        weights ([(cout_1, duree_1),..., (cout_n, duree_n)])
        epsilon (int)
    """
    edges = list(G.edges())
        
    prob = LpProblem("Minimize_Cost", LpMinimize)
    var_edges = LpVariable.dicts("Edges", edges, 0, 1, cat='Binary')
    
    if(a_minimiser == 0):
        prob += lpSum([var_edges[edge] * weights[i][0] for i, edge in enumerate(edges)])
        prob += lpSum([var_edges[edge] * weights[i][1] for i, edge in enumerate(edges)]) <= epsilon
    elif(a_minimiser == 1):
        prob += lpSum([var_edges[edge] * weights[i][1] for i, edge in enumerate(edges)])
        prob += lpSum([var_edges[edge] * weights[i][0] for i, edge in enumerate(edges)]) <= epsilon
    
    source, target = 1, max(G.nodes()) - 1
    
    for node in G.nodes():
        if node == source:  # Le nœud source doit avoir un flux sortant = 1
            prob += lpSum([var_edges[edge] for edge in G.out_edges(node)]) == 1
        elif node == target:  # Le nœud cible doit avoir un flux entrant = 1
            prob += lpSum([var_edges[edge] for edge in G.in_edges(node)]) == 1
        else:  # Tous les autres nœuds doivent avoir des flux entrants et sortants égaux (conservation du flux)
            prob += lpSum([var_edges[edge] for edge in G.in_edges(node)]) == lpSum([var_edges[edge] for edge in G.out_edges(node)])
    
    prob.solve()

    s = [var_edges[edge].varValue for edge in edges]
    solution = []
    for i, e in enumerate(s):
        if(e == 1.0):
            solution.append(edges[i])
    objective_value = value(prob.objective)

    return solution, objective_value

def execute_epsilon(G, weights, pc, pd):
    frontiere_pareto = []
    frontiere_pareto.append(pc)
    frontiere_pareto.append(pd)

    nv_epsilon = pc[1] - 1
    
    while(True):
        s, _ = solveur_containte(G, weights, 0, nv_epsilon)
        p_s = get_weight_path(G, s, weights)
        if(p_s == pd[1] or p_s in frontiere_pareto):
            break
        s2, _ = solveur_containte(G, weights, 1, p_s[0])
        p_s2 = get_weight_path(G, s2, weights)
        frontiere_pareto.append(p_s)
        if(p_s2 not in frontiere_pareto):
            frontiere_pareto.append(p_s2)
        
        nv_epsilon = p_s2[1] - 1
        

    print(frontiere_pareto)

    """plt.figure(figsize=(8, 6))
    pareto_couts = [cout for cout, duree in frontiere_pareto]
    pareto_durees = [duree for cout, duree in frontiere_pareto]
    plt.scatter(pareto_couts, pareto_durees, color='red', label='Frontière de Pareto')
    plt.title("Frontière de Pareto Epsilon")
    plt.xlabel("Coût")
    plt.ylabel("Durée")
    plt.grid(True)
    plt.legend()"""

    return _, frontiere_pareto









