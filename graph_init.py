import networkx as nx
import random
import math


def add_block_edges(G, nodes):
    n1, n2, n3, n4 = nodes
    G.add_edges_from([(n1, n2), (n2, n3), (n4, n3), (n1, n4)])
    G.add_edges_from([(n1, n3), (n4, n2)])

def add_weight_edge(G, nb_max_cout, nb_max_duree):
    """
    1 <= cout <= NB_MAX_COUT, 1 <= duree <= NB_MAX_DUREE
    """
    longueur = len(G.edges())
    weights = []
    
    for _ in range(longueur):
        choix = random.randint(1, 2)
        if choix == 1:
            cout = random.randint(1, nb_max_cout)
            duree = random.randint(1, math.ceil(nb_max_duree / 3))
        else:
            cout = random.randint(1, math.ceil(nb_max_cout / 3))
            duree = random.randint(1, nb_max_duree)
        weights.append((cout, duree))
    return weights



#----------------------------------------------Graph en blocks----------------------------------------------

def blocks_graph(nb_blocks, blocks_spacing):
    G = nx.DiGraph()

    # Initialiser les positions des nœuds
    pos = {}
    node_counter = 1  

    # Créer le premier bloc de 4 nœuds et les positionner
    first_block_nodes = list(range(node_counter, node_counter + 4))
    add_block_edges(G, first_block_nodes)

    pos[first_block_nodes[0]] = (0, 0)  
    pos[first_block_nodes[1]] = (1, 0)  
    pos[first_block_nodes[2]] = (1, 1)  
    pos[first_block_nodes[3]] = (0, 1)  

    # Mettre à jour le compteur de nœuds
    node_counter += 4
    first_iteration = True

    for b in range(1, nb_blocks):
        new_block_nodes = list(range(node_counter, node_counter + 2))
        
        if first_iteration:
            last_block_nodes = [2, 3]
            first_iteration = False
        else:
            last_block_nodes = [node_counter - 2, node_counter - 1]
            
        G.add_edges_from([(last_block_nodes[0], new_block_nodes[0]),
                        (last_block_nodes[1], new_block_nodes[1]),
                        (new_block_nodes[1], new_block_nodes[0]),
                        (last_block_nodes[0], new_block_nodes[1]),  # Diagonale
                        (last_block_nodes[1], new_block_nodes[0])])  # Diagonale

        # Positionner les nouveaux nœuds pour qu'ils soient alignés
        pos[new_block_nodes[0]] = (1 + b * blocks_spacing, 1)  # Aligné sur la première rangée
        pos[new_block_nodes[1]] = (1 + b * blocks_spacing, 0)  # Aligné sur la deuxième rangée
        
        node_counter += 2
        
    return G


#----------------------------------------------Small World----------------------------------------------












