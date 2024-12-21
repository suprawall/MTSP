from graphviz import Digraph

# Créer un objet Digraph (graphe orienté)
dot = Digraph(comment='Figure 2: Linearly Priced Timed Automaton')
dot.attr(rankdir='LR')

# Ajouter les nœuds avec leurs étiquettes (les poids sont entre parenthèses)
dot.node('1', '1')
dot.node('2', '2')
dot.node('3', '3')
dot.node('4', '4')
dot.node('5', '5')
dot.node('6', '6')


# Ajouter les arcs avec des conditions de temps (au-dessus) et des prix (en dessous)
dot.edge('1', '2')
dot.edge('1', '3')
dot.edge('1', '4')
dot.edge('2', '3')
dot.edge('2', '4')
dot.edge('4', '3')
dot.edge('3', '5')
dot.edge('3', '6')
dot.edge('4', '5')
dot.edge('4', '6')
dot.edge('6', '5')

"""dot.edge('A', 'C', style='dotted')
dot.edge('B', 'D', label='x = 2, +1\n')
dot.edge('C', 'D', label='x = 2, +5\n')"""

#dot.edge('B', 'B', label= 'x ≤ 2', style ='invis', color='white')

# Sauvegarder et visualiser le graphe
dot.render('output/forme_graphe', format='png', cleanup=True)

# Afficher le graphe dans le notebook (si nécessaire)
from IPython.display import Image
Image(filename='output/forme_graphe.png')
