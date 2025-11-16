from graph import draw_graph
from node import Node
    
a = Node(2.0, label="a")
b = Node(-3.0, label="b")
c = Node(10.0, label="c")
e = a * b
e.label = "e"
d = e + c
d.label = "d"
f = Node(-2.0, label="f")
L = d * f
L.label = "L"
print(L)

# Visualize the computational graph.
graph = draw_graph(L)
graph.render("computational_graph", view=True)