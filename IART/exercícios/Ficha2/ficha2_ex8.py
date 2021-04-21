from search import *

ex8_graph = Graph({'1': {'2': 0, '3': 0}}, directed=False)
ex8_graph.connect('2', '4', 0)
ex8_graph.connect('2', '5', 0)
ex8_graph.connect('3', '6', 0)
ex8_graph.connect('3', '7', 0)
ex8_graph.connect('4', '8', 0)
ex8_graph.connect('4', '9', 0)
ex8_graph.connect('5', '10', 0)
ex8_graph.connect('5', '11', 0)
ex8_graph.connect('6', '12', 0)
ex8_graph.connect('6', '13', 0)
ex8_graph.connect('7', '14', 0)
ex8_graph.connect('7', '15', 0)

ex8_problem = GraphProblem('1', '11', ex8_graph)

print(ex8_graph.nodes())

print(breadth_first_graph_search(ex8_problem).solution())
print(breadth_first_graph_search(ex8_problem).path())
print(breadth_first_graph_search(ex8_problem).depth)

#print(depth_first_graph_search(ex8_problem).solution())
#print(depth_first_graph_search(ex8_problem).path())
#print(depth_first_graph_search(ex8_problem).depth)

print(depth_limited_search(ex8_problem, 3).solution())
print(depth_limited_search(ex8_problem, 3).path())
print(depth_limited_search(ex8_problem, 3).depth)

print(iterative_deepening_search(ex8_problem).solution())
print(iterative_deepening_search(ex8_problem).path())
print(iterative_deepening_search(ex8_problem).depth)