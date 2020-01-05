#!/usr/bin/python

import pygraph.classes.graph
import pygraph.algorithms
import pygraph.algorithms.minmax
import time
import random
import sys

if len(sys.argv) != 3:
    print ('usage %s: node_exponent edge_exponent' % sys.argv[0])
    sys.exit(1)

nnodes = 10**int(sys.argv[1])
nedges = 10**int(sys.argv[2])

start_time = time.clock()
def timestamp(s):
    t = time.gmtime(time.clock() - start_time)
    print('biggraph', s.ljust(24), time.strftime('%H:%M:%S', t))

timestamp('generate %d nodes' % nnodes)
bg = pygraph.classes.graph.graph()
bg.add_nodes(range(nnodes))

timestamp('generate %d edges' % nedges)
edges = set()
while len(edges) < nedges:
    left, right = random.randrange(nnodes), random.randrange(nnodes)
    if left == right:
        continue
    elif left > right:
        left, right = right, left
    edges.add((left, right))

timestamp('add edges')
for edge in edges:
    bg.add_edge(edge)

timestamp("Dijkstra")
target = 0
span, dist = pygraph.algorithms.minmax.shortest_path(bg, target)
timestamp('shortest_path done')

# the paths from any node to target is in dict span, let's
# pick any arbitrary node (the last one) and walk to the
# target from there, the associated distance will decrease
# monotonically
lastnode = nnodes - 1
path = []
while lastnode != target:
    nextnode = span[lastnode]
    print('step:', nextnode, dist[lastnode])
    assert nextnode in bg.neighbors(lastnode)
    path.append(lastnode)
    lastnode = nextnode
path.append(target)
timestamp('walk done')
print('path:', path)