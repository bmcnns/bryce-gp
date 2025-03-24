from copy import deepcopy

from brycegp.program import Program
import random

def get_crossover_point(program):
    candidates = []

    def traverse(node, parent, attr, is_root):
        if not is_root:
            candidates.append((node, parent, attr))
        if not node.is_leaf_():
            traverse(node.left, parent=node, attr='left', is_root=False)
            traverse(node.right, parent=node, attr='right', is_root=False)

    traverse(program, parent=None, attr=None, is_root=True)
    return random.choice(candidates) if candidates else (program, None, None)

def crossover(A, B):
    child1 = deepcopy(A)
    child2 = deepcopy(B)
    node1, parent1_ref, attr1 = get_crossover_point(child1)
    node2, parent2_ref, attr2 = get_crossover_point(child2)

    if parent1_ref is None or parent2_ref is None:
        return child2, child1

    setattr(parent1_ref, attr1, node2)
    setattr(parent2_ref, attr2, node1)

    return child1, child2
