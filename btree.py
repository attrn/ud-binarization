import json
import networkx as nx
import networkx.algorithms.dag as nx_dag


obliqueness_hierarchy_path = 'ud2-obliqueness-hierarchy.json'
obliqueness_hierarchy = {}
with open(obliqueness_hierarchy_path, 'r') as f:
    json_data = json.load(f)
    for obj in json_data:
        obliqueness_hierarchy[obj['name']] = obj['priority']


class BTree:
    def __init__(self, btree=None):
        self.btree = btree

    @staticmethod
    def from_dtree(dtree):
        # initialize binary tree
        btree = nx.DiGraph()

        # follow https://www.aclweb.org/anthology/D17-1009.pdf
        def _binarize(btree, dtree, parent):
            # get immediate children
            children = dtree.get_children(parent)

            # add children to their stacks according to their position relative to parent
            left_stack = []
            for child in children:
                if child < parent:
                    left_stack.append(child)

            right_stack = []
            for child in reversed(children):
                if child > parent:
                    right_stack.append(child)

            # get obliqueness hierarchy scores of children
            obliqueness_hierarchy_scores = {}
            for child in children:
                # deprel with '*' indicates it has been moved from its original position;
                # in this case we move it to the top of the hierarchy (EXPERIMENTAL)
                deprel = dtree.get_deprel(child)
                if '*' in deprel:
                    obliqueness_hierarchy_scores[child] = 100
                else:
                    # obliqueness hierarchy doesn't include sub-dependency types (e.g., nmod:poss)
                    deprel_parts = deprel.split(':')
                    deprel = deprel_parts[0]
                    obliqueness_hierarchy_scores[child] = obliqueness_hierarchy[deprel]

            # sort children by scores (naive sort)
            # sorted_children = sorted(obliqueness_hierarchy_scores,
            #                          key=obliqueness_hierarchy_scores.get,
            #                          reverse=False)

            # sort children on top of the stacks by scores
            sorted_children = []
            while len(left_stack) > 0 or len(right_stack) > 0:
                if len(left_stack) == 0 and len(right_stack) > 0:
                    sorted_children.append(right_stack[-1])
                    right_stack.pop()
                    continue
                elif len(left_stack) > 0 and len(right_stack) == 0:
                    sorted_children.append(left_stack[-1])
                    left_stack.pop()
                    continue

                top_left = left_stack[-1]
                top_right = right_stack[-1]

                top_left_score = obliqueness_hierarchy_scores[top_left]
                top_right_score = obliqueness_hierarchy_scores[top_right]

                if top_left_score <= top_right_score:
                    sorted_children.append(top_left)
                    left_stack.pop()
                else:
                    sorted_children.append(top_right)
                    right_stack.pop()

            # recursively binarize tree
            dtree_parent = dtree.dtree.nodes[parent]
            btree_parent = (dtree_parent['form'] + '*' + str(parent))
            btree.add_node(btree_parent,
                           name=dtree_parent['form'],
                           deprel=None,
                           idx=parent)

            for child in sorted_children:
                this_deprel = dtree.get_deprel(child)
                temp_root = (this_deprel + ':' + str(child))
                btree.add_node(temp_root,
                               name=this_deprel,
                               deprel=this_deprel,
                               idx=child)

                btree.add_edge(temp_root, btree_parent)
                btree.add_edge(temp_root, _binarize(btree, dtree, child))

                btree_parent = temp_root

            return btree_parent

        # root = 0
        # root only has one child
        _binarize(btree, dtree, dtree.get_children(0)[0])

        return BTree(btree)

    def tree(self):
        return self.btree

    def get_root(self):
        return [n for n, d in self.btree.in_degree() if d == 0][0]

    def get_children(self, idx):
        children = sorted(list(self.btree.successors(idx)))
        if idx in children:
            children.remove(idx)
        return children

    def get_all_descendants(self, idx):
        return nx_dag.descendants(self.btree, idx)
