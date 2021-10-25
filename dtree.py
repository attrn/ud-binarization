import networkx as nx
import networkx.algorithms.dag as nx_dag


class DTree:
    def __init__(self, dtree=None):
        self.dtree = dtree

    @staticmethod
    def from_sentence(sentence):
        # initialize dependency tree
        dtree = nx.DiGraph()

        # create a dummy root node
        dtree.add_node(0, form='ROOT', upos='ROOT')

        # add nodes to dtree
        for token in sentence:
            dtree.add_node(token.idx, form=token.form, upos=token.upos, deprel=token.deprel)

        # add edges to dtree
        for token in sentence:
            dtree.add_edge(token.head, token.idx, deprel=token.deprel)

        return DTree(dtree)

    def tree(self):
        return self.dtree

    def copy(self):
        return DTree(self.dtree.copy())

    def get_parent(self, child):
        parent_list = list(self.dtree.predecessors(child))
        if parent_list:
            return parent_list[0]
        else:
            return None

    def get_children(self, idx, only_edges=None):
        if only_edges is None:
            return sorted(list(self.dtree.successors(idx)))
        else:
            valid_children = list()
            for child in self.dtree.successors(idx):
                this_deprel = self.dtree.get_edge_data(idx, child)['deprel']
                if this_deprel in only_edges:
                    valid_children.extend([child])
            return sorted(set(valid_children))

    def get_all_descendants(self, idx):
        return sorted(nx_dag.descendants(self.dtree, idx))

    def get_deprel(self, idx):
        # get dependency relation with its head
        parent = list(self.dtree.predecessors(idx))
        if not parent:
            return None
        else:
            return self.dtree.get_edge_data(parent[0], idx)['deprel']

    def get_pos(self, idx):
        return self.dtree.nodes[idx]['upos']

    def get_form(self, idx):
        return self.dtree.nodes[idx]['form']
