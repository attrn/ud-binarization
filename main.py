import os
import argparse
import networkx as nx
from pathlib import Path
from data_loader import read_conllu
from dtree import DTree
from btree import BTree


class TRange:
    def __init__(self, start_idx, end_idx, type_changed=False):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.type_changed = type_changed

    def __repr__(self):
        return ':'.join([str(self.start_idx), str(self.end_idx)])

    def __key(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, TRange):
            if (self.type_changed and other.type_changed) or (not self.type_changed and not other.type_changed):
                return self.__key() == other.__key()
            else:
                return False
        return NotImplemented

    def contains(self, idx):
        # both start and end indices are inclusive
        if self.start_idx <= idx <= self.end_idx:
            return True
        else:
            return False

    def contains_range(self, tr):
        # both start and end indices are inclusive
        if self.start_idx <= tr.start_idx and self.end_idx >= tr.end_idx:
            return True
        else:
            return False

    @staticmethod
    def merge_range(tr1, tr2):
        return TRange(min(tr1.start_idx, tr2.start_idx), max(tr1.end_idx, tr2.end_idx))

    @staticmethod
    def new_range(idx, old_range):
        if idx < old_range.start_idx:
            return TRange(idx, old_range.end_idx)
        elif idx > old_range.end_idx:
            return TRange(old_range.start_idx, idx)
        else:
            return old_range


def check_cross_dependencies(sentence):
    range_list = []

    for token in sentence:
        token_idx = token.idx
        token_head = token.head
        new_range = TRange(min(token_idx, token_head), max(token_idx, token_head))

        for r in range_list:
            # if not one range contains the other
            if not new_range.contains_range(r) and not r.contains_range(new_range):
                if (max(new_range.start_idx, r.start_idx) - min(new_range.end_idx, r.end_idx)) < 0:
                    return True

        range_list.append(new_range)

    return False


def to_sexp(dtree, btree, head_map):
    class Description:
        def __init__(self, tag, content=None, is_head=False):
            self.tag = tag
            self.content = content
            self.is_head = is_head

            # to avoid conflicts with sexp's parentheses
            if self.content is not None:
                self.content = self.content.replace('(', '-LRB-')
                self.content = self.content.replace(')', '-RRB-')

            # remove '*' from lifted deprel
            if self.tag is not None:
                self.tag = self.tag.replace('*', '')

        def __repr__(self):
            if self.content is None:
                if self.is_head:
                    return '{}-H'.format(self.tag)
                else:
                    return '{}'.format(self.tag)
            else:
                if self.is_head:
                    return '{}-H {}'.format(self.tag, self.content)
                else:
                    return '{} {}'.format(self.tag, self.content)

    class Node:
        def __init__(self, description, left_child=None, right_child=None):
            self.description = description
            self.left_child = left_child
            self.right_child = right_child

        def __repr__(self):
            if self.left_child is not None and self.right_child is not None:
                template = '({} {}{})'
                return template.format(self.description,
                                       self.left_child,
                                       self.right_child)
            else:
                return '({})'.format(self.description)

    # for each node in btree
    # - get index of its head from dtree
    # - check if its head is in the left branch or right branch
    # - assign H to the branch that contains its head
    btree_root = btree.get_root()

    def _traverse(btree_node, is_head=False):
        # get idx of current node
        idx = btree.btree.nodes[btree_node]['idx']

        # get head_idx of current node
        head_idx = head_map[idx]

        children = btree.get_children(btree_node)
        if children:
            if btree.btree.nodes[children[0]]['idx'] < btree.btree.nodes[children[1]]['idx']:
                left_child = children[0]
                right_child = children[1]
            else:
                left_child = children[1]
                right_child = children[0]

            description = Description(tag=btree.btree.nodes[btree_node]['name'], is_head=is_head)

            # get idx of all descendants' heads
            def get_all_descendants_idx(node):
                idx_set = set()
                idx_set.add(btree.btree.nodes[node]['idx'])
                descendants = btree.get_all_descendants(node)
                for descendant in descendants:
                    idx_set.add(btree.btree.nodes[descendant]['idx'])

                return idx_set

            # check which branch the head of current node belongs to
            if head_idx in get_all_descendants_idx(left_child):
                return Node(description,
                            _traverse(left_child, is_head=True),
                            _traverse(right_child))
            else:
                return Node(description,
                            _traverse(left_child),
                            _traverse(right_child, is_head=True))
        else:
            upos = dtree.get_pos(idx)
            description = Description(tag=upos, content=btree.btree.nodes[btree_node]['name'], is_head=is_head)
            return Node(description)

    sexp_root = _traverse(btree_root, is_head=False)

    return str(sexp_root)


def pprint_sexp(sexp):
    pretty_sexp = ''
    opening_brackets = list()
    break_point = 0
    offset = 0
    for i in range(len(sexp)-1):
        if sexp[i] == '(':
            opening_brackets.append(i-offset)
        elif sexp[i] == ')':
            if sexp[i+1] == ')':
                try:
                    opening_brackets.pop()
                except IndexError:
                    print(i)
                    print(sexp)
            else:
                last_opening_bracket = opening_brackets.pop()
                pretty_sexp = pretty_sexp + sexp[break_point:i+1] + '\n' + ' '*last_opening_bracket
                offset = i-last_opening_bracket+1
                break_point = i+1
    pretty_sexp += sexp[break_point:]
    return pretty_sexp


def ud_binarize(in_path, out_path, use_pseudo_projective=False):
    # read UD data
    ud_sentences = read_conllu(in_path)

    with open(out_path, 'w') as f_out:
        for ud_sentence in ud_sentences:
            sentence = ud_sentence.sentence
            sent_id = ud_sentence.sent_id
            text = ud_sentence.text

            # create a map of dependent_idx -> head_idx
            head_map = dict()
            for token in sentence:
                head_map[token.idx] = token.head

            # store UD data in a dependency tree data structure
            dtree = DTree.from_sentence(sentence)

            # check projectivity
            if check_cross_dependencies(sentence) and use_pseudo_projective:
                dtree_root = dtree.get_children(0)[0]
                bottom_up = list()
                range_list = list()

                # top-down traverse
                # save arcs that cross previously traversed arcs
                def _traverse(dtree, parent):
                    # get immediate children
                    children = dtree.get_children(parent)

                    for child in children:
                        this_range = TRange(min(child, parent), max(child, parent))

                        for r in range_list:
                            # if not one range contains the other
                            if not this_range.contains_range(r) and not r.contains_range(this_range):
                                if (max(this_range.start_idx, r.start_idx) - min(this_range.end_idx, r.end_idx)) < 0:
                                    if child not in bottom_up:
                                        bottom_up.append(child)

                        range_list.append(this_range)

                    # recursion
                    for child in children:
                        _traverse(dtree, child)

                _traverse(dtree, dtree_root)

                # traverse crossing arcs from bottom-up
                for n in reversed(bottom_up):
                    # get parent of current node
                    parent = dtree.get_parent(n)

                    range_list.remove(TRange(min(n, parent), max(n, parent)))

                    to_lift = True
                    final_lift_dest = parent

                    # keep lifting until projective
                    current_parent = parent
                    while to_lift:
                        grandparent = dtree.get_parent(current_parent)

                        if grandparent:
                            new_range = TRange(min(n, grandparent), max(n, grandparent))

                            is_projective = True
                            for r in range_list:
                                # if not one range contains the other
                                if not new_range.contains_range(r) and not r.contains_range(new_range):
                                    if (max(new_range.start_idx, r.start_idx) - min(new_range.end_idx, r.end_idx)) < 0:
                                        is_projective = False

                            if is_projective:
                                to_lift = False
                                final_lift_dest = grandparent
                            else:
                                current_parent = grandparent
                        else:
                            to_lift = False

                    if final_lift_dest != parent:
                        # mark this deprel as a result of lifting
                        new_deprel = dtree.tree().nodes[n]['deprel'] + '*'
                        dtree.tree().add_edge(final_lift_dest, n, deprel=new_deprel)
                        dtree.tree().remove_edge(parent, n)

            # convert dtree to binary tree
            btree = BTree.from_dtree(dtree)

            # convert to s-expression
            sexpr = to_sexp(dtree, btree, head_map)

            f_out.write('# sent_id = {}\n'.format(sent_id))
            f_out.write('# text = {}\n'.format(text))
            f_out.write('{}\n\n'.format(pprint_sexp(sexpr)))


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--ud-path', action='store', dest='ud_path', required=True,
                        help='path to UD directory (e.g.: ud-treebanks-v2.8)')

    parser.add_argument('--export-path', action='store', dest='export_path', required=True,
                        help='where converted treebanks should be stored')

    parser.add_argument('--use-pseudo-projective', action='store_true', default=False,
                        dest='use_pseudo_projective',
                        help='apply pseudo-projective approach to binarize non-projective trees')

    args = parser.parse_args()

    ud_path = args.ud_path
    export_path = args.export_path
    use_pseudo_projective = args.use_pseudo_projective

    for root, subdirs, files in sorted(os.walk(ud_path)):
        dirpath, dirname = os.path.split(root)
        current_export_path = os.path.join(export_path, dirname)

        for file in files:
            if file.endswith('.conllu'):
                filename = os.path.splitext(file)[0]

                conllu_path = os.path.join(root, file)
                Path(current_export_path).mkdir(parents=True, exist_ok=True)
                binarized_path = os.path.join(current_export_path, filename + '.binarized')

                print('Binarizing {}'.format(conllu_path))

                ud_binarize(conllu_path, binarized_path, use_pseudo_projective)
