import copy


class UDToken:
    def __init__(self, idx, form, upos, feats, head, deprel):
        self.idx = idx   # index of this token
        self.form = form
        self.upos = upos
        self.feats = feats
        self.head = head   # index of head
        self.deprel = deprel


class UDSentence:
    def __init__(self, sentence, sent_id, text):
        self.sentence = sentence
        self.sent_id = sent_id
        self.text = text


def read_conllu(path, remove_empty_nodes=True):
    sentences = []

    with open(path, 'r') as f:
        # add empty line at the end to imitate conllu sentence break
        lines = f.readlines()
        lines.append('')

        sentence = []
        sent_id = 'None'
        text = 'None'

        for line in lines:
            # empty line in conllu file indicates sentence break
            if line.strip() == '':
                if len(sentence) > 0:
                    new_sentence = UDSentence(copy.deepcopy(sentence), sent_id, text)
                    sentences.append(new_sentence)

                sentence = []
                sent_id = 'None'
                text = 'None'
                continue

            # extract sent_id and text
            # skip other comments in conllu file
            if line.startswith('#'):
                if line.startswith('# sent_id'):
                    parts = line.strip().split(' ')
                    sent_id = parts[-1]
                    continue
                elif line.startswith('# text ='):
                    text = line[9:].strip()
                    continue
                else:
                    continue

            # split field by tab
            fields = line.strip().split('\t')

            if remove_empty_nodes:
                if '.' in fields[0] or '-' in fields[0]:
                    continue

            # 0 = word index (starting at 1)
            # 1 = word form
            # 3 = UPOS
            # 5 = features
            # 6 = head of current word index
            # 7 = UD relation
            try:
                current_token = UDToken(int(fields[0]),
                                        fields[1],
                                        fields[3],
                                        fields[5],
                                        int(fields[6]),
                                        fields[7])
            except ValueError:
                pass
            else:
                sentence.append(current_token)

    return sentences
