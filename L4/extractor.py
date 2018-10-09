'''
https://web.archive.org/web/20161105025307/http://ilk.uvt.nl/conll/
'''

import conll
import sys

pairs = {}
trios = {}

def compute_pairs_trios(formatted_corpus):
    if len(pairs) > 0 or len(trios) > 0: # empty dictionaries if not first time since code is reused
        pairs.clear()
        trios.clear()

    for sentence in formatted_corpus: # sentence is a list, contains a dictionary for each word.
        for word in sentence: # word is a dictionary, contains a dict entry for each column_names as key
            if word['deprel'] == 'SS' or word['deprel'] == 'nsubj':
                subject = word['form'].lower() # form contains the actual word
                verb_id = int(word['head'])# head contains the id of the verb, which is the same as the index of the verb in the sentence.
                verb = sentence[verb_id]['form'].lower()
                add_to_count(subject, verb)
                # Since we found a subject-verb pair, check to see if it's also a subject-verb-object trio
                for w in sentence:
                    if (w['deprel'] == 'OO' or w['deprel'] == 'obj') and w['head'] == word['head']: # both subject and object head should point to the same verb
                        obj = w['form'].lower()
                        add_to_count(subject, verb, obj)
    print_pairs_trios();

def add_to_count(*args):
    # use tuples as keys
    if len(args) == 2:
        if args in pairs:
            pairs[args] += 1
        else:
            pairs[args] = 1
    else:
        if args in trios:
            trios[args] += 1
        else:
            trios[args] = 1

def print_pairs_trios():
    sorted_pairs = sorted(pairs.items(), key = lambda x: x[1], reverse=True)
    cumulative_nr = 0
    print("The five most frequent subject-verb pairs are:")
    for i in range(len(sorted_pairs)):
        cumulative_nr += sorted_pairs[i][1]
        if (i < 5):
            print(sorted_pairs[i][0], ":", sorted_pairs[i][1]) # here are the frequencies you should find: 537, 261, 211, 171, 161
    print("Total number of subject-verb pairs:", cumulative_nr, "\n") # you should find 18885 pairs.

    sorted_trios = sorted(trios.items(), key = lambda x: x[1], reverse=True)
    cumulative_nr = 0
    print("The five most frequent trios are:")
    for i in range(len(sorted_trios)):
        cumulative_nr += sorted_trios[i][1]
        if (i < 5):
            print(sorted_trios[i][0], ":", sorted_trios[i][1]) # here are the frequencies you should find: 37, 36, 36, 19, 17
    print("Total number of subject-verb-object trios:", cumulative_nr) # you should find 5844 triples.

if __name__ == '__main__':
    if sys.argv[1] == 'x':
        column_names_x = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
        # [id:1, form:Äktenskapet, lemma:_, cpostag:NN, postag:NN, feats:_, head:4, deprel:SS, phead:_, pdeprel:_]
        train_file = 'swedish_talbanken05_train.conll'
        sentences = conll.read_sentences(train_file)
        formatted_corpus = conll.split_rows(sentences, column_names_x)
        compute_pairs_trios(formatted_corpus)
    elif sys.argv[1] == 'u':
        column_names_u = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']
        files = conll.get_files('ud-treebanks-v2.2', 'train.conllu') # https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2837
        for file in files:
            print("\nFile:", file)
            sentences = conll.read_sentences(file)
            formatted_corpus = conll.split_rows(sentences, column_names_u)
            compute_pairs_trios(formatted_corpus)
            print("-------------------------------------------------")
