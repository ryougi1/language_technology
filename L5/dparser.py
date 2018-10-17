"""
Gold standard parser
"""
__author__ = "Pierre Nugues"

import conll, transition, features
import sys, time
from sklearn import metrics
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
import pickle

def reference(stack, queue, graph):
    """
    Gold standard parsing
    Produces a sequence of transitions from a manually-annotated corpus:
    sh, re, ra.deprel, la.deprel
    :param stack: The stack
    :param queue: The input list
    :param graph: The set of relations already parsed
    :return: the transition and the grammatical function (deprel) in the
    form of transition.deprel
    """
    # Right arc
    if stack and stack[0]['id'] == queue[0]['head']:
        # print('ra', queue[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + queue[0]['deprel']
        stack, queue, graph = transition.right_arc(stack, queue, graph)
        return stack, queue, graph, 'ra' + deprel
    # Left arc
    if stack and queue[0]['id'] == stack[0]['head']:
        # print('la', stack[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + stack[0]['deprel']
        stack, queue, graph = transition.left_arc(stack, queue, graph)
        return stack, queue, graph, 'la' + deprel
    # Reduce
    if stack and transition.can_reduce(stack, graph):
        for word in stack:
            if (word['id'] == queue[0]['head'] or
                        word['head'] == queue[0]['id']):
                # print('re', stack[0]['cpostag'], queue[0]['cpostag'])
                stack, queue, graph = transition.reduce(stack, queue, graph)
                return stack, queue, graph, 're'
    # Shift
    # print('sh', [], queue[0]['cpostag'])
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'

def get_feature_names(feature_type):
    '''
    You will consider three feature sets and you will train the corresponding logistic regression models using scikit-learn:
        1. The top of the stack and the first word of the input list (word forms and parts of speech);
        2. The two first words and POS on the top of the stack and the two first words and POS of the input list;
        3. A feature vector that you will design that will extend the previous one with at least two features,
           one of them being the part of speech and the word form of the word following the top of the stack in the sentence order.
           You can read this paper (Table 6) to build your vector. In this paper, Sect. 4 contains the description of the feature codes:
           LEX, POS, fw, etc.
    These sets will include two additional Boolean parameters, "can do left arc" and "can do reduce", which will model constraints on the parser's actions. In total, the feature sets will then have six, respectively ten and 14, parameters.
    '''
    if feature_type == '1':
        return ['stack0_POS', 'stack0_form', 'queue0_POS', 'queue0_form', 'can_re', 'can_la']
    elif feature_type == '2':
        return ['stack0_POS', 'stack1_POS', 'stack0_form', 'stack1_form', 'queue0_POS', 'queue1_POS', 'queue0_form', 'queue1_form', 'can-re', 'can-la']
    elif feature_type == '3':
        return ['stack0_POS', 'stack1_POS', 'stack0_form', 'stack1_form', 'queue0_POS', 'queue1_POS', 'queue0_form', 'queue1_form', 'stack_next_POS', 'stack_next_form', 'stack_prev_POS', 'stack_prev_form', 'can-re', 'can-la']
    else:
        print('Feature type incorrect, exiting')
        sys.exit()

def extract_features(formatted_corpus, feature_names, do_print):
        x_features = []
        y_transitions = []
        sent_cnt = 0
        for sentence in formatted_corpus:
            sent_cnt += 1
            # if sent_cnt % 1000 == 0:
            #     print(sent_cnt, 'sentences on', len(formatted_corpus), flush=True)
            stack = []
            queue = list(sentence)
            graph = {}
            graph['heads'] = {}
            graph['heads']['0'] = '0'
            graph['deprels'] = {}
            graph['deprels']['0'] = 'ROOT'
            while queue:
                x_features.append(features.extract(stack, queue, graph, feature_names, sentence))
                stack, queue, graph, trans = reference(stack, queue, graph)
                y_transitions.append(trans)
            stack, graph = transition.empty_stack(stack, graph)

            # Poorman's projectivization to have well-formed graphs.
            for word in sentence:
                word['head'] = graph['heads'][word['id']]

            if(sent_cnt == 1 and do_print):
                print(y_transitions)

        return x_features, y_transitions

def parse_ml(stack, queue, graph, trans):
    '''
    Continued from assignment instructions https://pnugues.github.io/edan20/cw6.xml
    '''
    if stack and trans[:2] == 'ra':
        stack, queue, graph = transition.right_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'ra'
    elif stack and trans[:2] == 'la':
        stack, queue, graph = transition.left_arc(stack, queue, graph, trans[3:])
        return(stack, queue, graph, 'la')
    elif stack and trans == 're':
        stack, queue, graph = transition.reduce(stack, queue, graph)
        return(stack, queue, graph, 're')
    else:
        stack, queue, graph = transition.shift(stack, queue, graph)
        return(stack, queue, graph, 'sh')

if __name__ == '__main__':
    '''
    Setup and Training - Assignment 5
    '''
    start_time = time.time()
    if(len(sys.argv) < 2):
        print("Please enter feature type")
        sys.exit()
    train_file = 'data/swedish_talbanken05_train.conll'
    test_file = 'data/swedish_talbanken05_test_blind.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']
    feature_type = sys.argv[1]
    feature_names = get_feature_names(feature_type)
    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)
    features_train, transitions_train = extract_features(formatted_corpus, feature_names, do_print = True)

    try:
        with open("pickle/model"+feature_type, "rb") as m:
            model = pickle.load(m)
        with open("pickle/vec"+feature_type, "rb") as v:
            vec = pickle.load(v)
        print("Succesfully loaded models and vectorizer")
    except FileNotFoundError:
        vec = DictVectorizer(sparse=True)
        X  = vec.fit_transform(features_train)
        classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
        model = classifier.fit(X, transitions_train)
        # y_test_predicted = classifier.predict(X)
        # print("Classification report for classifier %s:\n%s\n"
        #   % (classifier, metrics.classification_report(transitions_train, y_test_predicted)))
        with open("pickle/model"+feature_type, "wb") as m:
            pickle.dump(model, m)
        with open("pickle/vec"+feature_type, "wb") as v:
            pickle.dump(vec, v)

    '''
    Testing - Assignment 6
    '''
    sentences = conll.read_sentences(test_file)
    # formatted_corpus = conll.split_rows(sentences, column_names_2006_test)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    y_test_transitions = []
    sent_cnt = 0

    for sentence in formatted_corpus:
        sent_cnt += 1
        if sent_cnt % 1000 == 0:
            print(sent_cnt, 'sentence on', len(formatted_corpus), flush = True)
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'
        while queue:
            X_test = vec.transform(features.extract(stack, queue, graph, feature_names, sentence))
            predicted_transition = model.predict(X_test)[0]

            stack, queue, graph, trans = parse_ml(stack, queue, graph, predicted_transition)
            y_test_transitions.append(trans)

        stack, graph = transition.empty_stack(stack, graph)

        # Poorman's projectivization to have well-formed graphs.
        for word in sentence:
            try:
                word['head'] = graph['heads'][word['id']]
            except:
                word['head'] = '_'
            try:
                word['deprel'] = graph['deprels'][word['id']]
            except:
                word['deprel'] = '_'

    conll.save('results'+feature_type, formatted_corpus, column_names_2006)
    print("\n--- Execution time: %s seconds ---" % (time.time() - start_time))
