import transition

do_print = 0

def extract(stack, queue, graph, feature_names, sentence):
    '''
    Extract the features from one sentence
    Extracted features are determined by feature_names
    Returns the extracted features in a dictionary format compatible with scikit-learn
    TODO: Add final feature
    '''
    global do_print
    vec = list()
    for key in feature_names[:len(feature_names)-2]: # can_re and can_la added later
        try:
            if key[:5] == 'stack':
                if key[-4:] == 'form':
                    vec.append(stack[int(key[5:6])]['form'])
                elif key[-3:] == 'POS':
                    vec.append(stack[int(key[5:6])]['postag'])
            else:
                if key[-4:] == 'form':
                    vec.append(queue[int(key[5:6])]['form'])
                elif key[-3:] == 'POS':
                    vec.append(queue[int(key[5:6])]['postag'])
        except:
            vec.append('nil')

    vec.append(str(transition.can_reduce(stack, graph)))
    vec.append(str(transition.can_leftarc(stack, graph)))
    if do_print < 9:
        print(vec)
        do_print += 1
    return dict(zip(feature_names, vec))

'''
The first lines of your features for the 4 (READ:6?) parameters (x) and labelled actions (y) should look like the excerpt below, where the columns correspond to:
[stack0_POS, stack1_POS, stack0_word, stack1_word, queue0_POS, queue1_POS, queue0_word, queue1_word, can-re, can-la, and the transition value]

x = ['nil', 'nil', 'nil', 'nil', 'ROOT', 'NN', 'ROOT', 'Äktenskapet', False, False], y = sh
x = ['ROOT', 'nil', 'ROOT', 'nil', 'NN', '++', 'Äktenskapet', 'och', True, False], y = sh
x = ['NN', 'ROOT', 'Äktenskapet', 'ROOT', '++', 'NN', 'och', 'familjen', False, True], y = sh
x = ['++', 'NN', 'och', 'Äktenskapet', 'NN', 'AV', 'familjen', 'är', False, True], y = la.++
x = ['NN', 'ROOT', 'Äktenskapet', 'ROOT', 'NN', 'AV', 'familjen', 'är', False, True], y = ra.CC
x = ['NN', 'NN', 'familjen', 'Äktenskapet', 'AV', 'EN', 'är', 'en', True, False], y = re
x = ['NN', 'ROOT', 'Äktenskapet', 'ROOT', 'AV', 'EN', 'är', 'en', False, True], y = la.SS
x = ['ROOT', 'nil', 'ROOT', 'nil', 'AV', 'EN', 'är', 'en', True, False], y = ra.ROOTx = ['AV', 'ROOT', 'är', 'ROOT', 'EN', 'AJ', 'en', 'gammal', True, False], y = sh
'''
