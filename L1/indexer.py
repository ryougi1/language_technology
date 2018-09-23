# import re
import regex as re
import os.path
import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import copy
import math
import numpy

master_dict = {}
count_dict = {}

def get_files(dir, suffix):
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    return files

def index(file):
    with open("Selma/" + file, "r") as unread:
        # pattern = r'[a-zA-ZåäöÅÄÖ]+(\s|\.|\?|,|!|;|:)'
        # regex = re.compile(pattern, re.IGNORECASE)

        # text = unread.read()
        text = unread.read().lower()
        count_dict[file] = len(text)
        for match in re.finditer('\p{L}+', text):
        # for match in regex.finditer(text):
            # word = match.group().lower()[:-1]
            word = match.group()
            if word not in master_dict:
                master_dict[word] = {file:[match.start()]}
            elif file not in master_dict[word]:
                master_dict[word][file] = [match.start()]
            else:
                master_dict[word][file].append(match.start())

def calc_tfidf(files):
    '''
    tfidf_vectorizer = TfidfVectorizer(input = 'filename', smooth_idf = False)
    t_d_matrix = tfidf_vectorizer.fit_transform(files)
    vocab = tfidf_vectorizer.vocabulary_
    for i in range(len(files)):
        print(files[i][6:], list(t_d_matrix.todense()[i].flat)[vocab["gås"]])
    '''

    #TF is term count / total word count, per document
    #IDF is log10(nr docs / nr docs word appears in)

    tfidf_table = copy.deepcopy(master_dict)
    for key, value in tfidf_table.items():
        idf = math.log10((len(files) / len(value)))
        tfidf_table[key] = {x:0 for x in files}
        for k, v in value.items():
            tfidf_table[key][k] = (len(v) / count_dict[k]) * idf

    return tfidf_table


def calc_cosine_similarities(tfidf_matrix):
    cosine_matrix = {}
    for f in files:
        cosine_matrix[f] = []
        for w in tfidf_matrix:
            cosine_matrix[f].append(tfidf_matrix[w][f])

    most_similar = (0, None, None)
    for k1, v1 in cosine_matrix.items():
        for k2, v2 in cosine_matrix.items():
            if k1 == k2:
                continue

            similarity = numpy.dot(v1, v2)/(numpy.linalg.norm(v1)*numpy.linalg.norm(v2)) #Should
            if similarity > most_similar[0]:
                most_similar = (similarity, k1, k2)

    print("Most similar documents: ", most_similar[1], most_similar[2], "with value ", most_similar[0])



if __name__ == '__main__':
    files = get_files("Selma", ".txt")
    for file in files:
        index(file)
    #print(master_dict.get("samlar"))
    #open("master_index.txt", "w").write(str(master_dict))
    tfidf_matrix = calc_tfidf(files)
    calc_cosine_similarities(tfidf_matrix)
    #for cos in cosine:
        #print(cos)

#pickle.dump(master_dict, open("bannlyst.txt", "wb"))
