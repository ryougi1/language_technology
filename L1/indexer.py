# import re
import regex as re
import os.path
import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.metrics.pairwise import cosine_similarity
import copy
import math

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
        for k, v in value.items():
            tfidf_table[key][k] = (len(v) / count_dict[k]) * idf

    print(tfidf_table["et"]["bannlyst.txt"])

def calc_cosine_similarities(tfidf_matrix):
    cosines = [None] * len(files)

    for x in range(len(files)):
        cosines[x] = cosine_similarity(tfidf_matrix[x:x+1], tfidf_matrix)

    return cosines


if __name__ == '__main__':
    files = get_files("Selma", ".txt")
    for file in files:
        index(file)

    # open("master_index.txt", "w").write(str(master_dict))
    tfidf_matrix = calc_tfidf(files)

#pickle.dump(master_dict, open("bannlyst.txt", "wb"))

#for key, value in master_dict.items():
    #print(key, value)

#for key, value in master_dict.items():
    #if (key == "samlar" or key == "ände"):
        #print(key, value)

#print(master_dict)
