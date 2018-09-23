import sys
import regex as re
import math

def normalize(corpus):
    with open(corpus, "r") as unread:
        text = unread.read()
        text = re.sub(r'\r?\n|\r', r'', text) # Removes all newlines and carriage returns
        text = re.sub(r'([\.\?!]|\.")(\s)([A-ZÅÖÄ]|")', r'\1\2\n<s> \3', text) # Inserts <s> at the start of all sentences.
        text = re.sub(r'([\.\?!])', r'\1 </s>', text) # Inserts </s> at the end of all sentences.
        text = re.sub(r'[\.\?!,"]', r'', text).lower() # Removes all punctuation and quotation marks, sets text to lower case.
        return text

def tokenize(corpus_norm):
        words = re.findall("\p{L}+", corpus_norm)
        return words

# Reused from Pierre Nugues
def count_unigrams(words):
    frequency = {}
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1
    return frequency

# Reused from Pierre Nugues
def count_bigrams(words):
    bigrams = [tuple(words[inx:inx + 2])
               for inx in range(len(words) - 1)]
    frequencies = {}
    for bigram in bigrams:
        if bigram in frequencies:
            frequencies[bigram] += 1
        else:
            frequencies[bigram] = 1
    return frequencies

# Reused from Pierre Nugues
def print_ngrams(frequencies):
    for word in sorted(frequencies.keys(), key=frequencies.get, reverse=True):
        print(frequencies[word], '\t', word)

def print_prob_unigram(sentence, frequencies, words):
    print("UNIGRAM MODEL")
    print("=====================================================")
    print('wi', '\t', 'C(wi)', '\t', '#words', '\t', 'P(wi)')
    print("=====================================================")
    cum_probability = 1
    entropy = 0
    for word in sentence:
        prob = frequencies[word]/len(words)
        print(word, '\t', frequencies[word], '\t', len(words), '\t', prob)
        cum_probability = cum_probability * prob # https://en.wikipedia.org/wiki/Geometric_mean
        entropy = entropy + math.log2(prob) # https://github.com/pnugues/ilppp/blob/master/slides/EDAN20_ch05.pdf slide 38

    print("Prob. unigrams: ", '\t', cum_probability)
    print("Geometric mean prob.:", '\t', cum_probability**(1.0/len(sentence))) # https://en.wikipedia.org/wiki/Geometric_mean
    entropy = -1 / len(sentence) * entropy # https://github.com/pnugues/ilppp/blob/master/slides/EDAN20_ch05.pdf slide 38
    print("Entropy rate: ", '\t', entropy)
    print("Perplexity: ", '\t', math.pow(2, entropy)) # https://github.com/pnugues/ilppp/blob/master/slides/EDAN20_ch05.pdf slide 38

def print_prob_bigram(sentence, frequencies, words):
    print("BIGRAM MODEL")
    print("=====================================================")
    print('wi', '\t', 'wi+1', '\t', 'Ci,i+1', '\t', 'C(idea)', '\t', 'P(wi+1|wi)')
    print("=====================================================")

    

if __name__ == '__main__':
    normalized_text = normalize("Selma.txt")
    words = tokenize(normalized_text)
    frequency_unigrams = count_unigrams(words)
    frequency_bigrams = count_bigrams(words)
    # print_ngrams(frequency_unigrams)
    # print_ngrams(frequency_bigrams)

    sentence = sys.argv[1].lower().split()
    print_prob_unigram(sentence, frequency_unigrams, words)
    print_prob_bigram(sentence, frequency_bigrams, words)
