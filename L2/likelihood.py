import sys
import regex as re
import math

unigram_frequency = {}
bigram_frequencies = {}
words = []

def learn(corpus):
    with open(corpus, "r") as unread:
        text = unread.read()
        # text = re.sub(r'\r?\n|\r', r'', text) # Removes all newlines and carriage returns
        # text = re.sub(r'([\.\?!]|\.")(\s)([A-ZÅÖÄ]|")', r'\1\2\n<s> \3', text) # Inserts <s> at the start of all sentences.
        # text = re.sub(r'([\.\?!])', r'\1 </s>', text) # Inserts </s> at the end of all sentences.
        text = re.sub(r'\s+([^\.\?!]*[\.\?!])', r'<s> \1 </s>', " " + text)
        text = re.sub(r'[\.\?!,"]', r'', text).lower() # Removes all punctuation and quotation marks, sets text to lower case.

    global words # Works normally for dicts but not lists?
    words = re.findall("\p{L}+", text)

    '''
    Reused from Pierre Nugues
    '''
    for word in words:
        if word in unigram_frequency:
            unigram_frequency[word] += 1
        else:
            unigram_frequency[word] = 1

    bigrams = [tuple(words[inx:inx + 2])
               for inx in range(len(words) - 1)]

    for bigram in bigrams:
        if bigram in bigram_frequencies:
            bigram_frequencies[bigram] += 1
        else:
            bigram_frequencies[bigram] = 1

    print("Learning from", corpus, "complete. Total word count:", len(words))

# Reused from Pierre Nugues
def print_ngrams(frequencies):
    for word in sorted(frequencies.keys(), key=frequencies.get, reverse=True):
        print(frequencies[word], '\t', word)

def print_prob_unigram(sentence):
    print("UNIGRAM MODEL")
    print("=====================================================")
    print('wi', '\t', 'C(wi)', '\t', '#words', '\t', 'P(wi)')
    print("=====================================================")
    cum_probability = 1
    entropy = 0
    for word in sentence:
        unigram_freq = unigram_frequency.get(word)
        if (unigram_freq is None):
            unigram_freq = 0
        prob = unigram_freq / len(words)
        print(word, '\t', unigram_freq, '\t', len(words), '\t', prob)
        cum_probability = cum_probability * prob # https://en.wikipedia.org/wiki/Geometric_mean
        entropy = entropy + math.log2(prob) # https://github.com/pnugues/ilppp/blob/master/slides/EDAN20_ch05.pdf slide 38

    print("\nProb. unigrams: ", '\t', cum_probability)
    print("Geometric mean prob.:", '\t', cum_probability**(1.0/len(sentence))) # https://en.wikipedia.org/wiki/Geometric_mean
    entropy = -1 / len(sentence) * entropy # https://github.com/pnugues/ilppp/blob/master/slides/EDAN20_ch05.pdf slide 38
    print("Entropy rate: ", '\t\t', entropy)
    print("Perplexity: ", '\t\t', math.pow(2, entropy), '\n\n') # https://github.com/pnugues/ilppp/blob/master/slides/EDAN20_ch05.pdf slide 38

def print_prob_bigram(sentence):
    print("BIGRAM MODEL")
    print("=====================================================")
    print('wi', '\t', 'wi+1', '\t', 'Ci,i+1', '\t', 'C(i)', '\t', 'P(wi+1|wi)')
    print("=====================================================")

    cum_probability = 1
    entropy = 0
    bigrams = [tuple(sentence[inx:inx + 2])
               for inx in range(len(sentence) - 1)]

    for bigram in bigrams:
        bigram_freq = bigram_frequencies.get(bigram)
        if bigram_freq is None:
            bigram_freq = 0
        unigram_freq = unigram_frequency.get(bigram[0])
        if unigram_freq is None:
            unigram_freq = 0

        prob = bigram_freq / unigram_freq
        if not prob:
            # If probability is 0, use back off value https://github.com/pnugues/ilppp/blob/master/slides/EDAN20_ch05.pdf slide 35
            backoff_val = unigram_frequency.get(bigram[1]) / len(words)
            if backoff_val is None:
                backoff_val = 0
            cum_probability = cum_probability * backoff_val
            entropy = entropy + math.log2(backoff_val)
        else:
            cum_probability = cum_probability * prob
            entropy = entropy + math.log2(prob)

        print(bigram[0], '\t', bigram[1], '\t', bigram_freq, '\t\t', unigram_freq, '\t', ("* backoff:" + str(backoff_val)) if not prob else prob)

    print("\nProb. bigrams: ", '\t', cum_probability)
    print("Geometric mean prob.:", '\t', cum_probability**(1.0/(len(sentence) - 1)))
    entropy = -1 / (len(sentence) - 1) * entropy
    print("Entropy rate: ", '\t\t', entropy)
    print("Perplexity: ", '\t\t', math.pow(2, entropy), '\n\n')


if __name__ == '__main__':
    '''
    LEARNING
    '''
    learn("Selma.txt")

    '''
    COMPUTING LIKELIHOOD
    '''
    # sentence = sys.argv[1].lower().split()
    sentence = str("Det var en gång en katt som hette Nils").lower().split()
    print("Calculating for sentence:", ' '.join(sentence), '\n')
    print_prob_unigram(sentence)
    print_prob_bigram(sentence)
