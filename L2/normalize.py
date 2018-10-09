import regex as re
import sys

'''
Insert <s> and </s> tags to delimit sentences.
Remove all punctuation and set all text to lower case.
'''
def normalize(corpus):
    with open(corpus, "r") as unread:
        text = unread.read()
        '''
        text = re.sub(r'\r?\n|\r', r'', text) # Removes all newlines and carriage returns
        text = re.sub(r'([\.\?!]|\.")(\s)([A-ZÅÖÄ]|")', r'\1\2\n<s> \3', text) # Inserts <s> at the start of all sentences.
        text = re.sub(r'([\.\?!])', r'\1 </s>', text) # Inserts </s> at the end of all sentences.
        text = re.sub(r'[\.\?!,"]', r'', text).lower() # Removes all punctuation and quotation marks, sets text to lower case.
        '''
        text = re.sub(r'\r?\n|\r', r'', text) # Removes all newlines and carriage returns
        text = re.sub(r'\s+([^\.\?!]*[\.\?!])', r'<s> \1 </s> \n', " " + text)
        text = re.sub(r'[\.\?!,"]', r'', text).lower() # Removes all punctuation and quotation marks, sets text to lower case.
        return text.split("\n")

if __name__ == '__main__':
    file_name = sys.argv[1]

    lines = normalize(file_name)
    for line in lines:
        print(line)
