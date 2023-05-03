import re

from nltk.util import ngrams, pad_sequence, everygrams
from nltk.tokenize import word_tokenize
from nltk.lm import MLE, WittenBellInterpolated

# import numpy as np

# N-grams

# Steps to implement
# 1.- Read in a pre-process the training data. (With this, remove punctuation, format, etc.)
# 2.- Tokenize training data. "this is a sentence" -> ["this", "is", "a", "sentence"]
# 3.- Generate n-grams from the training data
# 4.- Fit the model

def pad_data(full_text: str) -> list:
    """
    Pad the training data by sentences.
    """

    # Separate text into sentences
    sentences = full_text.split(". ")
    sentences = [s.strip(".") for s in sentences]
    sentences = [s.strip("./n") for s in sentences]
    pad_list = []
    for i in sentences:
        pad_list.append("<s>")
        splitted_sentence = i.split()
        pad_list.extend(splitted_sentence)
        pad_list.append("<\s>")
    sentences = full_text.split(". ")
    sentences = [s.strip(".") for s in sentences]
    sentences = [s.strip("./n") for s in sentences]
    pad_list = []
    for i in sentences:
        pad_list.append("<s>")
        splitted_sentence = i.split()
        pad_list.extend(splitted_sentence)
        pad_list.append("<\s>")
    
    return pad_list
    

def group_creation(pad_list: list):
    grouped_list = []
    for i in range(len(pad_list)-1):
        pair = (pad_list[i],)
        grouped_list.append(pair)
        pair = (pad_list[i], pad_list[i+1])
        grouped_list.append(pair)
    pair = (pad_list[-1],)
    grouped_list.append(pair)
    
    return grouped_list

def markov_assumption(ngrams_set: list, ngram: int):
    pass 

def count(word, prev_word, tokenize_data) -> int:
    found_prev = False
    c = 0
    for token in tokenize_data:
        if token == prev_word:
            found_prev = True
        elif found_prev and toke == word:
            c += 1
            found_prev = False

    return c

def sequence_probability(word: str, prev_word: str, bag_words: dict, tokenize_data: list) -> float:
    return count(word, prev_word, tokenize_data)/bag_words(prev_word)

if __name__ == "__main__": 

    # 1.- Preprocess training data

    training_data_path = "text_one.txt"

    with open(training_data_path, "r") as f:
        train_text = f.read().lower()
    
    training_data = pad_data(train_text)

    print(f"Our padded data: {training_data}")

    ngrams = group_creation(training_data)

    print(f"N-grams: {ngrams}")

    # N-Gram number
    N = 2

    # Build ngram language models
    model = WittenBellInterpolated(N) 
    model.fit([ngrams], vocabulary_text=training_data)

    test_data_file = "./text_two.txt"

    # Read testing data
    with open(test_data_file) as f:
        test_text = f.read().lower()
    test_text = re.sub(r'[^\w\s]', "", test_text)

    # Tokenize and pad the text
    testing_data = pad_data(test_text)
    print(f"Testing data: {testing_data}")

    # assign scores
    scores = []
    for i, item in enumerate(testing_data[N-1:]):
        s = model.score(item, testing_data[i:i+N-1])
        print(f"Predictability: {round(s, 2):4} Word: '{item}'")
        scores.append(s)

    print(f"avg: {sum(scores)/len(scores)}")

