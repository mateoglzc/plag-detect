import re

from nltk.util import ngrams, pad_sequence, everygrams
from nltk.tokenize import word_tokenize
from nltk.lm import MLE, WittenBellInterpolated

import numpy as np

# N-grams

# Steps to implement
# 1.- Read in a pre-process the training data. (With this, remove punctuation, format, etc.)
# 2.- Tokenize training data. "this is a sentence" -> ["this", "is", "a", "sentence"]
# 3.- Generate n-grams from the training data
# 4.- Fit the model

def pad_data(tokenize_data: list) -> list:
    """
    Pad the training data by sentences.
    Ex.
    i am Sam.
    Sam am i.
    i don't like green eggs and ham.
    ->
    [<s>, i, am, sam, <\s>, <s>, sam, i, am, <\s>, i, don't, like, green, eggs, and, ham, <\s>]
    <s> -> start sentence
    <\s> -> end sentence
    """
    pass

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

    # Remove punctuation
     
    train_text = re.sub(r"\[.*\]|\{.*\}", "", train_text)
    train_text = re.sub(r'[^\w\s]', "", train_text)

    # N-Gram number
    N = 4

    # Pad Text (Basically just insert '<s>' in the beginning)
    training_data = list(pad_sequence(word_tokenize(train_text), N, 
                                  pad_left=True, 
                                  left_pad_symbol="<s>"))

    print(f"Training data: {training_data}\n")

    # Generate ngrams
    ngrams = list(everygrams(training_data, max_len=N))

    print("Number of ngrams:", len(ngrams))

    # Build ngram language models
    model = WittenBellInterpolated(N) # Quien sabe que tipo de modelo es este?
    model.fit([ngrams], vocabulary_text=training_data)
    print(model.__dict__)

    # testing data file
    test_data_file = "text_two.txt"

    # Read testing data
    with open(test_data_file) as f:
        test_text = f.read().lower()
    test_text = re.sub(r'[^\w\s]', "", test_text)

    # Tokenize and pad the text
    testing_data = list(pad_sequence(word_tokenize(test_text), N, 
                                    pad_left=True,
                                    left_pad_symbol="<s>"))
    print("Length of test data:", len(testing_data))

    # assign scores
    scores = []
    for i, item in enumerate(testing_data[N-1:]:
        s = model.score(item, testing_data[i:i+N-1])
        print(f"Predictability: {round(s, 2):4} Word: '{item}'")
        scores.append(s)

    print(f"avg: {sum(scores)/len(scores)}")

