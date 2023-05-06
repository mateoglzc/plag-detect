import re

from preprocessing import pad_data, group_creation
from NgramCounter import NgramCounter
from NgramModel import NgramModel

from nltk.lm import WittenBellInterpolated

# N-grams

# Steps to implement
# 1.- Read in a pre-process the training data. (With this, remove punctuation, format, etc.)
# 2.- Tokenize training data. "this is a sentence" -> ["this", "is", "a", "sentence"]
# 3.- Generate n-grams from the training data
# 4.- Fit the model


if __name__ == "__main__": 

    # 1.- Preprocess training data

    training_data_path = "training.txt"

    with open(training_data_path, "r") as f:
        train_text = f.read().lower()
    
    training_data = pad_data(train_text)

    # N-Gram number
    N = 2

    ngrams = group_creation(training_data, N)

    # Test our model
    ngc = NgramCounter(ngrams)
    ngm = NgramModel(ngc, ngrams)
    ngm.train()

    # Build ngram language models
    model = WittenBellInterpolated(N) 
    model.fit([ngrams], vocabulary_text=training_data)

    test_data_file = "suspicious.txt"

    # Read testing data
    with open(test_data_file) as f:
        test_text = f.read().lower()
    test_text = re.sub(r'[^\w\s]', "", test_text)

    # Tokenize and pad the text
    testing_data = pad_data(test_text)

    # assign scores
    scores = []
    our_scores = []
    for i, item in enumerate(testing_data[N-1:]):
        s = model.score(item, testing_data[i:i+N-1])
        score_custom_model = ngm.score(item, testing_data[i:i+N-1][0])
        print(f"Predictability: {round(s, 2):4} Word: '{item}'")
        print(f"Our predictability: {round(score_custom_model, 2):4} Word: '{item}'")
        print("------")
        scores.append(s)
        our_scores.append(score_custom_model)

    print(f"avg nltk model: {sum(scores)/len(scores)}")
    print(f"avg our model: {sum(our_scores)/len(our_scores)}")

