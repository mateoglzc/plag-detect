import re
import math

from nltk.util import ngrams, pad_sequence, everygrams
from nltk.tokenize import word_tokenize
from nltk.lm import MLE, WittenBellInterpolated

# N-grams

# Steps to implement
# 1.- Read in a pre-process the training data. (With this, remove punctuation, format, etc.)
# 2.- Tokenize training data. "this is a sentence" -> ["this", "is", "a", "sentence"]
# 3.- Generate n-grams from the training data
# 4.- Fit the model
class NgramCounter():
    
    def __init__(self, ngrams: list) -> None:
        self.ngrams = ngrams
        self.count()

    def count(self) -> None:

        self.token_appearance = {}
        self.unique_tokens = set() # Tokens that have only appeared once
        self.n_tokens_appereance = {} # Used for good-turing smoothing

        for ngram in self.ngrams:
            self.token_appearance[ngram] = self.token_appearance.get(ngram, 0) + 1

        self.total_tokens = sum(self.token_appearance.values())

        for token, count in self.token_appearance.items():
            if count == 1:
                self.unique_tokens.add(token)
            
            self.n_tokens_appereance[count] = self.n_tokens_appereance.get(count, 0) + 1

class NgramModel():

    def __init__(self, counter, ngrams):

        self.ngram_counter = counter
        self.ngrams = ngrams

        self.vocab = set([token[0] for token in self.ngrams if len(token) == 1])


    def _markov_assumption(self, word, prev_word) -> float:
        """P(w1 | w2) = count(W2 w1) / count(W2)"""
        return self.ngram_counter.token_appearance[(prev_word, word)]/self.ngram_counter.token_appearance[(prev_word,)]

    def calculate_ngram_probability(self, ngram: tuple) -> None:

        if len(ngram) == 1:
            # If our ngrams are not of size 2 then move on
            count = self.ngram_counter.token_appearance[ngram]
            self.uni_prob[ngram] = count/sum(self.ngram_counter.token_appearance.values())
            return

        count = self.ngram_counter.token_appearance

        # Markov assumption
        self.prob[ngram] = self._markov_assumption(ngram[1],ngram[0])

    def train(self) -> None:

        # The probability that each ngram appears on the text
        self.prob = {}
        self.uni_prob = {}
        # The propability that each ngram appears at the start of a sentence
        self.start_prob = {}

        for ngram in self.ngrams:
            self.calculate_ngram_probability(ngram)

    def score(self, word, prev_word) -> float:
        """
        Given the a bigram of the test set calculate the probability of that bigram happening
        """
    
        # Calculate probability
        return self.good_turing_smoothing(word, prev_word)

    def _perplexity(self, ngrams) -> float:
        """
        This belongs in evaluation and the ngrams used in this are ngrams of the test set.
        """
        p = 0
        for ngram in ngrams:
            if len(ngram) == 1:
                continue

            if ngram not in self.prob:
                self.calculate_ngram_probability(ngram)
            
            p += math.log(1/self.prob[ngram])

        p = math.ceil(math.e**(p))

        return math.sqrt(p)


    def laplace_smoothing(self, word, prev_word) -> float:
        v = len(self.vocab)

        bigram = (prev_word, word)
        unigram = (prev_word,)

        prob_nom = self.ngram_counter.token_appearance.get(bigram, 0) + 1
        prob_den = self.ngram_counter.token_appearance.get(unigram, 0) + v

        return prob_nom/prob_den

    def good_turing_smoothing(self, word, prev_word) -> float:

        bigram = (prev_word, word)
        unigram = (prev_word,)

        # Calculate discounting factor
        # disc_factor = (number_unique_events - number_of_events_occurred_twice)/total_events
        disc_factor = (self.ngram_counter.n_tokens_appereance[1] - self.ngram_counter.n_tokens_appereance[2])/self.ngram_counter.total_tokens

        lmb = 0.35

        # Calculate probability
        prob_nom = self.ngram_counter.token_appearance.get(bigram, 0) * disc_factor
        prob_den = self.ngram_counter.token_appearance.get(unigram, 0)

        if prob_nom == 0 or prob_den == 0:
            prob_unigram = 0
            if unigram in self.uni_prob:
                prob_unigram = self.uni_prob[unigram]
                
            return lmb * prob_unigram + (1 - lmb) * 0.02
        
        return prob_nom/prob_den

def pad_data(full_text: str) -> list:
    """
    Pad the training data by sentences.
    """
    # get paragraphs
    paragraphs = full_text.split("\n")
    #print(f"paragraphs:{paragraphs}")
    sentences = []
    for paragraph in paragraphs:
        paragraphSentences = paragraph.split(". ")
        sentences = sentences + paragraphSentences
    
    sentences = [sentence.replace(".", "") for sentence in sentences]
    sentences = [sentence.replace(",", "") for sentence in sentences]
    #print(f"sentence v2: {sentences}")

    pad_list = []

    for sentence in sentences:
        pad_list.append("<s>")
        splitted = sentence.split()
        pad_list.extend(splitted)
        pad_list.append("<\s>")
    #print(f"pad_list: {pad_list}")
    return pad_list
    

def group_creation(pad_list: list, n_gram: int):
    print(range(len(pad_list)))
    print(f"pad_list: {pad_list}\n")
    grouped_list = []
    for i in range(len(pad_list)):
        # print(f"i: {i}")
        current_group = []
        j= 0
        while j < n_gram and i < len(pad_list)-n_gram:
            #print(f"j: {j}")
            # print(pad_list[i+j])
            current_group.append(pad_list[i+j])
            print(current_group)
            tuple_to_append = tuple(current_group)
            grouped_list.append(tuple_to_append)
            j +=1
    for i in range(len(pad_list)-n_gram, len(pad_list)):
        for j in range(i, len(pad_list)):
            grouped_list.append(tuple(pad_list[i:j+1]))
    
    print(grouped_list)
    return grouped_list


if __name__ == "__main__": 

    # 1.- Preprocess training data

    training_data_path = "training.txt"

    with open(training_data_path, "r") as f:
        train_text = f.read().lower()
    
    training_data = pad_data(train_text)

    # N-Gram number
    N = 2

    ngrams = group_creation(training_data, N)

    # print(f"N-grams: {ngrams}")

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

