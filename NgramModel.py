import math

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
        """
        Slightly modified version of good-turing smoothing.
        """
        # For some reason normal good-turing formula c* = ((c+1)N_c+1)/N_c and disc_factor = N_1/N is not working correctly.
        # Probability skyrockets and model is not accurate at all.
        # For that reason we use backoff. But the idea of good-turing is still the same.

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