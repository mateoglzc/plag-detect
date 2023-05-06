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