class Appearance:

    def __init__(self, docId, frequency, score=0):
        self.docId = docId
        self.frequency = frequency  
        self.score = score 
    def __repr__(self):
        """
        String representation of the Appearance object
        """
        return str(self.__dict__)

    
class InvertedIndex:
    """
    Inverted Index class.
    """
    def __init__(self):
        self.index = dict()

    def __repr__(self):
        """
        String representation of the Database object
        """
        return str(self.index)   

    def _calculate_score(self, frequency: int, total_word_count: int) -> float:
        return frequency/total_word_count

    def add(self, bag_words: dict, docId: int, total_word_count: int) -> None:

        for word, frequency in bag_words.items():
            app = Appearance(docId, frequency, self._calculate_score(frequency, total_word_count))
            if word in self.index:
                self.index[word].append(app)
            else: 
                self.index[word] = [app]
            
    def query(self, query: str) -> dict:
        return {term: self.index[term] for term in query.split()}