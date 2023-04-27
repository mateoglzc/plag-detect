# Explicit Semantic Analysis Test
import os

def compare_texts(inverted_index: InvertedIndex, text: list) -> float:
    pass

def main():

    # Our text collection is our sample set of texts which we will use to compare our input text.
    # The main goal is to look for plagerism in this input text.  
    text_collection = ["text_one.txt", "text_two.txt"]

    # Inverted Index is a data structure that will help us query words quickly.
    inv_idx = InvertedIndex()

    # To start we analyze each text in our text collection.
    for i, text_name in enumerate(text_collection):

        # 1.- Create bag of words and count how many words are in the document
        amount_words, bag_words = analyze_text(f"./{text_name}")

        # 2.- Input bag of words into inverted index and calculate the score of each word
        inv_idx.add(bag_words, i, amount_words)

    # Once populated our Inverted Index, we analyze our input text

    input_text = "./input_text.txt"

    # Read input text 

    w = []

    with open(input_text, "r") as f:
        lines = f.readlines()
        for line in lines:
            words = line.split()
            for word in words:
                w.append(word.lower())






if __name__ == "__main__":
    main()