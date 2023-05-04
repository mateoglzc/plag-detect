import os

def analyze_text(path: str):
    """
    Given a input path of a .txt file, return a bag of words vector of that given file and the amount of words that file contains.
    """

    # Get the extension of a file
    ext = os.path.splitext(path)[-1].lower()

    if ext != ".txt":
        raise Exception("Input file is not a text file")

    bag_words = {}
    amount_words = 0

    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # This split is not efficient enough. It will be better if we can omit punctuation and consider parethesis as one word
            words = line.split()
            for word in words:
                w = word.lower()
                bag_words[w] = bag_words.get(w, 0) + 1
                amount_words += 1

    return amount_words, bag_words

def preprocesing():
    pass

if __name__ == "__main__":
    preprocesing()