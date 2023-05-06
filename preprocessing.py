import os
import re

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

def pad_data(full_text: str) -> list:
    """
    Pad the training data by sentences. Remove puncutation, quotes, parenthesis and commas.
    Ex.
    "This is a sentence." -> ["<s>", "this", "is", "a", "sentence", "<\s>"]
    """
    # get paragraphs
    paragraphs = full_text.split("\n")
    sentences = []
    for paragraph in paragraphs:
        paragraphSentences = paragraph.split(". ")
        sentences = sentences + paragraphSentences
    
    sentences = [sentence.replace(".", "") for sentence in sentences]
    sentences = [sentence.replace(",", "") for sentence in sentences]
    new_sentences = []
    for sentence in sentences:
        pattern = r'"[^"]*"\s*\([^)]*\)'  # matches the format of "quote" (citation)
        sentece = re.sub(pattern, '', sentence)
        new_sentences.append(sentece)
    sentences = new_sentences

    pad_list = []

    for sentence in sentences:
        pad_list.append("<s>")
        splitted = sentence.split()
        pad_list.extend(splitted)
        pad_list.append("<\s>")
    found_quotes = False
    found_parenthesis = False
    i = 0
    for word in pad_list:
        if '"' in word and found_quotes == False:
            without_quote = word.replace('"', "")
            pad_list.pop(i)
            pad_list.insert(i, without_quote)
            pad_list.insert(i, "<q>")
            found_quotes = True
        elif '"' in word and found_quotes == True:
            without_quote = word.replace('"', "")
            pad_list.pop(i)
            pad_list.insert(i, without_quote)
            pad_list.insert(i+1, "<\q>")
            found_quotes = False
        if '(' in word and found_parenthesis == False:
            without_quote = word.replace('(', "")
            pad_list.pop(i)
            pad_list.insert(i, without_quote)
            pad_list.insert(i, "<p>")
            found_parenthesis = True
        elif ')' in word and found_parenthesis == True:
            without_quote = word.replace(')', "")
            pad_list.pop(i)
            pad_list.insert(i, without_quote)
            pad_list.insert(i+1, "<\p>")
            found_parenthesis = False

        i+=1
    return pad_list


def group_creation(pad_list: list, n_gram: int):
    """
    Given a list of tokens already padded and a ngram number. Create ngram groups.
    Ex.
    N = 2
    ["<s>", "this", "is", "a", "sentence", "<\s>"] -> [("<s>",), ("<s>", "this"), ("this",), ("this","is"), ("a",), ("a", "sentence"), ("sentence",), ("sentence", "<\s>"), ("</s>",)]
    """
    grouped_list = []
    for i in range(len(pad_list)):
        current_group = []
        j= 0
        while j < n_gram and i < len(pad_list)-n_gram:
            current_group.append(pad_list[i+j])
            tuple_to_append = tuple(current_group)
            grouped_list.append(tuple_to_append)
            j +=1
    for i in range(len(pad_list)-n_gram, len(pad_list)):
        for j in range(i, len(pad_list)):
            grouped_list.append(tuple(pad_list[i:j+1]))
    
    return grouped_list