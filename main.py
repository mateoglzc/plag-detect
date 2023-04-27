# Explicit Semantic Analysis Test
from math import sqrt
import os
from preprocesing import *
from inverted_index import *

def compare_texts(inverted_index: InvertedIndex, text: list) -> float:
    pass

def multiply_appearances(bag_of_words: InvertedIndex):
    all_words_dict = bag_of_words.index
    sum_of_all_multiplications = 0
    # print(all_words_dict)
    for value in all_words_dict:
        if 1 < len(all_words_dict[value]) : # this means that if the word appears in both texts, we'll do:
            print(value)
            sum_of_all_multiplications += all_words_dict[value][0].frequency * all_words_dict[value][1].frequency
    return sum_of_all_multiplications

def get_square_root(bag_of_words: dict):
    sum_of_appearance_square = 0
    for key, value in bag_of_words.items():
        sum_of_appearance_square+= value**2
    return sqrt(sum_of_appearance_square)


def cosine_similarity(first_text_path:str, second_text_path: str) -> float:
    first_amount, first_bag = analyze_text(first_text_path)
    second_amount, second_bag = analyze_text(second_text_path)
    all_words_index = InvertedIndex()
    all_words_index.add(first_bag, 1, first_amount)
    all_words_index.add(second_bag, 2, second_amount)
    # at this point, all words are stored in all_words_index.index

    # using this all_words_index, we get the summation of the frequency of the words that exist in both texts
    sum_of_words_present_in_both_texts = multiply_appearances(all_words_index)
    # we need to multiply thew roots of the power of each word in both texts 
    # (Hopefully makes sense)
    first_root = get_square_root(first_bag)
    second_root = get_square_root(second_bag)
    sum_of_roots = first_root * second_root
    return sum_of_words_present_in_both_texts/sum_of_roots
    # una para cada texto
    # diccionario para cada una, con estos diccionarios hacemos el cosine similarity

def main():

    # Our text collection is our sample set of texts which we will use to compare our input text.
    # The main goal is to look for plagerism in this input text.  
    text_collection = ["text_one.txt", "text_two.txt"]
    cosSimilarity = cosine_similarity(text_collection[0], text_collection[1])
    print(cosSimilarity)
    # Inverted Index is a data structure that will help us query words quickly.
    # inv_idx = InvertedIndex()

    # # To start we analyze each text in our text collection.
    # for i, text_name in enumerate(text_collection):

    #     # 1.- Create bag of words and count how many words are in the document
    #     amount_words, bag_words = analyze_text(f"./{text_name}")

    #     # 2.- Input bag of words into inverted index and calculate the score of each word
    #     inv_idx.add(bag_words, i, amount_words)

    # # Once populated our Inverted Index, we analyze our input text

    # input_text = "./input_text.txt"

    # # Read input text 

    # w = []

    # with open(input_text, "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         words = line.split()
    #         for word in words:
    #             w.append(word.lower())






if __name__ == "__main__":
    main()