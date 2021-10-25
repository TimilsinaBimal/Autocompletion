import pickle
from main import preprocess_data
from preprocessing import count_n_grams

FILEPATH = "data/en-US.twitter.txt"


def save(data, n_gram, vocabulary_condition):
    n_gram_counts = count_n_grams(data, n_gram)
    pickle.dump(n_gram_counts, open(
        f"models/{n_gram}-gram_{vocabulary_condition}.pkl", "wb"))


min_frequency = int(input(
    "Enter minimum frequency of words for vocabulary (Enter 0 if you want to specify maximum number of words): "))

if not min_frequency:
    max_words = int(input("Enter maximum number of words in vocabulary: "))

n_gram = int(input("Enter n of n_gram(eg. for bigram enter 2 and so on.): "))

if min_frequency:
    _, data = preprocess_data(
        FILEPATH,
        min_frequency=min_frequency, return_data=True)
    save(data, n_gram, min_frequency)

else:
    _, data = preprocess_data(
        FILEPATH,
        max_words=max_words, return_data=True)
    save(data, n_gram, max_words)
