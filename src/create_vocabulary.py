import pickle
from main import preprocess_data

FILEPATH = "data/en-US.twitter.txt"

min_frequency = int(input(
    "Enter minimum frequency of words for vocabulary (Enter 0 if you want to specify maximum number of words): "))

if not min_frequency:
    max_words = int(input("Enter maximum number of words in vocabulary: "))
if min_frequency:
    vocabulary = preprocess_data(FILEPATH, min_frequency=min_frequency)
    pickle.dump(vocabulary, open(
        f"models/vocabulary_{min_frequency}.pkl", "wb"))
else:
    vocabulary = preprocess_data(FILEPATH, max_words=max_words)
    pickle.dump(vocabulary, open(
        f"models/vocabularymax_{max_words}.pkl", "wb"))
