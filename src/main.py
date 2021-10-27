from preprocessing import handle_unk_words, load_data, tokenize, create_vocabulary
from suggestions import get_suggestions
import pickle
from nltk.tokenize import word_tokenize

DATA_PATH = "data/en_US.twitter.txt"


def preprocess_data(filepath, min_frequency=0, max_words=0, return_data=False):
    data = load_data(filepath)
    tokenized_data = tokenize(data, return_sentences=False)
    vocabulary = create_vocabulary(
        tokenized_data, min_frequency=min_frequency, max_words=max_words)
    if return_data:
        preprocessed_data = handle_unk_words(tokenized_data, vocabulary)
        return vocabulary, preprocessed_data

    return vocabulary


def load_models():
    unigram_counts = pickle.load("models/unigram_counts_2.pkl")
    bigram_counts = pickle.load("models/bigram_counts_2.pkl")
    vocabulary = pickle.load("models/vocabulary_2.pkl")
    return unigram_counts, bigram_counts, vocabulary


def get_suggestion(sentence, starts_with=None, n_suggestions=2):
    tokenized_sentence = word_tokenize(sentence)
    unigram_counts = pickle.load(open("models/unigram_counts_2.pkl", 'rb'))
    bigram_counts = pickle.load(open("models/bigram_counts_2.pkl", 'rb'))
    vocabulary = pickle.load(open("models/vocabulary_2.pkl", 'rb'))

    suggestions = get_suggestions(
        tokenized_sentence, unigram_counts, bigram_counts, vocabulary, n_suggestions=n_suggestions, starts_with=starts_with)
    return suggestions


def main():
    sentence = input("Enter the sentence: ")
    starts_with = input(
        "Enter the letter from which you want your suggestions to starts with(Enter 0 if you want all suggestions): ")
    try:
        if not int(starts_with):
            starts_with = None
    except ValueError:
        pass
    n_suggestions = int(input("Enter the number of suggestions you want: "))
    probability = input(
        "Do you want to get probabilities or just words? [Y] for Yes and [N] for No: ")

    suggestions = get_suggestion(
        sentence, starts_with=starts_with, n_suggestions=n_suggestions)
    if probability.lower() == 'y':
        print(suggestions)
    elif probability.lower() == 'n':
        words = [word[0] for word in suggestions]
        print(words)


if __name__ == "__main__":
    main()
