from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split


def load_data(filename):
    with open(filename, "rb") as data:
        text = data.read()
    text = text.decode("utf-8")
    return text


def tokenize(data, return_sentences=True):
    sentences = data.split("\r\n")
    sentences = [sent.strip() for sent in sentences]
    sentences = [sent for sent in sentences if len(sent) > 0]
    word_tokens = [word_tokenize(sent.lower()) for sent in sentences]
    if return_sentences:
        return sentences, word_tokens
    return word_tokens


def split_data(word_tokens, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(
        word_tokens, test_size=0.2, random_state=42)
    return train_data, test_data


def count_words(data):
    cnt = Counter()
    for words in data:
        cnt.update(words)
    return cnt


def create_vocabulary(data, min_frequency=0, max_words=0):
    word_count = count_words(data)
    if min_frequency > 0:
        vocabulary = []
        for word, count in word_count.items():
            if count >= min_frequency:
                vocabulary.append(word)
    if max_words > 0:
        vocabulary = [word[0] for word in word_count.most_common(max_words)]

    return vocabulary


def handle_unk_words(data, vocabulary, unk_token="<unk>"):
    handled_data = []
    for sentences in data:
        sentence = []
        for word in sentences:
            if word in vocabulary:
                sentence.append(word)
            else:
                sentence.append(unk_token)
        handled_data.append(sentence)
    return handled_data


def count_n_grams(data, n, start_token="<s>", end_token="<e>"):
    """
    Define and count the numbers of n-grams in text corpus.

    Args:
        data : preprocessed text corpus
        n : n in n_gram
        start_token: starting token symbol
        end_token: ending token symbol

    Returns:
        n_grams: Dictionary containing n_gram tuples as keys and their counts as
        values
    """

    n_grams = defaultdict(int)
    for sentence in data:
        sentence = [start_token] * n + sentence + [end_token]

        """
            If we are calculating 1-gram we need to include all single words so we
            need to loop over all the words of sentence
            but if n > 2 we need to include last word only once, so we cannot
            loop over all the words but n-1
            e.g. ```
                A = ["A","B","C","D"]
                if n=1, then len = len(A) = 4,
                A1 = ["A"], A2 = ["B"], A3=["C"], A4=["D"]
                if n> 2,
                then len = len(A)-1 = 3
                A1 = ["A","B"], A2 = ["B","C"] A3 = ["C","D"]
            ```
        """
        m = len(sentence) if n == 1 else len(sentence) - 1

        for i in range(m):
            n_gram = sentence[i:i + n]
            n_gram = tuple(n_gram)
            n_grams[n_gram] += 1
    return n_grams
