def calculate_probability(word, previous_n_gram,
                          n_gram_counts, n_plusone_gram_counts,
                          vocabulary_size,
                          k=1):

    """
    Calculates the probability of given word in sentences given the previous n
    words (n-gram)

    Args:
        word: current word
        previous_n_gram: Previous n words
        n_gram_counts : dictionary of n-gram counts
        n_plusone_gram_counts: dictionary of (n+1)-gram counts
        vocabulary_size: Size of the vocabulary
        k: smoothing constant

    Returns:
        Probability
        P = {count(previous_n_gram + current_word) + k}
            / {count(previous_n_gram)+ k * voc_size}
    """

    n_plus_one_gram = tuple(previous_n_gram + [word])
    n_plus_one_gram_count = n_plusone_gram_counts[n_plus_one_gram] \
                    if n_plus_one_gram in n_plusone_gram_counts else 0
    numerator = n_plus_one_gram_count + k

    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts[previous_n_gram] \
                    if previous_n_gram in n_gram_counts else 0

    denominator = previous_n_gram_count + k * vocabulary_size


    probability = numerator / denominator
    return probability


def calculate_probabilities(previous_n_gram,
                            n_gram_counts, n_plusone_gram_counts,
                            vocabulary, k=1.0):
    probabilities = dict()
    vocabulary = vocabulary + ["<e>","<unk>"]
    vocabulary_size = len(vocabulary)

    for word in vocabulary:
        probability = calculate_probability(word,
                                            previous_n_gram,
                                            n_gram_counts,
                                            n_plusone_gram_counts,
                                            vocabulary_size,
                                            k=1.0)
        probabilities[word] = probability

    return probabilities