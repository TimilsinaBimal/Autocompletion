from modeling import calculate_probabilities


def suggest_a_word(previous_tokens,
                   n_gram_counts,
                   n_plusone_gram_counts,
                   vocabulary,
                   k=1.0,
                   start_with=None):

    n = len(list(n_gram_counts.keys())[0])

    previous_n_gram = previous_tokens[-n:]

    probabilities = calculate_probabilities(previous_n_gram,
                                            n_gram_counts,
                                            n_plusone_gram_counts,
                                            vocabulary,
                                            k=k)
    max_probability = 0
    suggestion = None

    for word, prob in probabilities.items():
        if start_with:
            if not word.startswith(start_with):
                continue

        if prob > max_probability:
            max_probability = prob
            suggestion = word
    return (suggestion, max_probability)


def get_suggestions(previous_tokens,
                    n_gram_counts,
                    n_plusone_gram_counts,
                    vocabulary,
                    k=1.0,
                    n_suggestions=4,
                    starts_with=None):

    n = len(list(n_gram_counts.keys())[0])

    previous_n_gram = previous_tokens[-n:]

    probabilities = calculate_probabilities(
        previous_n_gram,
        n_gram_counts,
        n_plusone_gram_counts,
        vocabulary,
        k=k
    )

    sorted_probabilities = {
        word: probability for word, probability in
        sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    }
    if starts_with:
        suggestions = dict()
        for word, prob in sorted_probabilities.items():
            if word.startswith(starts_with):
                suggestions[word] = prob
            else:
                continue
        return list(suggestions.items())[:n_suggestions]

    return list(sorted_probabilities.items())[:n_suggestions]
