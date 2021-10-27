from modeling import calculate_probability


def perplexity(sentence, n_gram_counts,
               n_plusone_gram_counts,
               vocabulary, k=1.0, log_perplexity=True):

    # find n as in n-grams
    n = len(list(n_gram_counts.keys())[0])

    sentence = ["<s>"] * n + sentence + ["<e>"]
    sentence = tuple(sentence)

    vocabulary_size = len(vocabulary)

    # N = len(sentence) if n == 1 else len(sentence)-1
    N = len(sentence)

    if log_perplexity:
        temp_sum = 0.0
    else:
        temp_product = 1.0

    for t in range(n, N):
        n_gram = sentence[t - n:t]
        # Because the loop starts at n i.e. it excludes first n words as
        # part of n-gram where t is current word so n-gram will be previous n
        # words

        word = sentence[t]

        probability = calculate_probability(word, n_gram,
                                            n_gram_counts,
                                            n_plusone_gram_counts,
                                            vocabulary, k=k)
        if log_perplexity:
            temp_sum += (1 / probability)
        else:
            temp_product *= (1 / probability)

    if log_perplexity:
        perplexity = - (1 / N) * temp_sum
    else:
        perplexity = temp_product**(1 / N)

    return perplexity
