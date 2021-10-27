# Autocompletion Using N-Gram Language Model
This project implements n-gram language model for autocompletion of any sentence. It will suggest next word given n previous words. **n** can be defined by user.

**The project currently assumes only one previous words to suggest next word. You can change it running ```change_n_gram_count.py``` file and by changing ```get_suggestion()``` function of ```src/main.py```**

### Change N-gram counts
1. First Run ```change_n_gram_count.py``` using:
    ```
    python src/change_n_gram_count.py
    ```
    and follow the steps accordingly. This will create and save n-gram count pickle file in models folder.
2. Now Go to ```src/main.py``` and change the following function(starts from line 28):
```python
def get_suggestion(sentence, starts_with=None, n_suggestions=2):
    tokenized_sentence = word_tokenize(sentence)
    n_gram_count_filepath = "models/unigram_counts_2.pkl"
    nplusone_gram_count_filepath = "models/bigram_counts_2.pkl"
    unigram_counts = pickle.load(open(n_gram_count_filepath, 'rb'))
    bigram_counts = pickle.load(open(nplusone_gram_count_filepath, 'rb'))
    vocabulary = pickle.load(open("models/vocabulary_2.pkl", 'rb'))

    suggestions = get_suggestions(
        tokenized_sentence, unigram_counts, 
        bigram_counts, vocabulary, n_suggestions=n_suggestions, 
        starts_with=starts_with)
    return suggestions
```

Here change line no 2 i.e. 
```python 
n_gram_count_filepath = "models/unigram_counts_2.pkl"
``` 
to your n-gram count filepath. For eg. if you are creating 2-gram count
then the file will be saved as *2-gram_2.pkl* where first 2 denotes 2-gram and 2nd 2 denotes vocabulary having words that have occured more than 2 times in corpus.

**REMEMBER: You must create n and n+1 gram counts in order for the function to run. So if you want to consider 2 previous words you need to create 2-gram and 3-gram counts using ```src/change_n_gram_count.py```
file**

### Change Vocabulary

SIMILARLY,
You can create new dictionary using the file ```src/create_vocabulary.py```. Remember you must then create separate n-gram counts for same vocabulary. And replace all filepath in above ```get_suggestion()``` function.

### Running file
To run the project simply run the file ```src/main.py``` from your command line and follow the steps.

### Options
- You can enable or disable probabilities of words you get. This can be achieved when running ```src/main.py``` file.
- You can also filter words based on their starting letters. This option is also available in ```src/main.py``` file.

### Preview
#### With Probability and filter
```
$ python .\src\main.py
  Enter the sentence: I like
  Enter the letter from which you want your suggestions to starts with(Enter 0 if you want all suggestions): c
  Enter the number of suggestions you want: 4
  Do you want to get probabilities or just words? [Y] for Yes and [N] for No: Y
  The suggestions for the sentence: "I like" starting with c are:
  crazy    probability: 0.0006174109899156205
  can    probability: 0.00015435274747890512
  christmas    probability: 0.00015435274747890512
  crawling    probability: 0.00015435274747890512
```

#### Without Probability and filter
``` 
$ python .\src\main.py
  Enter the sentence: I like
  Enter the letter from which you want your suggestions to starts with(Enter 0 if you want all suggestions):
  Enter the number of suggestions you want: 4
  Do you want to get probabilities or just words? [Y] for Yes and [N] for No: N
  The suggestions for the sentence: "I like" are:
  a, the, to, i,
 ```

