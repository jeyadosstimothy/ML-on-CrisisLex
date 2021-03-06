# Machine Learning Techniques on CrisisLexT26 dataset
This repository consists of programs which apply different Machine Learning Techniques to perform Text Classification and Topic Modelling on the CrisisLexT26 dataset. The dataset consists of tweets related to 26 disasters, which are labelled based on their informativeness. Text Classification techniques are used to predict which disaster each tweet is talking about. Topic Modelling is used to find the words that belong to the top 5 topics from the collection of tweets.

This document lists the prerequisites, execution steps and outlines the steps involved in processing the dataset and the algorithms used.

## Prerequisites
The following packages were used with Python 3.5:
* gensim
* langid
* tensorflow
* keras
* pandas
* scikit-learn
* xgboost

## Execution
The programs can be executed as:
```sh
$ python3 topicModelling.py
$ python3 scikitClassifiers.py
$ python3 neuralNetworks.py
```
When the `-r` or `--reprocessDataset` option is specified, the code reads, processes and saves the processed dataset to `resources/`. Else the code will search for the preprocessed dataset and use it. This is to avoid the time spent on processing the dataset, instead of training the models, on subsequent executions. Hence, `-r` must be specified when executing the program for the first time.

For Neural Networks, the type of network (`lstm`, `gru`, `mlp`) can be specified as argument. If not specified, `mlp` is used by default.

The `-p` or `--printMetrics` option can be used to print the Classification Reports and Confusion Matrices for the models.

The above information can also be obtained by specifying the `-h` or `--help` option.

The trained models are automatically saved to `models/`

## CrisisLexT26 dataset
The CrisisLexT26 dataset consists of the following features: Tweet ID, Tweet Text, Information Source, Information Type, Informativeness. The Tweet Text and the Informativeness are the ones we consider for applying ML. The dataset consists of tweets which can be of different languages, can be retweets and can have URL links, usernames and special characters(emojis). The tweets that are marked *Not related* or *Not applicable* do not talk about disasters and are hence labelled as *Off-Topic*. The tweets that are marked as *Related and Informative* and *Related but not Informative* are both labelled with the disaster's name.

## Preparation of Dataset
We need the dataset to be in an appropriate form for training the Machine Learning models. The dataset is cleaned by removing tweets that are not in English, removing stop words and punctuation, removing URLs and usernames, etc., *Stemming* and *Lemmatization* are also performed. *langid* is used for identifying the language of the tweets and the tweets that are not in English are removed. For Text Classification, the tweets are split into unigrams and bigrams (n-grams with n=1, 2) and then converted into *Count Vectors* and *TF-IDF Vectors*. For Topic Modelling, the tweets related to each disaster are converted into *Bag of Words*.

## Topic Modelling
For Topic Modelling, we use *Latent Dirichlet Allocation (LDA)* and *Latent Semantic Analysis (LSA)*, which are provided by *gensim*. The models are trained on the Bag of Words and the words in the Top 5 topics are found. The implementation can be found in *topicModelling.py*

## Text Classification
The following models are used for classifying the tweets: *Logistic Regression*, *Naive Bayes*, *Random Forests*, *Gradient Boosting* and *Neural Networks*. Neural Networks are implemented using *keras*, Gradient Boosting using *XGboost* and the remaining ones using *scikit-learn*. The Scikit Classifiers are trained on the Count Vectors and TF-IDF Vectors independently in order to compare their performance. Logistic Regression with count vectors seems to give the highest accuracy.
In case of Neural Networks, the tweets are converted into *Word Embeddings* and then fed to *Long Short-Term Memory (LSTM)*, *Gated Recurrent Unit (GRU)* or *Multi-Layer Perceptron (MLP)* Models, whichever is specified.

## Performance Comparison of Classifiers
|Model                              |Validation Accuracy(%)|
|-----------------------------------|:--------------------:|
|Logistic Regression, Count Vectors |                 93.18|
|Logistic Regression, TF-IDF Vectors|                 89.26|
|Naive Bayes, Count Vectors         |                 90.26|
|Naive Bayes, TF-IDF Vectors        |                 77.97|
|Random Forests, Count Vectors      |                 91.81|
|Random Forests, TF-IDF Vectors     |                 91.75|
|Gradient Boosting, Count Vectors   |                 92.29|
|Gradient Boosting, TF-IDF Vectors  |                 91.96|
|LSTM, Word Embeddings              |                 91.49|
|GRU, Word Embeddings               |                 91.86|
|MLP, Word Embeddings               |                 90.84|

## References
* A. Olteanu, C. Castillo, F. Diaz, S. Vieweg. (2014). CrisisLex: A Lexicon for Collecting and Filtering Microblogged Communications in Crises. In Proceedings of the AAAI Conference on Weblogs and Social Media (ICWSM'14). AAAI Press, Ann Arbor, MI, USA.
