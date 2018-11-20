import pandas, os, re, string, nltk, langid, itertools, pickle
import constants

LANGID_EN = 'en'
NLTK_EN = 'english'
NLTK_WORDNET = 'wordnet'
NLTK_STOPWORDS = 'stopwords'

nltk.download(NLTK_WORDNET)
nltk.download(NLTK_STOPWORDS)

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer


stopwordsSet = set(stopwords.words(NLTK_EN))
punctuations = string.punctuation
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer(NLTK_EN)


def cleanTweets(data, splitWords=False):
    def clean(text):
        lower = text.lower()
        usersRemoved = re.sub(r'(rt )?@[^ \t]*:?', '', lower)
        urlRemoved = re.sub(r'http[^ \t]*', ' ', usersRemoved)
        specialRemoved = re.sub(r'[“”–—…😩😢🙏❤😐😕😔☔😱😥💔😨【】’]', '', urlRemoved)
        numbersRemoved = re.sub(r'[^ \t]*[0-9]+[^ \t]*', '', specialRemoved)
        x = re.sub(r'&amp;', '', numbersRemoved)
        puncRemoved = x.translate(str.maketrans(punctuations, ' '*len(punctuations)))
        singleSpace = re.sub(r'[ \t]+', ' ', puncRemoved)
        stopwordsRemoved = tuple(i for i in singleSpace.strip().split() if i not in stopwordsSet and len(i)>3)
        lemmatized = tuple(lemmatizer.lemmatize(word) for word in stopwordsRemoved)
        stemmed = tuple(stemmer.stem(word) for word in lemmatized)
        results = lemmatized
        if splitWords:
            return results
        else:
            return ' '.join(results)
    if type(data) is str:
        return clean(data)
    elif type(data) is pandas.DataFrame:
        data[constants.TWEET_COLUMN] = data[constants.TWEET_COLUMN].apply(clean)
        return data


def removeUnrelatedTweets(df):
    # Removes tweets that are marked 'Not Related' or 'Not Applicable'
    df = df[df[constants.LABEL_COLUMN]!= constants.NOT_RELATED]
    df = df[df[constants.LABEL_COLUMN]!= constants.NOT_APPLICABLE]
    return df


def keepEnglishTweets(df):
    # Removes tweets that are not in english
    return df[df[constants.TWEET_COLUMN].map(lambda x: langid.classify(x)[0] == LANGID_EN)]


def getDatasetCsvPaths(path):
    # gets all the CSV files of the CrisisLexT26 Dataset
    csvPaths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if re.match(constants.TWEET_DATA_CSV_REGEX, file):
                csvPaths.append(os.path.join(root, file))
    return csvPaths


def getDisasterNameFromPath(path):
    # extracts the disaster name from the folder name in the given path
    return re.match(constants.DISASTER_NAME_REGEX, path).group(1)


def setDisasterName(df, name):
    # Sets 'Not Related' and 'Not Applicable' to 'off-topic' and  sets remaining rows to disaster name
    df.loc[df[constants.LABEL_COLUMN] == constants.NOT_RELATED, constants.LABEL_COLUMN] = constants.OFF_TOPIC_LABEL
    df.loc[df[constants.LABEL_COLUMN] == constants.NOT_APPLICABLE, constants.LABEL_COLUMN] = constants.OFF_TOPIC_LABEL
    df.loc[df[constants.LABEL_COLUMN] != constants.OFF_TOPIC_LABEL, constants.LABEL_COLUMN] = name
    return df


def createClassificationDf(dfList, splitWords):
    # Returns Dataframe that is the combination of all 26 disasters
    # Marks tweets with disaster name, drops duplicate tweets, removes non-english tweets and cleans the remaining ones
    for i in range(len(dfList)):
        disasterName = getDisasterNameFromPath(dfList[i]['path'])
        dfList[i]['df'] = setDisasterName(dfList[i]['df'],  disasterName)
    df = pandas.concat(i['df'] for i in dfList)
    df = df.drop_duplicates(constants.TWEET_COLUMN)
    df = keepEnglishTweets(df)
    df = cleanTweets(df, splitWords)
    return df


def createDocumentCorpus(dfList, splitWords):
    # Creates a list of documents, each document containing all the words in the tweets related to the corresponding disaster
    dfList = list(i['df'] for i in dfList)
    documentList = []
    for i in range(len(dfList)):
        dfList[i] = removeUnrelatedTweets(dfList[i])
        dfList[i] = dfList[i].drop_duplicates(constants.TWEET_COLUMN)
        dfList[i] = keepEnglishTweets(dfList[i])
        dfList[i] = cleanTweets(dfList[i], splitWords)
        document = tuple(itertools.chain.from_iterable(dfList[i][constants.TWEET_COLUMN]))
        documentList.append(document)
    return documentList


def readDataset(path, classification, splitWords):
    csvPaths = getDatasetCsvPaths(path)
    dfList = [{'path': path, 'df': pandas.read_csv(path)} for path in csvPaths]

    if classification:
        return createClassificationDf(dfList, splitWords)
    else:
        return createDocumentCorpus(dfList, splitWords)


def getClassName(obj):
    return obj.__class__.__name__


def saveModel(trainedModel, filePrefix=''):
    # saves the trained model to pickle file
    filename = filePrefix + getClassName(trainedModel) + '.pickle'
    createDirectoryIfNotExists(constants.MODELS_PATH)
    path = os.path.join(constants.MODELS_PATH, filename)
    print('Saving model to %s' % path)
    pickle.dump(trainedModel, open(path, 'wb'))



def loadDataset(reprocessDataset=False, classification=True, splitWords=False):
    # reads dataset, processes it and stores it for future use if reprocessDataset is True
    # else loads the preprocessed dataset and returns it
    createDirectoryIfNotExists(constants.RESOURCES_PATH)
    processedDataPath = constants.CLASSIFICATION_DATA_PATH if classification else constants.TOPIC_MODEL_DATA_PATH

    if reprocessDataset:
        print('Reading and Processing Dataset from %s' % constants.DATASET_PATH)
        dataset = readDataset(constants.DATASET_PATH, classification, splitWords)
        print('Storing Processed Dataset to %s' % processedDataPath)
        pickle.dump(dataset, open(processedDataPath, 'wb'))
    else:
        print('Reading Preprocessed Dataset from %s' % processedDataPath)
        dataset = pickle.load(open(processedDataPath, 'rb'))
    return dataset


def createDirectoryIfNotExists(path):
    # Returns True if directory did not exist and was created. Else returns False
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False
