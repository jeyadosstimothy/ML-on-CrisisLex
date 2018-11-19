from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import ensemble
from sklearn.metrics import classification_report, confusion_matrix
import xgboost
import os, re, pickle, sys, argparse
import utils, constants


def parseArgs():
    # Parses commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reprocessDataset', action='store_true',
                        help='If specified, reads and processes the dataset again.'+
                             'Else reads an already processed dataset from ' + constants.CLASSIFICATION_DATA_PATH)
    return parser.parse_args(sys.argv[1:])


def printMetrics(classifier, xValid, yValid):
    # print Classification report and Confusion matrix based on predictions on validation data
    predictions = classifier.predict(xValid)

    print('Classification Report:')
    print(classification_report(yValid, predictions))

    print('Confusion Matrix:')
    for i in confusion_matrix(yValid, predictions):
        print('[', ', '.join(map(str, i)), ']')


def trainModel(classifier, xTrain, yTrain, xValid, yValid):
    # fit the training dataset on the classifier
    classifier.fit(xTrain, yTrain)

    # predict the labels on validation dataset
    predictions = classifier.predict(xValid)

    # return the trained model and the accuracy
    return classifier, metrics.accuracy_score(predictions, yValid)


def loadDataset(reprocessDataset):
    # reads dataset, processes it and stores it for future use if reprocessDataset is True
    # else loads the preprocessed dataset and returns it
    if reprocessDataset:
        print('Reading and Processing Dataset from %s' % constants.DATASET_PATH)
        dataset = utils.readDataset(constants.DATASET_PATH, splitWords=False)
        print('Storing Processed Dataset to %s' % constants.CLASSIFICATION_DATA_PATH)
        pickle.dump(dataset, open(constants.CLASSIFICATION_DATA_PATH, 'wb'))
    else:
        print('Reading Preprocessed Dataset from %s' % constants.CLASSIFICATION_DATA_PATH)
        dataset = pickle.load(open(constants.CLASSIFICATION_DATA_PATH, 'rb'))
    return dataset


if __name__ == '__main__':
    arguments = parseArgs()
    dataset = loadDataset(arguments.reprocessDataset)
    xData, yData = dataset[constants.TWEET_COLUMN], dataset[constants.LABEL_COLUMN]

    dataEncodersList = [(CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2)), preprocessing.LabelEncoder()),
                        (TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2)), preprocessing.LabelEncoder())
                        ]

    modelsList = [  naive_bayes.MultinomialNB(),
                    linear_model.LogisticRegression(solver='saga', multi_class='auto'),
                    ensemble.RandomForestClassifier(n_estimators=25),
                    xgboost.XGBClassifier()
                    ]

    for xEncoder, yEncoder in dataEncodersList:
        print('Using {} and {} for encoding xData and yData'.format(utils.getClassName(xEncoder), utils.getClassName(yEncoder)))

        # fit the encoders on the dataset
        xEncoder.fit(xData)
        yEncoder.fit(yData)

        print('Encoding and splitting xData, yData')
        xDataEncoded, yDataEncoded = xEncoder.transform(xData), yEncoder.transform(yData)
        xTrain, xValid, yTrain, yValid = model_selection.train_test_split(xDataEncoded, yDataEncoded)

        for model in modelsList:
            print('Training model:', utils.getClassName(model))
            trainedModel, accuracy = trainModel(model, xTrain, yTrain, xValid, yValid)
            print('Accuracy:', accuracy)

            # Uncomment to print Classification Reports and Confusion Matrices:
            # printMetrics(trainedModel, xValid, yValid)

            filePrefix = utils.getClassName(xEncoder) + '_'
            utils.saveModel(trainedModel, filePrefix=filePrefix)
