from sklearn import model_selection, preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, GRU, Flatten
from keras.layers.embeddings import Embedding
import pickle, sys, argparse
from sklearn.metrics import classification_report, confusion_matrix

import utils, constants

# Constants for determining type of neural network
LSTM_LABEL = 'lstm'
GRU_LABEL = 'gru'
MLP_LABEL = 'mlp'


def parseArgs():
    # Parses commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reprocessDataset', action='store_true',
                        help='Must be specified when running the program for the first time '+
                             '(when preprocessed dataset is not available). '+
                             'If specified, reads and processes the dataset again. '+
                             'Else reads an already processed dataset from ' + constants.CLASSIFICATION_DATA_PATH)
    parser.add_argument('-p', '--printMetrics', action='store_true',
                        help='If specified, prints the Classification Reports and Confusion Matrices')
    parser.add_argument('networkType', nargs='?', default=MLP_LABEL, choices=[LSTM_LABEL, GRU_LABEL, MLP_LABEL])
    return parser.parse_args(sys.argv[1:])


def printMetrics(classifier, xValid, yValid):
    # print Classification report and Confusion matrix based on predictions on validation data
    predictions = classifier.predict(xValid)
    predictions = predictions.argmax(axis=-1)
    yValid = yValid.argmax(axis=-1)

    print('Classification Report:')
    print(classification_report(yValid, predictions))

    print('Confusion Matrix:')
    for i in confusion_matrix(yValid, predictions):
        print('[', ', '.join(map(str, i)), ']')


def encodeX(xEncoder, xData):
    xEncoder.fit_on_texts(xData)
    sequences = xEncoder.texts_to_sequences(xData)
    xData = pad_sequences(sequences, maxlen=150)
    return xData


def encodeY(yEncoder, yData):
    yData = yEncoder.fit_transform(yData.values.reshape(-1, 1))
    return yData


def saveModel(model, networkType):
    # Saves the model to h5 file
    path = ''
    if networkType == LSTM_LABEL:
        path = constants.LSTM_NN_MODEL_PATH
    elif networkType == GRU_LABEL:
        path = constants.GRU_NN_MODEL_PATH
    elif networkType == MLP_LABEL:
        path = constants.EMBEDDING_NN_MODEL_PATH
    print('Saving model to %s' % path)
    utils.createDirectoryIfNotExists(constants.MODELS_PATH)
    model.save(path)


def buildModel(vocabularySize, networkType):
    # Adds Layers and compiles the built neural network
    model = Sequential()
    model.add(Embedding(vocabularySize, 200, input_length=150))
    model.add(Dropout(0.2))
    if networkType == LSTM_LABEL:
        model.add(LSTM(150, dropout=0.2, recurrent_dropout=0.2))
    elif networkType == GRU_LABEL:
        model.add(GRU(150, dropout=0.2, recurrent_dropout=0.2))
    elif networkType == MLP_LABEL:
        model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(27, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    arguments = parseArgs()
    dataset = utils.loadDataset(arguments.reprocessDataset)
    xData, yData = dataset[constants.TWEET_COLUMN], dataset[constants.LABEL_COLUMN]

    vocabularySize = 13000
    xEncoder, yEncoder = Tokenizer(num_words=vocabularySize), preprocessing.OneHotEncoder()

    print('Encoding and splitting xData, yData')
    xDataEncoded, yDataEncoded = encodeX(xEncoder, xData), encodeY(yEncoder, yData)
    xTrain, xValid, yTrain, yValid = model_selection.train_test_split(xDataEncoded, yDataEncoded)

    model = buildModel(vocabularySize, arguments.networkType)
    print(model.summary())

    print('Commencing training of neural network')
    model.fit(xTrain, yTrain, validation_data=(xValid, yValid), epochs=4, batch_size=32)

    if arguments.printMetrics:
        printMetrics(model, xValid, yValid)

    saveModel(model, arguments.networkType)
