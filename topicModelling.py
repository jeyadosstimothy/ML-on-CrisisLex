import utils, constants
import sys, pickle, argparse
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel



def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reprocessDataset', action='store_true',
                        help='If specified, reads and processes the dataset again.'+
                             'Else reads an already processed dataset from ' + constants.CLASSIFICATION_DATA_PATH)
    return parser.parse_args(sys.argv[1:])


def printTopics(model):
    predicted_topics = model.print_topics(num_topics=5, num_words=5)
    for i, topics in predicted_topics:
        print('Words in Topic {}:\n {}'.format(i+1, topics))


def loadDataset(reprocessDataset):
    if reprocessDataset:
        print('Reading and Processing Dataset from %s' % constants.DATASET_PATH)
        dataset = utils.readDataset(constants.DATASET_PATH, classification=False)
        print('Storing Processed Dataset to %s' % constants.TOPIC_MODEL_DATA_PATH)
        pickle.dump(dataset, open(constants.TOPIC_MODEL_DATA_PATH, 'wb'))
    else:
        print('Reading Preprocessed Dataset from %s' % constants.TOPIC_MODEL_DATA_PATH)
        dataset = pickle.load(open(constants.TOPIC_MODEL_DATA_PATH, 'rb'))
    return dataset


if __name__ == '__main__':
    arguments = parseArgs()
    dataset = loadDataset(arguments.reprocessDataset)

    # Creating dictionary from dataset, where each unique term is assigned an index
    dictionary = corpora.Dictionary(dataset)

    # Converting list of documents into Bag of Words using dictionary
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in dataset]

    # Training models on the document term matrix
    modelList = [   LdaModel(doc_term_matrix, num_topics=10, id2word=dictionary, passes=2),
                    LsiModel(doc_term_matrix, num_topics=10, id2word=dictionary)
                ]

    for model in modelList:
        print('Topic Modelling using %s' % utils.getClassName(model))
        printTopics(model)
        utils.saveModel(model)
