import utils, constants
import sys, pickle, argparse
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel



def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reprocessDataset', action='store_true',
                        help='Must be specified when running the program for the first time '+
                             '(when preprocessed dataset is not available). '+
                             'If specified, reads and processes the dataset again. '+
                             'Else reads an already processed dataset from ' + constants.CLASSIFICATION_DATA_PATH)
    return parser.parse_args(sys.argv[1:])


def printTopics(model):
    predicted_topics = model.print_topics(num_topics=5, num_words=5)
    for i, topics in predicted_topics:
        print('Words in Topic {}:\n {}'.format(i+1, topics))


if __name__ == '__main__':
    arguments = parseArgs()
    dataset = utils.loadDataset(arguments.reprocessDataset, classification=False, splitWords=True)

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
