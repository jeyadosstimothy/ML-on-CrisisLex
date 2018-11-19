import os

# Constants related to Dataset
DATASET_PATH = 'CrisisLexT26'
DISASTER_NAME_REGEX = '^.*/.*?_(.*)/.*$'
TWEET_DATA_CSV_REGEX = '.*_labeled.csv'

# Constants related to Dataset Features
TWEET_COLUMN = ' Tweet Text'
LABEL_COLUMN = ' Informativeness'
NOT_RELATED = 'Not related'
NOT_APPLICABLE = 'Not applicable'
OFF_TOPIC_LABEL = 'off-topic'

# Paths for storing processed dataset
RESOURCES_PATH = 'resources'
CLASSIFICATION_DATA_PATH = os.path.join(RESOURCES_PATH,'classificationData.pickle')
TOPIC_MODEL_DATA_PATH = os.path.join(RESOURCES_PATH,'topicModelData.pickle')

# Paths for storing trained models
MODELS_PATH = 'models'
EMBEDDING_NN_MODEL_PATH = os.path.join(MODELS_PATH,'mlpNNModel.h5')
LSTM_NN_MODEL_PATH = os.path.join(MODELS_PATH,'lstmNNModel.h5')
GRU_NN_MODEL_PATH = os.path.join(MODELS_PATH,'gruNNModel.h5')
