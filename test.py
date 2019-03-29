# Import the necessary library
from nltk.tokenize import word_tokenize
import re, collections, nltk, pickle
from nltk.classify import NaiveBayesClassifier
from nltk.probability import FreqDist

'''Load Pickled Data'''
# Load the saved word corpus
The word features are loaded
saved_word_features = open('C:/Users/animi/Documents/Python Codes/NLP/PythonProgrammingDotNet/Pickle_Files/word_features_new.pickle', 'rb')
word_features = pickle.load(saved_word_features)
saved_word_features.close()

'''Load Pickled Data'''
# Load the saved Classifier
saved_trained_data = open('C:/Users/animi/Documents/Pickle_Files/Trained_NBC.pickle', 'rb')
classifier = pickle.load(saved_trained_data)
saved_trained_data.close()

def find_features(document):
    """
    Create a dict of features by using Boolean One-Hot Encoding
    Also called, word vectorization
    """
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# Test the classifier with custom input
while True:
    input_text = input('Enter Text: ')
    if input_text == 'exit':
        break
    
    conv_list = word_tokenize(input_text)

    features = find_features(input_text)
    sentiment_value = classifier.classify(features)
    print(sentiment_value)