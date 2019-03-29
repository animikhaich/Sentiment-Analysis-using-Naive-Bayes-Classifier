# Import the necessary library
from nltk.tokenize import word_tokenize
import re, collections, nltk, pickle
from nltk.classify import NaiveBayesClassifier
from nltk.probability import FreqDist

# Make one-hot encoding
def make_full_dict(words):
    return dict([(word, True) for word in words])

# Read the dataset
posSentences = open('Dataset/positive.txt', 'r')
negSentences = open('Dataset/negative.txt', 'r')

# Convert all the words to lower and tokenize them
posWords = word_tokenize(posSentences.read().lower())
negWords = word_tokenize(negSentences.read().lower())

# Create a corpus containing all the words
all_words = posWords + negWords

# Deterimne the frequency distribution and create a Frequency Distribution dictionary for each word
all_words_final = FreqDist(all_words)

# Get all the unique words
word_features = list(all_words_final.keys())


'''Save Pickled Data'''
# The Word features are saved since it takes a significant amount of time to create them everytime. This is not necessary
save_word_features = open('C:/Users/animi/Documents/Python Codes/NLP/PythonProgrammingDotNet/Pickle_Files/word_features_new.pickle', 'wb')
pickle.dump(word_features, save_word_features)
save_word_features.close()


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


# Read the dataset again for separately creating the positive and negative corpus 
posSentences = open('Dataset/positive.txt', 'r')
negSentences = open('Dataset/negative.txt', 'r')
posSentences = re.split(r'\n', posSentences.read())
negSentences = re.split(r'\n', negSentences.read())

posFeatures = []
negFeatures = []

# Label the positive words
for i in posSentences:
    posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
    posWords = [make_full_dict(posWords), 'pos']
    posFeatures.append(posWords)

# Laben the negative words
for i in negSentences:
    negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
    negWords = [make_full_dict(negWords), 'neg']
    negFeatures.append(negWords)

# Create the training set
trainFeatures = posFeatures + negFeatures

# Train the Classifier
classifier = NaiveBayesClassifier.train(trainFeatures)


'''Save Pickled Data'''
# Save the trained Classifier for future use
save_trained_data = open('C:/Users/animi/Documents/Python Codes/NLP/PythonProgrammingDotNet/Pickle_Files/Trained_NBC.pickle', 'wb')
pickle.dump(classifier, save_trained_data)
save_trained_data.close()


'''Load Pickled Data'''
saved_trained_data = open('C:/Users/animi/Documents/Pickle_Files/Trained_NBC.pickle', 'rb')
classifier = pickle.load(saved_trained_data)
saved_trained_data.close()

# Test the classifier with custom input
while True:
    input_text = input('Enter Text: ')
    if input_text == 'exit':
        break
    
    conv_list = word_tokenize(input_text)

    features = find_features(input_text)
    sentiment_value = classifier.classify(features)
    print(sentiment_value)