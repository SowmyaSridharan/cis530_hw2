#############################################################
# ASSIGNMENT 2 CODE SKELETON
## RELEASED: 2/2/2020
## DUE: 2/12/2020
# DESCRIPTION: In this assignment, you will explore the
# text classification problem of identifying complex words.
# We have provided the following skeleton for your code,
# with several helper functions, and all the required
# functions you need to write.
#############################################################

from collections import defaultdict
import gzip

#### 1. Evaluation Metrics ####

# Input: y_pred, a list of length n with the predicted labels,
# y_true, a list of length n with the true labels

# DONE
# Calculates the precision of the predicted labels


def get_precision(y_pred, y_true):
    # YOUR CODE HERE...
    true_positive = 0
    false_positive = 0

    for y_p, y_a in list(zip(y_pred, y_true)):
        if y_p == 1 and y_a == 1:
            true_positive += 1
        if y_p == 0 and y_a == 1:
            false_positive += 1

    sum = false_positive + true_positive
    precision = true_positive / sum if sum else 1

    return precision

# DONE
# Calculates the recall of the predicted labels


def get_recall(y_pred, y_true):
    # YOUR CODE HERE...
    true_positive = 0
    false_negative = 0

    for y_p, y_a in list(zip(y_pred, y_true)):

        if y_p == 1 and y_a == 1:
            true_positive += 1

        if y_p == 1 and y_a == 0:
            false_negative += 1

    recall = true_positive / (true_positive + false_negative)

    return recall

# DONE
# Calculates the f-score of the predicted labels


def get_fscore(y_pred, y_true):
    prec = get_precision(y_pred, y_true)
    rec = get_recall(y_pred, y_true)
    tot = prec + rec
    fscore = 2 * (prec * rec) / (prec + rec) if tot else 0

    return fscore


# combine all metrics , return all of precision, recall, and 1 score
def prediction_metrics(y_pred, y_true):
    prec_test = get_precision(y_pred, y_true)
    rec_test = get_recall(y_pred, y_true)
    f_score_test = f_score(y_pred, y_true)
    return prec_test, rec_test, f_score_test

# print the prediction result


def test_predictions(y_prediction, y_actual):
    prec, rec, f_score = prediction_metrics(y_prediction, y_actual)

    print("Precision result is %f" % prec)
    print("Recall result is %f" % rec)
    print("F-score test result is %f" % f_score)


#### 2. Complex Word Identification ####

# Loads in the words and labels of one of the datasets


def load_file(data_file):
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels

# 2.1: A very simple baseline

# Makes feature matrix for all complex


def all_complex_feature(words):
    features = []
    for i in len(words):
        features.append(1)
    return features


def all_complex(data_file):

    # import the training and development data
    words, labels = load_file(data_file)

    predicition = all_complex_feature(words)

    # test the prediction
    test_predictions(predicition, labels)
    res_performance = prediction_metrics(predicition, labels)

    return performance


# 2.2: Word length thresholding


# ADDITIONAL
# length of each of the word and add into the list
def length_of_word_list(pandas_list):
    ret_list = []
    for k in pandas_list:
        ret_list.append(len(k))

    ret_list = np.array(ret_list)

    return ret_list

# classification based on the thresdhold


def classify_based_on_threshold(list_words, thres):
    res = []
    for word in list_words:
        if counts[word] >= thres:
            res.append(1)
        else:
            res.append(0)
    return res

# MAIN CODE
# Makes feature matrix for word_length_threshold


def length_threshold_feature(words, threshold):
    res = []
    for word in words:
        if len(word) >= threshold:
            res.append(1)
        else:
            res.append(0)
    return res

# Finds the best length threshold by f-score, and uses this threshold to
# classify the training and development set


def word_length_threshold(training_file, development_file):
   # import the training and development data
    training_words, training_labels = load_file(training_file)
    development_words, development_labels = load_file(development_file)

    # convert the allwords into list  from training_words
    lw_training = length_of_word_list(training_words)

    # FINDING THRESHOLD
    # finding the threshold of the len_word_list; FINDING THRESHOLD
    lw_median = np.median(lw_training)

    # #generate predictions
    # #classify base on the threshold

    clf_training = classify_based_on_threshold(lw_training, lw_median)
    training_performance = prediction_metrics(clf_training, training_labels)

    print("===== Training Data Result =====")
    # print the result for training
    test_predictions(clf_training, training_labels)

    # generate predicition for development
    lw_development = length_of_word_list(development_words)
    clf_development = classify_based_on_threshold(lw_development, lw_median)

    # return the metrics of the development data
    development_performance = prediction_metrics(
        clf_development, development_labels)

    # print the result
    print("===== Development Data Result =====")
    test_predictions(clf_training, training_labels)

    return training_performance, development_performance


# 2.3: Word frequency thresholding
# Loads Google NGRAM counts

def load_ngram_counts(ngram_counts_file):
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt') as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
# classify the training and development set

# Make feature matrix for word_frequency_threshold


def frequency_threshold_feature(words, threshold, counts):


def word_frequency_threshold(training_file, development_file, counts):
    # YOUR CODE HERE
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

# 2.4: Naive Bayes

# generate all feature


def generate_feature(x_dataset, counts):
    # generate length of word list
    lwl = length_of_word_list(x_dataset)

    # generate frequency list
    fwf = frequency_list(x_dataset, counts)

    #hstack or concat
    feat_combine = np.hstack([lwl, fwf])
    return feat_combine


# normalization
def normalize(features):
    mean = np.mean(features)
    std_dev = np.std(features)
    delta = features - mean
    X_scaled = delta / std_dev
    return X_scaled


# shows the result of the predicition based on the recall, f_score, and precision.

def test_predictions(y_prediction, y_actual):
    prec, rec, f_score = prediction_metrics(y_prediction, y_actual)

    print("Precision result is %f" % prec)
    print("Recall result is %f" % rec)
    print("F-score test result is %f" % f_score)


def length_of_word_list(word_list):
    ret_list = []
    for x in word_list:
        l_w = len(x)
        l_f = [float(l_w)]
        ret_list.append(l_f)

    return np.array(ret_list)


def frequency_list(word_list, counts):
    ret_list = []
    for x in word_list:
        f_w = counts[x]
        f_f = [float(f_w)]
        ret_list.append(f_f)
    return np.array(ret_list)

# Trains a Naive Bayes classifier using length and frequency features


def naive_bayes(training_file, development_file, counts):

    # import the training and development data
    training_words, training_labels = load_file(training_file)
    development_words, development_labels = load_file(development_file)

    # generate feature
    training_features = generate_feature(training_words, counts)
    development_features = generate_feature(development_words, counts)

    # normalize
    training_features = normalize(training_features)
    development_features = normalize(development_features)

    # train the model
    y_test_reshape = np.reshape(training_labels, (-1, 1))
    clf = GaussianNB()
    clf.fit(training_features, y_test_reshape)

    # generate predictions

    training_predictions = clf.predict(y_test_reshape)
    development_predictions = clf.predict(development_features)

    # return the metrics of the training data
    training_performance = prediction_metrics(
        training_predictions, y_test_reshape)

    # return the metrics of the development data
    development_performance = prediction_metrics(
        development_predictions, development_labels)

    # print the result
    print("===== Training Data Result =====")
    test_predictions(training_predictions, training_labels)


# 2.5: Logistic Regression

# Trains a Naive Bayes classifier using length and frequency features


def logistic_regression(training_file, development_file, counts):
    # YOUR CODE HERE
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance

# 2.7: Build your own classifier

# Trains a classifier of your choosing, predicts labels for the test dataset
# and writes the predicted labels to the text file 'test_labels.txt',
# with ONE LABEL PER LINE


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    train_data = load_file(training_file)

    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

    naive_bayes(training_file, development_file, counts)
