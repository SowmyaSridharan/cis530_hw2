#############################################################
## ASSIGNMENT 2 CODE SKELETON
## RELEASED: 1/17/2018
## DUE: 1/24/2018
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################
import gzip
import io
from collections import defaultdict
import zipfile as zip
import matplotlib.pyplot as plt
import numpy as np
import sklearn
letter_sele = 'aeiou-'

#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    ## YOUR CODE HERE...
    tp = 0
    fp = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        if y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
    if tp == 0 and fp == 0:
        return 1
    precision = 1.0 * tp / (1.0 * (tp + fp))
    return precision


## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    ## YOUR CODE HERE...
    tp = 0
    fn = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        if y_pred[i] == 0 and y_true[i] == 1:
            fn += 1
    if tp == 0 and fn == 0:
        return 1
    recall = 1.0 * tp / (1.0 * (tp + fn))
    return recall


## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    ## YOUR CODE HERE...
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    if precision == 0 and recall == 0:
        return 0
    fscore = 2.0 * precision * recall / (precision + recall)
    return fscore


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

#for leaderboard output
def load_file2(data_file):
    with open(data_file, 'rt', encoding="utf8") as f:
        lines = f.readlines()
        lines = lines[1:]
        num_data = len(lines)
        Lst_pos1 = [line[:-1].find('\t') for line in lines]
        words = [lines[i][:Lst_pos1[i]] for i in range(num_data)]
        #Lst_pos2 = [lines[i][(Lst_pos1[i]+1):-1].find('\t')+Lst_pos1[i]+1 for i in range(num_data )]
        return words


### 2.1: A very simple baseline
def prediction_metrics(y_pred, y_true):
    prec_test = get_precision(y_pred, y_true)
    rec_test = get_recall(y_pred, y_true)
    f_score_test = get_fscore(y_pred, y_true)
    return prec_test, rec_test, f_score_test

def test_predictions(y_prediction, y_actual):
    prec, rec, f_score = prediction_metrics(y_prediction, y_actual)

    print("Precision result is %f" % prec)
    print("Recall result is %f" % rec)
    print("F-score test result is %f" % f_score)

def all_complex_feature(words):
    features = []
    for i in range(len(words)):
        features.append(1)
    return features

def all_complex(data_file):

    # import the training and development data
    words, labels = load_file(data_file)

    predicition = all_complex_feature(words)

    # test the prediction
    test_predictions(predicition, labels)
    res_performance = prediction_metrics(predicition, labels)

    return res_performance

## Labels every word complex



### 2.2: Word length thresholding

def length_threshold_feature(words,threshold):
    res = []
    for word in words:
        if len(word) >= threshold:
            res.append(1)
        else:
            res.append(0)
    return res

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    ## YOUR CODE HERE
    words, y_true = load_file(training_file)
    maxn = 0
    precisions = []
    recalls = []
    for i in range(len(words)):
        if maxn < len(words[i]):
            maxn = len(words[i])
    #print(maxn)
    index = -1
    max_fscore = -1
    for i in range(maxn):
        y_pred = []
        for j in range(len(words)):
            if len(words[j]) < i:
                y_pred.append(0)
            else:
                y_pred.append(1)
        precision = get_precision(y_pred, y_true)
        recall = get_recall(y_pred, y_true)
        fscore = get_fscore(y_pred, y_true)
        if max_fscore < fscore:
            max_fscore = fscore
            index = i
        precisions.append(precision)
        recalls.append(recall)

	# print (precisions)
    print(index)
    plt.step(recalls, precisions, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('2 class precision-recall curve')
    #plt.show()

    y_pred = []
    for i in range(len(words)):
        if len(words[i]) < index:
            y_pred.append(0)
        else:
            y_pred.append(1)

    tprecision = get_precision(y_pred, y_true)
    trecall = get_recall(y_pred, y_true)
    tfscore = get_fscore(y_pred, y_true)


    words, y_true = load_file(development_file)
    y_pred = length_threshold_feature()
    dprecision = get_precision(y_pred, y_true)
    drecall = get_recall(y_pred, y_true)
    dfscore = get_fscore(y_pred, y_true)

    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance


### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file):
    counts = {}
    with gzip.open(ngram_counts_file, 'rt',errors='ignore') as f:
       for line in f:
           token, count = line.strip().split('\t')
           if token[0].islower():
               counts[token] = int(count)
    return counts

## Make feature matrix for word_frequency_threshold
def frequency_threshold_feature(words, threshold, counts):

    y_pred = []
    for i in range(len(words)):
        if counts[words[i]] < threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

    return y_pred

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set
def word_frequency_threshold(training_file, development_file, counts):
    ## YOUR CODE HERE
    words, y_true = load_file(training_file)
    maxn = 0
    precisions = []
    recalls = []
    freq = []

    for i in range(len(words)):
        freq.append(counts[words[i]])

    threshold = -1
    max_fscore = -1

    for i in range(0, int(1e+10), int(1e+6)):
        # print ("h")
        #print(i)
        y_pred = []

        for j in range(len(words)):
            if counts[words[j]] < i:
                y_pred.append(1)
            else:
                y_pred.append(0)

        precision = get_precision(y_pred, y_true)
        recall = get_recall(y_pred, y_true)
        fscore = get_fscore(y_pred, y_true)

        if max_fscore < fscore:
            max_fscore = fscore
            threshold = i

        precisions.append(precision)
        recalls.append(recall)

    print(threshold)
    plt.plot(recalls, precisions, color='red')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('2 class precision-recall curve')
    #plt.show()

    y_pred = []
    for i in range(len(words)):
        if counts[words[i]] < threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

    tprecision = get_precision(y_pred, y_true)
    trecall = get_recall(y_pred, y_true)
    tfscore = get_fscore(y_pred, y_true)

    y_pred = frequency_threshold_feature(words,threshold,counts)
    dprecision = get_precision(y_pred, y_true)
    drecall = get_recall(y_pred, y_true)
    dfscore = get_fscore(y_pred, y_true)

    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance


from sklearn.naive_bayes import GaussianNB
## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    words,labels = load_file(training_file)
    labels_np = np.array(labels)
    X_features = np.array([[1.0*len(w), counts[w]] if w in counts else [1.0*len(w),0] for w in words])
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    clf = GaussianNB(); clf.fit(X_features, labels_np)
    Y_pred_np = clf.predict(X_features)

    tprecision = get_precision(Y_pred_np, labels_np)
    trecall = get_recall(Y_pred_np, labels_np)
    tfscore = get_fscore(Y_pred_np, labels_np)

    #training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words, labels = load_file(development_file)
    labels_np = np.array(labels)
    X_features = np.array([[1.0*len(w), counts[w]] if w in counts else [1.0*len(w),0] for w in words])
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    Y_pred_np = clf.predict(X_features)

    dprecision = get_precision(Y_pred_np, labels_np)
    drecall = get_recall(Y_pred_np, labels_np)
    dfscore = get_fscore(Y_pred_np, labels_np)

    #development_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return  development_performance

from sklearn.linear_model import LogisticRegression
def logistic_regression(training_file, development_file, counts):
    ## YOUR CODE HERE

    words,labels = load_file(training_file)
    labels_np = np.array(labels)
    X_features = np.array([[1.0*len(w), counts[w]] if w in counts else [1.0*len(w),0] for w in words])
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    clf = LogisticRegression(); #penalty = 'l1'
    clf.fit(X_features, labels_np)
    Y_pred_np = clf.predict(X_features)

    tprecision = get_precision(Y_pred_np, labels_np)
    trecall = get_recall(Y_pred_np, labels_np)
    tfscore = get_fscore(Y_pred_np, labels_np)
    #training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    words, labels = load_file(development_file)
    labels_np = np.array(labels)
    X_features = np.array([[1.0*len(w), counts[w]] if w in counts else [1.0*len(w),0] for w in words])
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    Y_pred_np = clf.predict(X_features)
    # print("Y_pred_np",Y_pred_np)

    dprecision = get_precision(Y_pred_np, labels_np)
    drecall = get_recall(Y_pred_np, labels_np)
    dfscore = get_fscore(Y_pred_np, labels_np)

    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return  development_performance


### 2.7: Build your own classifier

## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE
# def own_classifier(training_file,development_file,test_file1,counts):
from syllables import count_syllables

def preprocessor(words,labels, counts):
    Thres_opt_len = 7
    Thres_opt_freq = 19904037
    #19904037#<-19903996#<- 19903896#<-19903906# <-19902396 #<- 19881406 #<- 19802396
    # 1.0*len(w),
#     X_features = [[1.0*len(w),count_syllables(w),[0,1][len(w) > Thres_opt_len], int(counts[w] < Thres_opt_freq), counts[w] ]+[w.count(alp) for alp in letter_sele] if w in counts else [1.0*len(w),count_syllables(w),[0,1][len(w) > Thres_opt_len],1,1120679362]+[w.count(alp) for alp in letter_sele] for w in words]
    # best
    #retain this
    #X_features = np.array([[1.0*len(w),count_syllables(w),[0,1][len(w) > Thres_opt_len], int(counts[w] < Thres_opt_freq), counts[w] ]+[w.count(alp) for alp in letter_sele] if w in counts else [1.0*len(w),count_syllables(w),[0,1][len(w) > Thres_opt_len],1,1120679362]+[w.count(alp) for alp in letter_sele] for w in words])

#reduction for submission
    X_features = np.array([[1.0*len(w),count_syllables(w),[0,1][len(w) > Thres_opt_len], int(counts[w] < Thres_opt_freq), counts[w] ] if w in counts else [1.0*len(w),count_syllables(w),[0,1][len(w) > Thres_opt_len],1,1120679362] for w in words])
#further reduction
    #X_features = np.array([[1.0*len(w),count_syllables(w), counts[w] ] if w in counts else [1.0*len(w),count_syllables(w),1120679362] for w in words])
    print(X_features[1])
#     X_features = np.array([[1.0*len(w),count_syllables(w),[0,1][len(w) > Thres_opt_len], int(counts[w] < Thres_opt_freq) ]+[w.count(alp) for alp in letter_sele] if w in counts else [1.0*len(w),count_syllables(w),[0,1][len(w) > Thres_opt_len],1]+[w.count(alp) for alp in letter_sele] for w in words])
    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    #print(X_features[1])
#     X_features = np.array([np.concatenate((row,np.convolve(row,row))) for row in X_features])
#     scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
    return X_features, np.array(labels)



from sklearn.tree import DecisionTreeClassifier
def decision_tree(training_file, development_file, counts,show_err_word_flags=False):
    words,labels = load_file(training_file)
    #print('X_features train')
    X_features, labels_np = preprocessor(words, labels, counts)
    clf = DecisionTreeClassifier(max_depth=2, random_state=0); clf.fit(X_features, labels_np) # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    Y_pred_np = clf.predict(X_features)
    if show_err_word_flags: L_train = [X_features,words,list(Y_pred_np),labels]
    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]


    words, labels = load_file(development_file)
    #print('X_features dev')
    X_features, labels_np = preprocessor(words, labels, counts)
    Y_pred_np = clf.predict(X_features)

    if show_err_word_flags: L_dev = [X_features,words,list(Y_pred_np),labels]
    development_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
    #return training_performance, development_performance
    if not show_err_word_flags: return training_performance, development_performance
    else: return clf,L_train, L_dev


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    train_data = load_file(training_file)

    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

#print(all_complex(training_file))
#print(all_complex(development_file))
#print('Word Length Threshold')
#print(word_length_threshold(training_file, development_file))
#print('Word Frequency Threshold')
#print(word_frequency_threshold(training_file,development_file,counts))
#print('Logistic Regression')
#print(logistic_regression(training_file,development_file,counts))
#print("Decision Trees")
#print(decision_tree(training_file,development_file,counts,show_err_word_flags=False))

#print(decision_tree(training_file,development_file,counts,show_err_word_flags=False))

#clf,L_train, L_dev = decision_tree(training_file,development_file,counts, show_err_word_flags=True)
#X,w, l_pred, l_true = L_train
#print("Correct Prediction:")
#num_prt =8
#print("Examples of true positive", [w[i] for i in range(len(l_true)) if 1 == l_pred[i] and 1 == l_true[i]][:num_prt])
#print("Examples of false negative", [w[i] for i in range(len(l_true)) if 0 == l_pred[i] and 0 == l_true[i]][:num_prt])
#print("Incorrect prediction:")
#print("Examples of false positive (i.e. not complex, but are predicted to be)", [w[i] for i in range(len(l_true)) if 1 == l_pred[i] and 0 ==  l_true[i]][:num_prt])
#print("Examples of true negative (i.e. complex, but are predicted not to be)", [w[i] for i in range(len(l_true)) if 0 == l_pred[i] and 1 == l_true[i]][:num_prt])
#print("-------------------------------------")



# leaderboard output

unlabeled_file = "data/complex_words_test_unlabeled.txt"

#def decision_tree_unlabeled_output(training_lst, unlabeled_file, counts):
#    words,labels = load_file(training_lst)
#    labels_np = np.array(labels)
#    X_features, labels_np = preprocessor(words, labels, counts)
#     print([len(fe) for fe in X_features])
#     print([row for row in X_features if None in row])
#     print([i for i in list(labels_np) if None == i])
#    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
#    clf = DecisionTreeClassifier(max_depth=2, random_state=0); clf.fit(X_features, labels_np)
#    Y_pred_np = clf.predict(X_features)
#    training_performance = [get_precision(labels_np, Y_pred_np), get_recall(labels_np, Y_pred_np), get_fscore(labels_np, Y_pred_np)]
#    words = load_file2(unlabeled_file)
#    X_features, labels_np = preprocessor(words, labels, counts) #labels are not useful here
#    scaler = sklearn.preprocessing.StandardScaler(); scaler.fit(X_features); X_features = scaler.transform(X_features)
#    Y_pred_np = clf.predict(X_features)
#    fname = "test_labels.txt"
#    with open(fname, 'wt', encoding="utf8") as f:
#        for l in Y_pred_np: f.write(str(l)+'\n')
#    return training_performance

#print(logistic_regression(training_file,development_file,counts))
#print(decision_tree_unlabeled_output(training_file,unlabeled_file,counts))
