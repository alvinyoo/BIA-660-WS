"""
Final project: classification
members: Yuchen Yao, Jiahao Li, Nuozhou Tang, Qingxi Liu, Shiying Li
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from nltk.corpus import stopwords
from sklearn import svm
from time import time
import nltk
import csv
import re


def loadData(files: list):
    """
    read the reviews and their polarities from a given file
    :param files: the path of review file
    :return: train reviews, train labels, test reviews and test labels
    """
    reviews, labels = [], []
    for file in files:
        f = open(file)
        reader = csv.reader(f)
        for line in reader:
            review, tittle = line[0].strip().replace('\n', ' ').replace('\t', ''), line[1]
            reviews.append(review.lower())
            labels.append(int(float(tittle)))
        f.close()
    return reviews, labels


def Filter(reviews):
    """
    decrease the dimension of dataset
    :param reviews: reviews from dataset
    :return: reviews without stop words
    """
    ans = []
    for review in reviews:
        temp = []
        review = re.sub(r'[^\w\s]', ' ', review)
        review = re.sub('[^a-z]', ' ', review)
        review = re.sub('data sci[a-z]+', ' ', review, re.I)
        review = re.sub('data eng[a-z]+', ' ', review, re.I)
        review = re.sub('software eng[a-z]+', ' ', review, re.I)
        review = re.sub('\[.*?\]', '', review)
        review = re.sub('https?://\S+|www\.\S+', '', review)
        review = re.sub('<.*?>+', '', review)
        review = re.sub('\n', '. ', review)
        review = re.sub('\w*\d\w*', '', review)
        review = re.sub(r'@[A-Za-z0-9]+', '', review)
        review = re.sub(r'#', '', review)
        review = re.sub(r'RT[\s]+', '', review)
        review = re.sub(r'[^\w]', ' ', review)

        ps = nltk.stem.porter.PorterStemmer()

        new_review = []
        for word in review.split():
            word = ps.stem(word)
            if word == '':
                continue  # ignore empty words and stopwords
            else:
                new_review.append(word)
        temp.append(' '.join(new_review))
        ans += temp
    return ans


def vt(predictors, counts_val, counts_train, lab_train):
    """
    Voting Classifier with different classification algorithms
    :param predictors: different classification algorithms
    :param counts_val: the transformed testing data
    :param counts_train: the transformed training data
    :param lab_train: the training labels
    :return: a array of predicted label
    """
    VT = VotingClassifier(predictors, voting='hard')
    VT.fit(counts_train, lab_train)
    predicted = VT.predict(counts_val)
    return predicted


def lgr_classifier(counts_train, lab_train):
    """
    Logistic regression classifier
    :param counts_train: the transformed training data
    :param lab_train: the training labels
    :return: An object for grid search
    """
    clf = LogisticRegression(solver='liblinear')
    LGR_grid = [{'penalty': ['l1', 'l2'], 'C': [0.5, 1, 1.5, 2, 3, 5, 10]}]
    gridsearchLGR = GridSearchCV(clf, LGR_grid, cv=5)
    return gridsearchLGR.fit(counts_train, lab_train)


def rf_classifier(counts_train, lab_train):
    """
    Random forest classifier
    :param counts_train: the transformed training data
    :param lab_train: the training labels
    :return: An object for grid search
    """
    clf = RandomForestClassifier(random_state=150, max_depth=600, min_samples_split=160)
    RF_grid = [{'n_estimators': [50, 100, 150, 200, 300, 500, 800, 1200, 1600, 2100],
                'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2']}]
    gridsearchRF = GridSearchCV(clf, RF_grid, cv=5)
    return gridsearchRF.fit(counts_train, lab_train)


def knn_classifier(counts_train, lab_train):
    """
    K-nearnest-neighbor classifier
    :param counts_train: the transformed training data
    :param lab_train: the training labels
    :return: An object for grid search
    """
    clf = KNeighborsClassifier()
    KNN_grid = [{'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17],
                 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}]
    gridsearchKNN = GridSearchCV(clf, KNN_grid, cv=5)
    return gridsearchKNN.fit(counts_train, lab_train)


def dt_classifier(counts_train, lab_train):
    """
    Decision tree classifier
    :param counts_train: the transformed training data
    :param lab_train: the training labels
    :return: An object for grid search
    """
    clf = DecisionTreeClassifier()
    DT_grid = [{'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random']}]
    gridsearchDT = GridSearchCV(clf, DT_grid, cv=5)
    return gridsearchDT.fit(counts_train, lab_train)


def nb_classifier(counts_train, lab_train):
    """
    Naive Bayes classifier
    :param counts_train: the transformed training data
    :param lab_train: the training labels
    :return: An object for grid search
    """
    clf = MultinomialNB()
    NB_grid = [{'alpha': [0.0001, 0.001, 0.01, 0.1, 0.8, 1, 10], 'fit_prior': [True, False]}]
    gridsearchNB = GridSearchCV(clf, NB_grid, cv=5)
    return gridsearchNB.fit(counts_train, lab_train)


def svm_classifier(counts_train, lab_train):
    """
    Support Vector Machine classifier
    :param counts_train: the transformed training data
    :param lab_train: the training labels
    :return: An object for grid search
    """
    clf = svm.SVC()
    SVM_grid = [{'C': [0.0001, 0.001, 0.01, 0.1, 0.8, 1, 10], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]
    gridsearchSVM = GridSearchCV(clf, SVM_grid, cv=5)
    return gridsearchSVM.fit(counts_train, lab_train)


def load_test(file):
    """
    load the reviews from test file
    :param file: the path of test file
    :return: test reviews
    """
    reviews = []
    f = open(file)
    reader = csv.reader(f)
    for line in reader:
        review = line[0].strip().split('\t')
        reviews.append(review[0].lower())
    f.close()
    return reviews


def write_test_file(file, labels):
    """
    Write the predicted answers onto test file
    :param file:
    :param labels:
    :return:
    """
    f = open(file)
    reader = csv.reader(f)
    des = []
    for line in reader:
        des.append(line[0])
    f.close()

    f = open(file, 'w')
    writer = csv.writer(f)
    for i in range(len(labels)):
        if labels[i] == 1:
            label = 'Data Scientist'
        elif labels[i] == 2:
            label = 'Software Engineer'
        else:
            label = 'Data Engineer'
        writer.writerow([des[i], label])
    f.close()


def test_case(test_file):
    """
    test if this code could do the job, with less data, less models, run faster
    :param test_file:
    :return:
    """
    files = ['New+York_data+scientist.csv', 'SE_NY.csv', 'New+York_data+engineer.csv']

    start = time()
    print('start training...')

    rev_train, lab_train = loadData(files=files)
    rev_test = load_test(test_file)
    print(f"loading data finished, run time: {time() - start}")

    # remove the noise
    rev_train = Filter(rev_train)
    rev_test = Filter(rev_test)

    # Build a counter based on the training dataset
    counter = CountVectorizer(stop_words=stopwords.words('english'))
    counter.fit(rev_train)

    # count the number of times each term appears in a document and transform each doc into a count vector
    counts_train = counter.transform(rev_train)  # transform the training data
    counts_test = counter.transform(rev_test)  # transform the testing data

    # fit the models
    lgr_time = time()
    lgr_classifier(counts_train, lab_train)
    print(f"Logistic regression finished, run time: {time() - lgr_time}")

    nb_time = time()
    nb_classifier(counts_train, lab_train)
    print(f"Naive Bayes finished, run time: {time() - nb_time}")

    predictors = [('lreg', LogisticRegression()), ('nb', MultinomialNB())]

    ans = vt(predictors, counts_test, counts_train, lab_train)
    print(f"type of model: {type(ans)}\nmodel: \n{ans}")

    write_test_file(test_file, labels=ans)
    print(f"all finished, run time: {time() - start}")


def main(test_file):
    # {city}_{tittle}.csv
    files = ['New+York_data+scientist.csv', 'New+York_software+engineer', 'New+York_data+engineer',
             'Seattle_data+scientist.csv', 'Seattle_software+engineer', 'Seattle_data+engineer',
             'Palo+Alto_data+scientist.csv', 'Palo+Alto_software+engineer', 'Palo+Alto_data+engineer', ]

    start = time()
    print('start training...')

    rev_train, lab_train = loadData(files=files)
    rev_test = load_test(test_file)
    print(f"loading data finished, run time: {time() - start}")

    # remove the noise
    rev_train = Filter(rev_train)
    rev_test = Filter(rev_test)

    # Build a counter based on the training dataset
    counter = CountVectorizer(stop_words=stopwords.words('english'))
    counter.fit(rev_train)

    # count the number of times each term appears in a document and transform each doc into a count vector
    counts_train = counter.transform(rev_train)  # transform the training data
    counts_test = counter.transform(rev_test)  # transform the testing data

    # fit the models
    lgr_time = time()
    lgr_classifier(counts_train, lab_train)
    print(f"Logistic regression finished, run time: {time() - lgr_time}")

    rf_time = time()
    rf_classifier(counts_train, lab_train)
    print(f"Random Forest finished, run time: {time() - rf_time}")

    knn_time = time()
    knn_classifier(counts_train, lab_train)
    print(f"KNN finished, run time: {time() - knn_time}")

    dt_time = time()
    dt_classifier(counts_train, lab_train)
    print(f"Decision tree finished, run time: {time() - dt_time}")

    nb_time = time()
    nb_classifier(counts_train, lab_train)
    print(f"Naive Bayes finished, run time: {time() - nb_time}")

    svm_time = time()
    svm_classifier(counts_train, lab_train)
    print(f"SVM finished, run time: {time() - svm_time}")

    predictors = [('lreg', LogisticRegression()), ('rf', RandomForestClassifier()), ('knn', KNeighborsClassifier()),
                  ('dt', DecisionTreeClassifier()), ('nb', MultinomialNB()), ('svm', svm.SVC())]

    ans = vt(predictors, counts_test, counts_train, lab_train)

    write_test_file(test_file, labels=ans)
    print(f"all finished, run time: {time() - start}")


if __name__ == '__main__':
    test_file = 'test.csv'  # put your test file here
    # test_case(test_file=test_file)
    main(test_file=test_file)
