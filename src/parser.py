__author__ = 'babak'

from bs4 import BeautifulSoup
import glob, os
import re
import numpy as np
import string
import nltk         #Natural Language Toolkit
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm,neighbors
from lxml.html.clean import Cleaner
from sklearn import metrics
import sys
class Analyser():
    def __init__(self,):
        self.cleaner = Cleaner() #for cleaning the html files
        self.cleaner.javascripts = True #Removes any Javascript, like an onclick attribute
        self.cleaner.scripts =  True #Removes any <script> tags.
        self.cleaner.style = True #Removes any style tags or attributes.
        self.cleaner.links = True #Removes any <link> tags


    def classifier_NB(self, train_tfidf, test_tfidf,labels):
        '''
        Naieve Bayes Classifier
        '''
        clf = MultinomialNB().fit(train_tfidf, labels)
        predicted = clf.predict(test_tfidf)
        return predicted
    
    def classifier_svm(self, train_tfidf, test_tfidf, labels):
        '''
        SVM with RBF kernel
        '''
        clf = svm.SVC()
        clf.fit(train_tfidf, labels) 
        predicted = clf.predict(test_tfidf)
        return predicted
    
    def classifier_KNN(self, train_tfidf, test_tfidf, labels, n_neighbors=2):
        '''
        k-nearest neighbor classifier
        '''
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        clf.fit(train_tfidf, labels)
        predicted = clf.predict(test_tfidf)
        return predicted
    
    def classification(self, train_data, test_data, y_train, classifier='NB'):
        '''
        master classification method
        '''
        vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word', min_df = 0, stop_words='english', ngram_range=(1,3))
        train_tfidf = vectorizer.fit_transform(train_data)
        test_tfidf = vectorizer.transform(test_data)
        if classifier.upper() == 'NB':
            predicted = self.classifier_NB(train_tfidf,test_tfidf, y_train)
        elif classifier.upper() == 'SVM':
            predicted = self.classifier_svm(train_tfidf,test_tfidf, y_train)
        elif classifier.upper() == 'KNN': 
            predicted = self.classifier_KNN(train_tfidf,test_tfidf, y_train)
        else:
            sys.exit('the {0} classifiers is undefined'.format(classifier))
        return predicted 
    
    def evalution(self, predicted, y_test, test_docs):
        '''
        evalution of prediction 
        '''
        print('>>relative accuracy: ')
        print(np.mean(predicted == y_test))
        
        print(metrics.classification_report(y_test, predicted))

        for i, doc in enumerate(test_docs):
            print(doc, '--->',predicted[i])

    def html_source_processing(self, data_path):
        '''
        read, clean and extract features from the  html files 
        '''
        os.chdir(data_path)
        labels = []
        corpus =[]
        files = glob.glob('*.html')
        for productPage in files:
            print(productPage)
            soup = BeautifulSoup(open(productPage),"html.parser")
            text = self.cleaner.clean_html(soup.get_text())
            textWithoutPunctuation = re.sub("[^a-zA-Z]", " ", text )

            lower_case = textWithoutPunctuation.lower()        # Convert to lower case
            corpus.append(lower_case)
            labels.append(productPage.split('_')[1].split('.')[0])
        print(labels)
        return corpus ,labels

if __name__ == "__main__":
    docs_test = [
            'AmazonCamera_1.html',
            'xleft_1.html',
            'Amazonram_1.html',
            'AmazonCanon_0.html',
            'Amazonfromseller_1.html',
            ]
    anlr = Analyser()
    train_data,y_train = anlr.html_source_processing("../Data/train-data/")
    test_data,y_test = anlr.html_source_processing("../test-data/")

    predicted = anlr.classification(train_data, test_data, y_train,'knn')
    anlr.evalution(predicted, y_test, docs_test )
