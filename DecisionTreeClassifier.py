from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from Resampling import Resampling
import numpy

import os
import collections
class MyDecisionTreeClassifier():
    'Handles Predicting Phishing URL by implementing scikit-learn DecisionTree Classifier'

    def decisionTreeSMOTE(self,featureMatrix,phishingURLLabel):
        re = Resampling()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/DecisionTreeResults.txt','a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20,random_state=40)

            #----------- Analysis

            print 'Train Test Split:'
            print 'Training Values:'
            print 'Total:' + str(len(Label_Train))
            print 'Phishy: '+str(list(Label_Train).count(1))
            print 'Non Phishy:' + str(list(Label_Train).count(0))

            print 'Testing Values:'
            print 'Total:' + str(len(Label_Test))
            print 'Phishy: ' + str(list(Label_Test).count(1))
            print 'Non Phishy:' + str(list(Label_Test).count(0))
            print 'Performing Oversampling'
            featureMatrix2, phishingLabel2 = re.smoteOversampling(URL_Train, Label_Train)

            print 'After Oversampling...'
            print 'Total: '+str(len(phishingLabel2))
            print 'Ratio: '
            print collections.Counter(phishingLabel2)

            parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
            estimator = DecisionTreeClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))       #should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount/float(totalSamples)))
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=1)
            clf.fit(featureMatrix2,phishingLabel2)
            result = clf.predict(URL_Test)
            #print "Type of REsult is:"
            #print type(result)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("Decision Tree Classification Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))
        print 'Decision Tree Classification Completed with Avg. Score: ' + str(np.mean(accuracy_matrix))

    #------------- Decision Tree Classifier with Random Minority Over-sampling with Replacement

    def decisionTreeRMR(self,featureMatrix,phishingURLLabel):
        re = Resampling()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/DecisionTreeResults.txt','a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20,random_state=40)

            #----------- Analysis

            print 'Train Test Split:'
            print 'Training Values:'
            print 'Total:' + str(len(Label_Train))
            print 'Phishy: '+str(list(Label_Train).count(1))
            print 'Non Phishy:' + str(list(Label_Train).count(0))

            print 'Testing Values:'
            print 'Total:' + str(len(Label_Test))
            print 'Phishy: ' + str(list(Label_Test).count(1))
            print 'Non Phishy:' + str(list(Label_Test).count(0))
            print 'Performing Oversampling'
            featureMatrix2, phishingLabel2 = re.RMROversampling(URL_Train, Label_Train)

            print 'After Oversampling...'
            print 'Total: '+str(len(phishingLabel2))
            print 'Ratio: '
            print collections.Counter(phishingLabel2)

            parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
            estimator = DecisionTreeClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))       #should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount/float(totalSamples)))
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=1)
            clf.fit(featureMatrix2,phishingLabel2)
            result = clf.predict(URL_Test)
            #print "Type of REsult is:"
            #print type(result)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("Decision Tree Classification Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))
        print 'Decision Tree Classification Completed with Avg. Score: ' + str(np.mean(accuracy_matrix))

    #---------- Decision Tree Classifier with Borderline SMOTE-1 ----------------------------

    def decisionTreebSMOTE1(self,featureMatrix,phishingURLLabel):
        re = Resampling()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/DecisionTreeResults.txt','a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20,random_state=40)

            #----------- Analysis

            print 'Train Test Split:'
            print 'Training Values:'
            print 'Total:' + str(len(Label_Train))
            print 'Phishy: '+str(list(Label_Train).count(1))
            print 'Non Phishy:' + str(list(Label_Train).count(0))

            print 'Testing Values:'
            print 'Total:' + str(len(Label_Test))
            print 'Phishy: ' + str(list(Label_Test).count(1))
            print 'Non Phishy:' + str(list(Label_Test).count(0))
            print 'Performing Oversampling'
            featureMatrix2, phishingLabel2 = re.b1smoteOversampling(URL_Train, Label_Train)

            print 'After Oversampling...'
            print 'Total: '+str(len(phishingLabel2))
            print 'Ratio: '
            print collections.Counter(phishingLabel2)

            parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
            estimator = DecisionTreeClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))       #should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount/float(totalSamples)))
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=1)
            clf.fit(featureMatrix2,phishingLabel2)
            result = clf.predict(URL_Test)
            #print "Type of REsult is:"
            #print type(result)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("Decision Tree Classification Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))
        print 'Decision Tree Classification Completed with Avg. Score: ' + str(np.mean(accuracy_matrix))

    # ------------------------- Decision Tree Classifier with Borderline SMOTE 2 ---------------------------

    def decisionTreebSMOTE2(self,featureMatrix,phishingURLLabel):
        re = Resampling()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/DecisionTreeResults.txt','a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20,random_state=40)

            #----------- Analysis

            print 'Train Test Split:'
            print 'Training Values:'
            print 'Total:' + str(len(Label_Train))
            print 'Phishy: '+str(list(Label_Train).count(1))
            print 'Non Phishy:' + str(list(Label_Train).count(0))

            print 'Testing Values:'
            print 'Total:' + str(len(Label_Test))
            print 'Phishy: ' + str(list(Label_Test).count(1))
            print 'Non Phishy:' + str(list(Label_Test).count(0))
            print 'Performing Oversampling'
            featureMatrix2, phishingLabel2 = re.b2smoteOversampling(URL_Train, Label_Train)

            print 'After Oversampling...'
            print 'Total: '+str(len(phishingLabel2))
            print 'Ratio: '
            print collections.Counter(phishingLabel2)

            parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
            estimator = DecisionTreeClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))       #should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount/float(totalSamples)))
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=1)
            clf.fit(featureMatrix2,phishingLabel2)
            result = clf.predict(URL_Test)
            #print "Type of REsult is:"
            #print type(result)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("Decision Tree Classification Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))
        print 'Decision Tree Classification Completed with Avg. Score: ' + str(np.mean(accuracy_matrix))

    # Decision Tree Classifier for Support Vector Machine SMOTE Technique

    def decisionTreeSVM_SMOTE(self,featureMatrix,phishingURLLabel):
        re = Resampling()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/DecisionTreeResults.txt','a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20,random_state=40)

            #----------- Analysis

            print 'Train Test Split:'
            print 'Training Values:'
            print 'Total:' + str(len(Label_Train))
            print 'Phishy: '+str(list(Label_Train).count(1))
            print 'Non Phishy:' + str(list(Label_Train).count(0))

            print 'Testing Values:'
            print 'Total:' + str(len(Label_Test))
            print 'Phishy: ' + str(list(Label_Test).count(1))
            print 'Non Phishy:' + str(list(Label_Test).count(0))
            print 'Performing Oversampling'
            featureMatrix2, phishingLabel2 = re.SVMsmoteOversampling(URL_Train, Label_Train)

            print 'After Oversampling...'
            print 'Total: '+str(len(phishingLabel2))
            print 'Ratio: '
            print collections.Counter(phishingLabel2)

            parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
            estimator = DecisionTreeClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))       #should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount/float(totalSamples)))
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=1)
            clf.fit(featureMatrix2,phishingLabel2)
            result = clf.predict(URL_Test)
            #print "Type of REsult is:"
            #print type(result)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("Decision Tree Classification Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))
        print 'Decision Tree Classification Completed with Avg. Score: ' + str(np.mean(accuracy_matrix))

    # --------------- Decision Tree Classifier with ADASYN (Adaptive Synthetic Sampling Approach for imbalanced Learning---

    def decisionTreeADASYN(self,featureMatrix,phishingURLLabel):
        re = Resampling()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/DecisionTreeResults.txt','a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20,random_state=40)

            #----------- Analysis

            print 'Train Test Split:'
            print 'Training Values:'
            print 'Total:' + str(len(Label_Train))
            print 'Phishy: '+str(list(Label_Train).count(1))
            print 'Non Phishy:' + str(list(Label_Train).count(0))

            print 'Testing Values:'
            print 'Total:' + str(len(Label_Test))
            print 'Phishy: ' + str(list(Label_Test).count(1))
            print 'Non Phishy:' + str(list(Label_Test).count(0))
            print 'Performing Oversampling'
            featureMatrix2, phishingLabel2 = re.ADASYNOversampling(URL_Train, Label_Train)

            print 'After Oversampling...'
            print 'Total: '+str(len(phishingLabel2))
            print 'Ratio: '
            print collections.Counter(phishingLabel2)

            parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
            estimator = DecisionTreeClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))       #should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount/float(totalSamples)))
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=1)
            clf.fit(featureMatrix2,phishingLabel2)
            result = clf.predict(URL_Test)
            #print "Type of REsult is:"
            #print type(result)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("Decision Tree Classification Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))
        print 'Decision Tree Classification Completed with Avg. Score: ' + str(np.mean(accuracy_matrix))

    # Decision Tree Classifier without any Oversampling Technique

    def decisionTreeNoOversampling(self,featureMatrix,phishingURLLabel):
        re = Resampling()
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/DecisionTreeResults.txt','a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix,phishingURLLabel,test_size=0.20,random_state=40)

            #----------- Analysis

            print 'Train Test Split:'
            print 'Training Values:'
            print 'Total:' + str(len(Label_Train))
            print 'Phishy: '+str(list(Label_Train).count(1))
            print 'Non Phishy:' + str(list(Label_Train).count(0))

            print 'Testing Values:'
            print 'Total:' + str(len(Label_Test))
            print 'Phishy: ' + str(list(Label_Test).count(1))
            print 'Non Phishy:' + str(list(Label_Test).count(0))
            parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
            estimator = DecisionTreeClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))       #should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount/float(totalSamples)))
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=1)
            clf.fit(URL_Train,Label_Train)
            result = clf.predict(URL_Test)
            #print "Type of REsult is:"
            #print type(result)
            predictionResult.write(str(result))
            predictionResult.flush()
            predictionResult.write("\nThe 1's are:" + str(collections.Counter(result)))
            predictionResult.flush()
            f1Score = f1_score(Label_Test, result, pos_label='1', average='macro')
            predictionResult.write("\nThe f1_score is:" + str(f1Score))
            predictionResult.flush()
            accuracy_matrix.append(f1Score)
        except Exception as e:
            predictionResult.write(str(e))

        predictionResult.write("Decision Tree Classification Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))
        print 'Decision Tree Classification Completed with Avg. Score: ' + str(np.mean(accuracy_matrix))