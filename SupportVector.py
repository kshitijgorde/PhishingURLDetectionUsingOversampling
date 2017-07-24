from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score,accuracy_score
import numpy as np
import os
import collections
from Resampling import Resampling
class MySupportVector():
    'Implements support vector classifier'

    def supportVectorNoOversampling(self,featureMatrix,phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/SVMResultsNoOversampling.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_SVC = {'C': [1.0, 10.0,15.0,20.0,22.0,30.0], 'kernel': ('rbf', 'sigmoid', 'linear', 'poly'), 'degree': [3, 4],
                              'probability': (True, False), 'shrinking': (True, False),
                              'decision_function_shape': ('ovo', 'ovr', 'None')}
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))  # should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount / float(totalSamples)))
            estimator = SVC()
            clf = GridSearchCV(estimator, parameters_SVC, n_jobs=8)
            clf.fit(URL_Train, Label_Train)
            result = clf.predict(URL_Test)
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

        predictionResult.write("SVM Classification without Oversampling Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    #--------------- SVM Classifier with SMOTE---------------------
    def supportVectorSMOTE(self,featureMatrix,phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/SVMResultsSmote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_SVC = {'C': [1.0, 10.0,15.0,20.0,22.0,30.0], 'kernel': ('rbf', 'sigmoid', 'linear', 'poly'), 'degree': [3, 4],
                              'probability': (True, False), 'shrinking': (True, False),
                              'decision_function_shape': ('ovo', 'ovr', 'None')}
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))  # should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount / float(totalSamples)))
            estimator = SVC()
            clf = GridSearchCV(estimator, parameters_SVC, n_jobs=8)
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
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

        predictionResult.write("SVM Classification with Smote Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    #--------------SVM Borderline-1 SMOTE
    def supportVectorb1Smote(self,featureMatrix,phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/SVMResultsb1Smote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_SVC = {'C': [1.0, 10.0,15.0,20.0,22.0,30.0], 'kernel': ('rbf', 'sigmoid', 'linear', 'poly'), 'degree': [3, 4],
                              'probability': (True, False), 'shrinking': (True, False),
                              'decision_function_shape': ('ovo', 'ovr', 'None')}
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))  # should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount / float(totalSamples)))
            estimator = SVC()
            clf = GridSearchCV(estimator, parameters_SVC, n_jobs=8)
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
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

        predictionResult.write("SVM Classification with Borderline-1 Smote Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    #--------- SVM Borderline-2 Smote----------------------

    def supportVectorb2Smote(self,featureMatrix,phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/SVMResultsB2Smote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_SVC = {'C': [1.0, 10.0,15.0,20.0,22.0,30.0], 'kernel': ('rbf', 'sigmoid', 'linear', 'poly'), 'degree': [3, 4],
                              'probability': (True, False), 'shrinking': (True, False),
                              'decision_function_shape': ('ovo', 'ovr', 'None')}
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))  # should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount / float(totalSamples)))
            estimator = SVC()
            clf = GridSearchCV(estimator, parameters_SVC, n_jobs=8)
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
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

        predictionResult.write("SVM Classification with Borderline-2 Smote Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    # --------------- SVM SVM Smote ------------------------

    def supportVectorSVMSmote(self,featureMatrix,phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/SVMResultsSVMSmote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_SVC = {'C': [1.0, 10.0,15.0,20.0,22.0,30.0], 'kernel': ('rbf', 'sigmoid', 'linear', 'poly'), 'degree': [3, 4],
                              'probability': (True, False), 'shrinking': (True, False),
                              'decision_function_shape': ('ovo', 'ovr', 'None')}

            estimator = SVC()
            clf = GridSearchCV(estimator, parameters_SVC, n_jobs=8)
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
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

        predictionResult.write("SVM Classification SVM Smote Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    # ------------------- SVM RMR -----------------------------
    def supportVectorRMR(self,featureMatrix,phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/SVMResultsRMR.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_SVC = {'C': [1.0, 10.0,15.0,20.0,22.0,30.0], 'kernel': ('rbf', 'sigmoid', 'linear', 'poly'), 'degree': [3, 4],
                              'probability': (True, False), 'shrinking': (True, False),
                              'decision_function_shape': ('ovo', 'ovr', 'None')}
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))  # should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount / float(totalSamples)))
            estimator = SVC()
            clf = GridSearchCV(estimator, parameters_SVC, n_jobs=8)
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
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

        predictionResult.write("SVM Classification with RMR Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    #------------------- SVM ADASYN ----------------------------------------------
    def supportVectorADASYN(self,featureMatrix,phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/SVMResultsADASyn.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_SVC = {'C': [1.0, 10.0,15.0,20.0,22.0,30.0], 'kernel': ('rbf', 'sigmoid', 'linear', 'poly'), 'degree': [3, 4],
                              'probability': (True, False), 'shrinking': (True, False),
                              'decision_function_shape': ('ovo', 'ovr', 'None')}

            estimator = SVC()
            clf = GridSearchCV(estimator, parameters_SVC, n_jobs=8)
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.ADASYNOversampling(URL_Train, Label_Train)
            clf.fit(featureMatrix2, phishingLabel2)
            result = clf.predict(URL_Test)
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

        predictionResult.write("SVM Classification with ADASYN Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))
