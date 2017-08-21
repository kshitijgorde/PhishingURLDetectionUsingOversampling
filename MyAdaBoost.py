import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score,accuracy_score
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import collections
from Resampling import Resampling
class MyAdaBoostClassifier():
    'Implements Ada boost classifier'

    #---- Ada Boost Classifier without Oversampling
    def adaBoostNoOversampling(self,featureMatrix, phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/'+technique+'AdaBoostResultsNoOversampling.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            URL_Train = list(URL_Train)
            for everyFeature in fakeFeatureMatrix:
                URL_Train.append(everyFeature)

            Label_Train = list(Label_Train)
            for everyFakeLabel in fakeLabels:
                Label_Train.append(everyFakeLabel)
            parameters_adaBoost = {'n_estimators': [50, 100, 1000], 'algorithm': ('SAMME', 'SAMME.R')}
            estimator = AdaBoostClassifier()
            clf = GridSearchCV(estimator, parameters_adaBoost, n_jobs=8)
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

        predictionResult.write("Ada Boost Classification without Oversampling Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    #-------------- Ada Boost Classifier with SMOTE Oversampling --------------------------------------
    def adaBoostSMOTE(self,featureMatrix, phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/'+technique+'AdaBoostResultsSmote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_adaBoost = {'n_estimators': [50, 100, 1000], 'algorithm': ('SAMME', 'SAMME.R')}
            URL_Train = list(URL_Train)
            for everyFeature in fakeFeatureMatrix:
                URL_Train.append(everyFeature)

            Label_Train = list(Label_Train)
            for everyFakeLabel in fakeLabels:
                Label_Train.append(everyFakeLabel)
            estimator = AdaBoostClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.smoteOversampling(URL_Train,Label_Train)
            clf = GridSearchCV(estimator, parameters_adaBoost, n_jobs=8)
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

        predictionResult.write("Ada Boost Classification with SMOTE Oversampling Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))


    #--------- Ada Boost Classifier with Borderline-1 SMOTE----------------------------------------
    def adaBoostb1SMOTE(self,featureMatrix, phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/'+technique+'AdaBoostResultsSmoteb1.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_adaBoost = {'n_estimators': [50, 100, 1000], 'algorithm': ('SAMME', 'SAMME.R')}
            URL_Train = list(URL_Train)
            for everyFeature in fakeFeatureMatrix:
                URL_Train.append(everyFeature)

            Label_Train = list(Label_Train)
            for everyFakeLabel in fakeLabels:
                Label_Train.append(everyFakeLabel)
            estimator = AdaBoostClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.b1smoteOversampling(URL_Train,Label_Train)
            clf = GridSearchCV(estimator, parameters_adaBoost, n_jobs=8)
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

        predictionResult.write("Ada Boost Classification with Borderline-1 SMOTE Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    # ---------------- Ada Boost Classifier with Borderline-2 SMOTE ----------------------------

    def adaBoostb2SMOTE(self,featureMatrix, phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/'+technique+'AdaBoostResultsSmoteb2.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_adaBoost = {'n_estimators': [50, 100, 1000], 'algorithm': ('SAMME', 'SAMME.R')}
            URL_Train = list(URL_Train)
            for everyFeature in fakeFeatureMatrix:
                URL_Train.append(everyFeature)

            Label_Train = list(Label_Train)
            for everyFakeLabel in fakeLabels:
                Label_Train.append(everyFakeLabel)
            estimator = AdaBoostClassifier()
            rm = Resampling()
            featureMatrix2,phishingLabel2 = rm.b2smoteOversampling(URL_Train,Label_Train)
            clf = GridSearchCV(estimator, parameters_adaBoost, n_jobs=8)
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

        predictionResult.write("Ada Boost Classification with Borderline-2 SMOTE Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    # ---------------------- Ada Boost Classifier with SVM Smote -----------------------------------------

    def adaBoostSVMSmote(self,featureMatrix, phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/'+technique+'AdaBoostResultsSVMSmote.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_adaBoost = {'n_estimators': [50, 100, 1000], 'algorithm': ('SAMME', 'SAMME.R')}
            URL_Train = list(URL_Train)
            for everyFeature in fakeFeatureMatrix:
                URL_Train.append(everyFeature)

            Label_Train = list(Label_Train)
            for everyFakeLabel in fakeLabels:
                Label_Train.append(everyFakeLabel)
            estimator = AdaBoostClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.SVMsmoteOversampling(URL_Train,Label_Train)
            clf = GridSearchCV(estimator, parameters_adaBoost, n_jobs=8)
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

        predictionResult.write("Ada Boost Classification with SVM Smote Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    # --------------- Ada Boost Classifier with Random Minority Oversampling ------------------
    def adaBoostRMR(self,featureMatrix, phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/'+technique+'AdaBoostResultsRMR.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_adaBoost = {'n_estimators': [50, 100, 1000], 'algorithm': ('SAMME', 'SAMME.R')}

            URL_Train = list(URL_Train)
            for everyFeature in fakeFeatureMatrix:
                URL_Train.append(everyFeature)

            Label_Train = list(Label_Train)
            for everyFakeLabel in fakeLabels:
                Label_Train.append(everyFakeLabel)
            estimator = AdaBoostClassifier()
            rm = Resampling()
            featureMatrix2, phishingLabel2 = rm.RMROversampling(URL_Train,Label_Train)
            clf = GridSearchCV(estimator, parameters_adaBoost, n_jobs=8)
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

        predictionResult.write("Ada Boost Classification with Random Minority Oversampling Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))

    #-------------------- ADa Boost Classifier with ADASYN Oversampling ---------------------------------------

    def adaBoostADASYN(self,featureMatrix, phishingURLLabel,fakeFeatureMatrix,fakeLabels,technique):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/'+technique+'AdaBoostResultsADASYN.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_adaBoost = {'n_estimators': [50, 100, 1000], 'algorithm': ('SAMME', 'SAMME.R')}
            URL_Train = list(URL_Train)
            for everyFeature in fakeFeatureMatrix:
                URL_Train.append(everyFeature)

            Label_Train = list(Label_Train)
            for everyFakeLabel in fakeLabels:
                Label_Train.append(everyFakeLabel)

            estimator = AdaBoostClassifier()
            rm = Resampling()
            featureMatrix2,phishingLabel2 = rm.ADASYNOversampling(URL_Train,Label_Train)
            clf = GridSearchCV(estimator, parameters_adaBoost, n_jobs=8)
            clf.fit(featureMatrix2,phishingLabel2)
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

        predictionResult.write("Ada Boost Classification with ADASYN Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))