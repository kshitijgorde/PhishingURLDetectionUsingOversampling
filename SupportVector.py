from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score,accuracy_score
import numpy as np
import os
import collections
class MySupportVector():
    'Implements support vector classifier'
    def supportVectorF1(self,featureMatrix,phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name+'/SVMResults.txt', 'a+')
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

        predictionResult.write("SVM Classification Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))