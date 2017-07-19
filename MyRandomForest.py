import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score,accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import collections
class MyRandomForestClassifier():
    'Implelments Random Forest Classifier'
    def randomForestF1(self,featureMatrix,phishingURLLabel):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        predictionResult = open(dir_name + '/RandomForestResults.txt', 'a+')
        predictionResult.truncate()
        accuracy_matrix = []
        try:
            URL_Train, URL_Test, Label_Train, Label_Test = train_test_split(featureMatrix, phishingURLLabel,
                                                                            test_size=0.20)
            parameters_RandomForest = {'n_estimators': [10, 100, 1000], 'criterion': ('gini', 'entropy'),
                                       'oob_score': (True, False), 'warm_start': (True, False)}
            estimator = RandomForestClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))  # should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount / float(totalSamples)))

            clf = GridSearchCV(estimator, parameters_RandomForest, n_jobs=8)
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

        predictionResult.write("Random Forest Classification Completed with Avg. Score: " + str(np.mean(accuracy_matrix)))