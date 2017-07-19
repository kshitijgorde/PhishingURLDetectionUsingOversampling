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

    def decisionTreeF1(self,featureMatrix,phishingURLLabel):
        re = Resampling()

        'Calculates F1-score by train-test split of standard 80%'
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





            #You should oversample only the Minority class

            #Create the minority feature matrix for oversampling and then add it to the URL_Train
            phishyTrainerFeatures=[]
            phishyTrainerLabel=[]
            NonphishyTrainerFeatures = []
            NonphishyTrainerLabel = []

            for i in range(0,len(Label_Train)):
                if Label_Train[i] == 1 or Label_Train[i] == '1' or Label_Train[i]==1.0 or Label_Train[i]=='1.0':
                    phishyTrainerFeatures.append(URL_Train[i])
                    phishyTrainerLabel.append(Label_Train[i])
                else:
                    NonphishyTrainerFeatures.append(URL_Train[i])
                    NonphishyTrainerLabel.append(Label_Train[i])





            #check which one is minority
            minorityClassFeatures = []
            minorityClassLabels =[]
            minority = ''
            print "Printing length"
            print len(phishyTrainerLabel)
            print len(NonphishyTrainerLabel)
            if len(phishyTrainerLabel) > len(NonphishyTrainerLabel):
                minorityClassFeatures = NonphishyTrainerFeatures
                minorityClassLabels = NonphishyTrainerLabel
                minority = 'Non-Phishy'
            else:
                minorityClassFeatures = phishyTrainerFeatures
                minorityClassLabels = phishyTrainerLabel
                minority = 'Phishy'

            print 'Minority class is:' + minority



            print 'Over-sampling the Minority class...'

            featureMatrix2, phishingLabel2 = re.smoteOversampling(minorityClassFeatures, minorityClassLabels)

            for everyItem in featureMatrix2:
                minorityClassFeatures.append(everyItem)

            for everylab in phishingLabel2:
                minorityClassLabels.append(everylab)

            print 'feature len:'+ str(len(featureMatrix2))
            print 'Minotiry len:' + str(len(minorityClassLabels))

            featureMatrix2 = minorityClassFeatures
            phishingLabel2 = minorityClassLabels

            OversampledURLFeatures=[]
            OversampledURLLabels=[]
            if minority=='Phishy':
                #phishy

                for everyFeature in featureMatrix2:
                    NonphishyTrainerFeatures.append(everyFeature)

                for everyLabel in phishingLabel2:
                    NonphishyTrainerLabel.append(everyLabel)

                OversampledURLFeatures = NonphishyTrainerFeatures
                OversampledURLLabels = NonphishyTrainerLabel

            else:
                #add non phishy
                for everyFeature in featureMatrix2:
                    phishyTrainerFeatures.append(everyFeature)

                for everyLabel in phishingLabel2:
                    phishyTrainerLabel.append(everyLabel)

                OversampledURLFeatures = phishyTrainerFeatures
                OversampledURLLabels = phishyTrainerLabel


            print '\n\nAfter Oversampling:'
            print 'Total:' + str(len(OversampledURLLabels))
            print 'Phishy: ' + str(list(OversampledURLLabels).count(1))
            print 'Non Phishy:' + str(list(OversampledURLLabels).count(0))


            phishyTrainerFeatures = numpy.array(phishyTrainerFeatures, dtype='double')
            phishyTrainerLabel = numpy.array(phishyTrainerLabel, dtype='double')
            NonphishyTrainerFeatures = numpy.array(NonphishyTrainerFeatures, dtype='double')
            NonphishyTrainerLabel = numpy.array(NonphishyTrainerLabel, dtype='double')
            minorityClassFeatures = numpy.array(minorityClassFeatures, dtype='double')
            minorityClassLabels = numpy.array(minorityClassLabels, dtype='double')
            OversampledURLFeatures = numpy.array(OversampledURLFeatures,dtype='double')
            OversampledURLLabels = numpy.array(OversampledURLLabels,dtype='double')
            parameters_DecisionTree = {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random')}
            estimator = DecisionTreeClassifier()
            # totalSamples = len(Label_Train)
            # positiveCount = int(Label_Train.count('1'))       #should be 65% of total
            # predictionResult.write("Percentage of positive samples in training phase: %.2f " % (positiveCount/float(totalSamples)))
            clf = GridSearchCV(estimator, parameters_DecisionTree, n_jobs=8)
            clf.fit(OversampledURLFeatures,OversampledURLLabels)
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