from LoadCSVDataset import LoadCSVDataset
from FeatureCSV import FeaturesCSV
from LoadFeatures import LoadFeatures
from GanPreProcess import GanPreProcess
import numpy
from DecisionTreeClassifier import MyDecisionTreeClassifier
from SupportVector import MySupportVector
from MyRandomForest import MyRandomForestClassifier
from MyRBM import MyRBM
from MyAdaBoost import MyAdaBoostClassifier
from Resampling import Resampling
from collections import OrderedDict

datasetObject = LoadCSVDataset()
cleanedDataset,header = datasetObject.loadDataset()
#print cleanedDataset[header[0]][34001]
#Here I've obtained the cleaned Dataset with N/A removed.
#Below I've created methods to create a csv file with all the relevant String features.
featureObject = FeaturesCSV()
threshold = 10
featureObject.createCSVFile(cleanedDataset,header,threshold)
features = LoadFeatures()
featureMatrix,phishingLabel = features.loadFeatures()

phishyURLs = features.loadPositiveFeatures()
nonPhishyURLs = features.loadNegativeFeatures()



print 'Phishy:' + str(phishingLabel.count(1))
print 'Non Phishy:' + str(phishingLabel.count(0))
pre = GanPreProcess()

if len(phishyURLs) > len(nonPhishyURLs):
	for everyURL in nonPhishyURLs:
		pre.preProcessURLs(everyURL)
else:
	for everyURL in phishyURLs:
		pre.preProcessURLs(everyURL)

featureMatrix = numpy.array(featureMatrix,dtype='double')
phishingLabel = numpy.array(phishingLabel,dtype='double')

print 'Before'
print len(featureMatrix)

decisionTree = MyDecisionTreeClassifier()
decisionTree.decisionTreeSMOTE(featureMatrix,phishingLabel)
#


print 'Value of FeatureMatrix'
print len(featureMatrix)
decisionTree2 = MyDecisionTreeClassifier()
decisionTree2.decisionTreebSMOTE1(featureMatrix,phishingLabel)
#
# decisionTree3 = MyDecisionTreeClassifier()
# decisionTree3.decisionTreebSMOTE2(featureMatrix,phishingLabel)
#
# decisionTree4 = MyDecisionTreeClassifier()
# decisionTree4.decisionTreeSVM_SMOTE(featureMatrix,phishingLabel)
#
# decisionTree5 = MyDecisionTreeClassifier()
# decisionTree5.decisionTreeADASYN(featureMatrix,phishingLabel)
#
# decisionTree6 = MyDecisionTreeClassifier()
# decisionTree6.decisionTreeRMR(featureMatrix,phishingLabel)
#
# decisionTree7 =  MyDecisionTreeClassifier()
# decisionTree7.decisionTreeNoOversampling(featureMatrix,phishingLabel)