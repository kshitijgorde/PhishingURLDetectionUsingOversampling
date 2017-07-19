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
def getLabelTerm(URldict, feature):
	#given a word vector, find it's corresponding CVE_Label
	label = [key for key, value in URldict.iteritems() if (value == feature).all()][0]
	return label



datasetObject = LoadCSVDataset()
cleanedDataset,header = datasetObject.loadDataset()
#print cleanedDataset[header[0]][34001]
#Here I've obtained the cleaned Dataset with N/A removed.
#Below I've created methods to create a csv file with all the relevant String features.
featureObject = FeaturesCSV()
featureObject.createCSVFile(cleanedDataset,header)
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


print len(featureMatrix)
print len(phishingLabel)



# #Once obtained Feature matrix, make a call to Classifiers
#    #17909
decisionTree = MyDecisionTreeClassifier()
decisionTree.decisionTreeF1(featureMatrix,phishingLabel)
# #
# svm = MySupportVector()
# svm.supportVectorF1(featureMatrix,phishingLabel)
# #
# rf = MyRandomForestClassifier()
# rf.randomForestF1(featureMatrix,phishingLabel)
# #
# ada = MyAdaBoostClassifier()
# ada.adaBoostF1(featureMatrix,phishingLabel)
# #
#rbm = MyRBM()
#rbm.rbmClassifyF1(featureMatrix,phishingLabel)

