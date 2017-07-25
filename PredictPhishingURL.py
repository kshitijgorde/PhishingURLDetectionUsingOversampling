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

# datasetObject = LoadCSVDataset()
# cleanedDataset,header = datasetObject.loadDataset()
# #print cleanedDataset[header[0]][34001]
# #Here I've obtained the cleaned Dataset with N/A removed.
# #Below I've created methods to create a csv file with all the relevant String features.
# featureObject = FeaturesCSV()
threshold = 5
# fileName = 'Outresponse.csv'
# featureObject.createCSVFile(cleanedDataset,header,threshold,fileName)
# features = LoadFeatures()
# featureMatrix,phishingLabel = features.loadFeatures(fileName)
#
# phishyURLs = features.loadPositiveFeatures()
# nonPhishyURLs = features.loadNegativeFeatures()



#print 'Phishy:' + str(phishingLabel.count(1))
#print 'Non Phishy:' + str(phishingLabel.count(0))
# pre = GanPreProcess()
#
# if len(phishyURLs) > len(nonPhishyURLs):
# 	for everyURL in nonPhishyURLs:				#Uncomment this if you need to Generate Training samples for GAN
# 		pre.preProcessURLs(everyURL)
# else:
# 	for everyURL in phishyURLs:
# 		pre.preProcessURLs(everyURL)

# featureMatrix = numpy.array(featureMatrix,dtype='double')
# phishingLabel = numpy.array(phishingLabel,dtype='double')






# ------------------------   Load the fake URL's and put them into a CSV file ----------------------
features = LoadFeatures()
FakefileName = 'EbayThreshold5_NP_Cleaned.txt'
fakeCSV = LoadCSVDataset()
fakeCSVFileName = fakeCSV.createCSVForFakeURL(FakefileName,threshold)
fakeURLcleanedDataset, fakeURLheader = fakeCSV.loadDataset(fakeCSVFileName)
featurecsv = FeaturesCSV()
featurecsv.createCSVFile(fakeURLcleanedDataset, fakeURLheader,threshold,'Features_'+fakeCSVFileName)
#--------------- Once you created a Fake CSV File, get it's feature matrix -------------------------

fakeURLFeatureMatrix,fakeURLphishingLabel = features.loadFeatures('Features_'+fakeCSVFileName)


#-------------- Apply Selection Techniques -----------------------------------
#1. Using Convex-Hull Algorithm
#
fakeURLFeatureMatrix = numpy.array(fakeURLFeatureMatrix,dtype='double')



#Uncomment below for k-best Feature selections and plotting
# print 'Applying k-best Feature selections...'
# from sklearn.feature_selection import SelectKBest,chi2,f_classif,mutual_info_classif
# fakeFeatureReshaped = SelectKBest(f_classif,k=2).fit_transform(fakeURLFeatureMatrix,fakeURLphishingLabel)
# print fakeFeatureReshaped.shape
#
#
# from scipy.spatial import ConvexHull
# hull = ConvexHull(fakeFeatureReshaped)
# print hull.vertices
#
#
# import matplotlib.pyplot as plt
# plt.plot(fakeFeatureReshaped[:,0], fakeFeatureReshaped[:,1],fakeFeatureReshaped[:,2], 'o')
# for simplex in hull.simplices:
#     plt.plot(fakeFeatureReshaped[simplex, 0],fakeFeatureReshaped[simplex, 1], fakeFeatureReshaped[simplex, 2], 'k-')
#
# plt.show()
#

#----------------2. Choose k farthest Fake URLs --------------------------------

from scipy import ndimage
from scipy.spatial import distance
centroid = ndimage.center_of_mass(fakeURLFeatureMatrix)
print centroid

ec_distance = distance.euclidean(fakeURLFeatureMatrix[0],centroid)








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