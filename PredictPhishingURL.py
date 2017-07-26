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
import os

def getDistancesFromCentre(centrePoint,fakeURLFeatureMatrix,distanceRange):
    from scipy.spatial import distance
    # Calculate Distances of each point from Centroid and store in a Dictionary...

    fakeURLDistances = OrderedDict()

    for i in range(0,len(fakeURLFeatureMatrix)):
        euclDistance = distance.euclidean(fakeURLFeatureMatrix[i],centrePoint)
        fakeURLDistances[i] = euclDistance

    #Sort Dictionary by Descending Order
    sorted_FakeURLDistances = sorted(fakeURLDistances.items(),key=lambda v:v[1],reverse=True)
    longIndexes = []

    for i in range(0,distanceRange):
        longIndexes.append(sorted_FakeURLDistances[i][0])

    return longIndexes



dir_name = os.path.dirname(os.path.realpath(__file__))
fileName = 'Outresponse-ebay.csv'
datasetObject = LoadCSVDataset()
cleanedDataset,header = datasetObject.loadDataset(fileName)
# #print cleanedDataset[header[0]][34001]
# #Here I've obtained the cleaned Dataset with N/A removed.
# #Below I've created methods to create a csv file with all the relevant String features.
featureObject = FeaturesCSV()
threshold = 8

createdCSVFile = featureObject.createCSVFile(cleanedDataset,header,threshold,fileName)
features = LoadFeatures()
featureMatrix,phishingLabel = features.loadFeatures(createdCSVFile)

# phishyURLs = features.loadPositiveFeatures()
# nonPhishyURLs = features.loadNegativeFeatures()

phy = phishingLabel.count(1)
nphy = phishingLabel.count(0)

print 'Phishy:' + str(phy)
print 'Non Phishy:' + str(nphy)
# pre = GanPreProcess()
#
# if len(phishyURLs) > len(nonPhishyURLs):
# 	for everyURL in nonPhishyURLs:				#Uncomment this if you need to Generate Training samples for GAN
# 		pre.preProcessURLs(everyURL)
# else:
# 	for everyURL in phishyURLs:
# 		pre.preProcessURLs(everyURL)

featureMatrix = numpy.array(featureMatrix,dtype='double')
phishingLabel = numpy.array(phishingLabel,dtype='double')






# ------------------------   Load the fake URL's and put them into a CSV file ----------------------
features = LoadFeatures()
FakefileName = 'EbayThreshold10Phishy_Cleaned.txt'
fakeCSV = LoadCSVDataset()
threshold2 = 11
fakeCSVFileName = fakeCSV.createCSVForFakeURL(FakefileName,threshold2)
fakeURLcleanedDataset, fakeURLheader = fakeCSV.loadDataset(fakeCSVFileName)
featurecsv = FeaturesCSV()
featurecsv.createCSVFile(fakeURLcleanedDataset, fakeURLheader,threshold,fakeCSVFileName)
#--------------- Once you created a Fake CSV File, get it's feature matrix -------------------------

fakeURLFeatureMatrix,fakeURLphishingLabel = features.loadFeatures('Features_'+fakeCSVFileName)
print 'Printing Fake URL pHishing label'
print fakeURLphishingLabel
fakeURLFeatureMatrix = numpy.array(fakeURLFeatureMatrix,dtype='double')
fakeURLphishingLabel = numpy.array(fakeURLphishingLabel, dtype = 'double')

#-------------- Apply Selection Techniques -----------------------------------
#1. Using Convex-Hull Algorithm

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

#--------------------------Technique 2. Choose k farthest Fake URLs ---------------------------------------------------

centroid = numpy.mean(fakeURLFeatureMatrix,axis=0)
from scipy.spatial import distance
# Calculate Distances of each point from Centroid and store in a Dictionary...

fakeURLDistances = OrderedDict()

for i in range(0,len(fakeURLFeatureMatrix)):
    euclDistance = distance.euclidean(fakeURLFeatureMatrix[i],centroid)
    fakeURLDistances[i] = euclDistance

#Sort Dictionary by Descending Order
sorted_FakeURLDistances = sorted(fakeURLDistances.items(),key=lambda v:v[1],reverse=True)
longIndexes = []

for i in range(0,5000):
    longIndexes.append(sorted_FakeURLDistances[i][0])

finalFakeFeatures = []
for index in longIndexes:
    finalFakeFeatures.append(fakeURLFeatureMatrix[index])


fakeURLphishingLabel = fakeURLphishingLabel[:5000]
technique = 'EucledianDistance'

decisionTree7 =  MyDecisionTreeClassifier()
decisionTree7.decisionTreeNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

decisionTree3 = MyDecisionTreeClassifier()
decisionTree3.decisionTreebSMOTE2(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

decisionTree4 = MyDecisionTreeClassifier()
decisionTree4.decisionTreeSVM_SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

decisionTree5 = MyDecisionTreeClassifier()
decisionTree5.decisionTreeADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

decisionTree6 = MyDecisionTreeClassifier()
decisionTree6.decisionTreeRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

decisionTree1 = MyDecisionTreeClassifier()
decisionTree1.decisionTreeSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

decisionTree2 = MyDecisionTreeClassifier()
decisionTree2.decisionTreebSMOTE1(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)



rf1 = MyRandomForestClassifier()
rf1.randomForestNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

rf2 = MyRandomForestClassifier()
rf2.randomForestRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

rf3 = MyRandomForestClassifier()
rf3.randomForestSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

rf4 = MyRandomForestClassifier()
rf4.randomForestADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

rf5 = MyRandomForestClassifier()
rf5.randomForestSVM_Smote(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

rf6 = MyRandomForestClassifier()
rf6.randomForestb2SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

rf7 = MyRandomForestClassifier()
rf7.randomForestb1SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)


ada1 = MyAdaBoostClassifier()
ada1.adaBoostNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

ada2 = MyAdaBoostClassifier()
ada2.adaBoostADASYN(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

ada3 = MyAdaBoostClassifier()
ada3.adaBoostRMR(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

ada4 = MyAdaBoostClassifier()
ada4.adaBoostSVMSmote(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

ada5 = MyAdaBoostClassifier()
ada5.adaBoostb2SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

ada6 = MyAdaBoostClassifier()
ada6.adaBoostb1SMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

ada7 = MyAdaBoostClassifier()
ada7.adaBoostSMOTE(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

# svm1 = MySupportVector()
# svm1.supportVectorNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)
#
# svm2 = MySupportVector()
# svm2.supportVectorNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)
#
# svm3 = MySupportVector()
# svm3.supportVectorNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)
#
# svm4 = MySupportVector()
# svm4.supportVectorNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)
#
# svm5 = MySupportVector()
# svm5.supportVectorNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)
#
# svm6 = MySupportVector()
# svm6.supportVectorNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)
#
# svm7 = MySupportVector()
# svm7.supportVectorNoOversampling(featureMatrix,phishingLabel,finalFakeFeatures,fakeURLphishingLabel,technique)

#
#*****************************************************************************************************************

# -------------------------- Technique 3: Selection using K-means clustering ------------------------------------
technique = 'kMeans'
# Apply K-means clustering algorithm on the fakeURLFeatureMatrix. Select k = 250. Then for each
# obatined centroid, calculate Distances and take 18 farthest points
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=250,random_state=10).fit(fakeURLFeatureMatrix)
kMeansFakeMatrix = []
for everyCenter in kmeans.cluster_centers_:
    #Calculate the distances from each point in feature matrix and sort it
    indexes = getDistancesFromCentre(everyCenter,fakeURLFeatureMatrix,20)
    for everyIndex in indexes:
        kMeansFakeMatrix.append(fakeURLFeatureMatrix[everyIndex])

kMeansFakeLabels = fakeURLphishingLabel[0:len(kMeansFakeMatrix)]

decisionTree7 = MyDecisionTreeClassifier()
decisionTree7.decisionTreeNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                         technique)

decisionTree3 = MyDecisionTreeClassifier()
decisionTree3.decisionTreebSMOTE2(featureMatrix, phishingLabel,kMeansFakeMatrix, kMeansFakeLabels, technique)

decisionTree4 = MyDecisionTreeClassifier()
decisionTree4.decisionTreeSVM_SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels,
                                    technique)

decisionTree5 = MyDecisionTreeClassifier()
decisionTree5.decisionTreeADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

decisionTree6 = MyDecisionTreeClassifier()
decisionTree6.decisionTreeRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

decisionTree1 = MyDecisionTreeClassifier()
decisionTree1.decisionTreeSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

decisionTree2 = MyDecisionTreeClassifier()
decisionTree2.decisionTreebSMOTE1(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

rf1 = MyRandomForestClassifier()
rf1.randomForestNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

rf2 = MyRandomForestClassifier()
rf2.randomForestRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

rf3 = MyRandomForestClassifier()
rf3.randomForestSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

rf4 = MyRandomForestClassifier()
rf4.randomForestADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

rf5 = MyRandomForestClassifier()
rf5.randomForestSVM_Smote(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

rf6 = MyRandomForestClassifier()
rf6.randomForestb2SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

rf7 = MyRandomForestClassifier()
rf7.randomForestb1SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

ada1 = MyAdaBoostClassifier()
ada1.adaBoostNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

ada2 = MyAdaBoostClassifier()
ada2.adaBoostADASYN(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

ada3 = MyAdaBoostClassifier()
ada3.adaBoostRMR(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

ada4 = MyAdaBoostClassifier()
ada4.adaBoostSVMSmote(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

ada5 = MyAdaBoostClassifier()
ada5.adaBoostb2SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

ada6 = MyAdaBoostClassifier()
ada6.adaBoostb1SMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

ada7 = MyAdaBoostClassifier()
ada7.adaBoostSMOTE(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

# svm1 = MySupportVector()
# svm1.supportVectorNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)
#
# svm2 = MySupportVector()
# svm2.supportVectorNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)
#
# svm3 = MySupportVector()
# svm3.supportVectorNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)
#
# svm4 = MySupportVector()
# svm4.supportVectorNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)
#
# svm5 = MySupportVector()
# svm5.supportVectorNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)
#
# svm6 = MySupportVector()
# svm6.supportVectorNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)
#
# svm7 = MySupportVector()
# svm7.supportVectorNoOversampling(featureMatrix, phishingLabel, kMeansFakeMatrix, kMeansFakeLabels, technique)

# Computer the best clusters with silhouette analysis
#cluster_range = [10]
#max_score = -999
# best_clusterSize = 0
# count = 0
# kmFptr = open(dir_name+'/Kmeansanalysis.txt','a+')
# kmFptr.flush()





# for eachRange in cluster_range:
#     clusterer = KMeans(n_clusters=eachRange, random_state=10)
#     clusterer_labels = clusterer.fit_predict(fakeURLFeatureMatrix)
#     silhouette_avg = silhouette_score(fakeURLFeatureMatrix,clusterer_labels)
#     if silhouette_avg > max_score:
#         kmFptr.write('Silhouette Avg is: ' + str(silhouette_avg))
#         kmFptr.flush()
#         if count > 0:
#             max_score = silhouette_avg
#             best_clusterSize = eachRange
#             count+=1
#             kmFptr.write('Best score updated and is: ' + str(max_score))
#             kmFptr.flush()
#             kmFptr.write('Best no of clusters are: ' + str(best_clusterSize))
#             kmFptr.flush()






# ***************************************************************************************************************
