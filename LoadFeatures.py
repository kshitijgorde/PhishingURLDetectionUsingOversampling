import csv
import os
import scipy
from collections import OrderedDict
from collections import defaultdict
class LoadFeatures():
    'This class contains methods to load and create feature matrix from the URLFeatures File'
    def loadFeatures(self):
        header = []
        UrlDict = OrderedDict()
        columns = defaultdict(list)
        dir_name = os.path.dirname(os.path.realpath(__file__))
        csvFile = open(dir_name+'/URLFeatures.csv', 'r')
        reader = csv.reader(csvFile)
        header = next(reader)
        with open(dir_name+'/URLFeatures.csv') as f:
            reader = csv.DictReader(f)  # read rows into a dictionary format
            for row in reader:  # read a row as {column1: value1, column2: value2,...}
                for (k, v) in row.items():
                    # go over each column name and value
                    columns[k].append(v)  # append the value into the appropriate list

        urlList = columns[header[0]]
        featureMatrix = []
        for i in range(0, len(urlList)):
            term_matrix = []
            for j in range(1, 12):
                term_matrix.append(float(columns[header[j]][i]))

            featureMatrix.append(term_matrix)

        #17115 duplicate
        for i in range(0,len(urlList)):
            UrlDict[urlList[i]] = featureMatrix[i]


        if len(UrlDict)!= len(urlList):
            print 'Duplicate values in URL found. Please clean dataset and re-ren'
            exit(111)


        phishingLabel = [int(x) for x in columns[header[12]]]
        return UrlDict,phishingLabel

    def loadPositiveFeatures(self):
        header = []
        columns = defaultdict(list)
        dir_name = os.path.dirname(os.path.realpath(__file__))
        csvFile = open(dir_name+'/URLFeatures.csv', 'r')
        reader = csv.reader(csvFile)
        header = next(reader)
        with open(dir_name+'/URLFeatures.csv') as f:
            reader = csv.DictReader(f)  # read rows into a dictionary format
            for row in reader:  # read a row as {column1: value1, column2: value2,...}
                for (k, v) in row.items():
                    # go over each column name and value
                    columns[k].append(v)  # append the value into the appropriate list

        urlList = []#columns[header[0]]
        phishingLabel = [int(x) for x in columns[header[12]]]

        for i in range(0,len(phishingLabel)):
            if phishingLabel[i] == '1' or phishingLabel[i]==1:
                urlList.append(columns[header[0]][i])

        return urlList
        #return featureMatrix,phishingLabel
