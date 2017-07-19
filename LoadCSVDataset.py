import os
from collections import defaultdict
import csv
import csv
class LoadCSVDataset:
    'This class loads the phishing URL dataset and stores it locally for processing'
    def cleanDataset(self,columns,header):
        'This method receives the Dataset and deletes the URLs for which reported/total cases is N/A'
        for (k,v) in columns.items():
            if k == 'Positive':
                indexes = [i for i, x in enumerate(v) if x  == 'N/A'] #get indexes for N/A values
                for i in sorted(indexes,reverse=True):
                    # #delete the entire row
                    del columns[header[0]][i]
                    del columns[header[1]][i]
                    del columns[header[2]][i]
        return columns



    def loadDataset(self):
        'Method for loading the CSV File'
        dir_name = os.path.dirname(os.path.realpath(__file__))
        columns = defaultdict(list)
        csvFile = open(dir_name+'/Outresponse.csv','a+')
        #DataSamples = open('randomDataSamples.txt','r').readlines()

        header = []
        reader = csv.reader(csvFile)
        header = next(reader)

        # for everyRow in DataSamples:
        #     everyRow = everyRow.replace('\n','').replace('\r','')
        #     fields = [everyRow,10,64]
        #     writer = csv.writer(csvFile)
        #     writer.writerow(fields)


        count=0
        with open (dir_name+'/Outresponse.csv') as file:
            reader = csv.DictReader(file)
            for row in reader:
                for(k,v) in row.items():
                    columns[k].append(v)


        #print len(columns[header[0]])
        cleanedColumns = self.cleanDataset(columns,header)
        return cleanedColumns,header
