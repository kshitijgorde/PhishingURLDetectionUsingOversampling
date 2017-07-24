#Create a CSV File denoted relevant String features
import socket
import re
import csv
import os
import scipy.stats
class FeaturesCSV():
    'This class generates a csv File with relevant String features'


    def validateIPAddress(self,URL):
        isValidIp = '0'
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)
        host = URL[hostStart+2:hostEnd]
        #Pattern match the Host part for IP address or Hexadecimal places
        try:
            socket.inet_aton(host)  #handles hexadecimal encoded IP address as well....
            #legal..return flag should be True
            isValidIp = '1'
        except:
            isValidIp = '0'

        return isValidIp

    def isLongURL(self,URL):
        'Consult for Ternary Values'
        isLongURL = '0'
        if len(URL) < 54:
            isLongURL = '0'
        elif len(URL) >= 54 and len(URL) <=75:
            isLongURL = '1'
        else:
            isLongURL = '2'
        return isLongURL

    def preSuffixInURL(self,URL):
        isPreSuffix = '0'
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)
        host = URL[hostStart + 2:hostEnd]
        count = str(host).count('-')
        if count > 0:
            isPreSuffix = '1'
        return isPreSuffix

    def subDomain(self,URL):
        'check if Ternary'
        isMultipleDomains = '0'
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)
        host = URL[hostStart + 2:hostEnd]
        #including www., if the dots are greater than 3, then High!
        count = str(host).count('.')
        if count < 3:
            isMultipleDomains = '0'
        elif count == 3:
            isMultipleDomains = '1'
        else:
            isMultipleDomains = '2'

        return isMultipleDomains

    def checkSymbol(self,URL):
        isSymbol = '0'
        #check if @ in host part
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)

        host = URL[hostStart + 2:hostEnd]
        if str(host).find("@") > 0:
            isSymbol = '1'
        return isSymbol

    #Check for HTTPS feature. If included without checking issuer, there will be high false positives

    def topLevelDomainCount(self,URL):
        'counts the occurences of top level domains by matching regular expressions'
        topLevelDomain = '0'
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)

        path = URL[hostEnd+1:]
        m = re.compile(r'\.([^.\n\s]*)$', re.M)
        f = re.findall(m, path)
        if len(f) > 0:
            topLevelDomain = '1'

        return topLevelDomain

    def suspicousWords(self,URL):
        'Counts certain suspicious words....'
        haveSuspicious = '0'
        suspicousDatabase = ["confirm","account","secure","ebayisapi","webscr","login","signin","submit","update","logon","wp","cmd","admin"]
        count=0
        for everySuspiciousKeyword in suspicousDatabase:
            if everySuspiciousKeyword in URL:
                count+=1
        if count>1:
            haveSuspicious = '1'
        return haveSuspicious


    def countPunctuation(self,URL):
        'Counts certain punctuation marks'
        punctuationFeature = '0'
        blacklistedPunctuations = ['!','#','$','*',';',':','\'']
        count = 0
        for everPunctuation in blacklistedPunctuations:
            if everPunctuation in URL:
                count+=1
        if count > 1:
            punctuationFeature = '1'

        return punctuationFeature

    def digitsInDomain(self,URL):
        isDigits = '0'
        try:
            hostStart = URL.index("//")
            hostEnd = URL.index("/", hostStart + 2)
        except:
            hostEnd = len(URL)
        try:
            host = URL[hostStart + 2:hostEnd]
            numbers = re.search(r'\d+', host).group()
        except:
            #no numbers found
            numbers = 0
            isDigits = '0'

        if numbers > 0:
            isDigits = '1'
        return isDigits


    def getCharacterFrequency(self,URL):
        import collections
        freq = collections.Counter(URL)
        freqSorted = sorted(freq.items())
        freqList = []
        for i in range(0, 26):
            freqList.append(0)
        for key, value in freqSorted:
            if key.isalpha():
                #check for
                freqList[ord(key.lower()) - 97] = int(value)
        return freqList

    def getEntropy(self,URL):
        freqList = self.getCharacterFrequency(URL)
        entropy = scipy.stats.entropy(freqList)
        return entropy

    def getKLDivergence(self,URL):
        freqEnglish = [8.12, 1.49, 2.71, 4.32, 12.02, 2.30, 2.03, 5.92, 7.31, 0.10, 0.69, 3.98, 2.61, 6.95, 7.68, 1.82, 0.11,
             6.02, 6.28, 9.10, 2.98, 1.11, 2.09, 0.17, 2.11, 0.07]
        freqList = self.getCharacterFrequency(URL)
        kld = scipy.stats.entropy(freqList,freqEnglish)
        return kld


    def createCSVFile(self,columns,originalHeader,threshold,fileName):
        'Creates a CSV File denoting features of the URL'
        dir_name = os.path.dirname(os.path.realpath(__file__))
        with open(dir_name+'/'+fileName, 'wb') as featureCSVFile:
            w = csv.writer(featureCSVFile)
            w.writerow(["URL","IP", "LongURL", "PreSuffix","SubDomain","@Symbol","TLDInPath","SuspiciousWords","PunctuationSymbols","DigitsInDomain","Entropy","KLDivergence","Phishy"])
            count = 0
            for everyURL in columns[originalHeader[0]]:
                features = []
                features.append(everyURL)
                features.append(self.validateIPAddress(everyURL))
                features.append(self.isLongURL(everyURL))
                features.append(self.preSuffixInURL(everyURL))
                features.append(self.subDomain(everyURL))
                features.append(self.checkSymbol(everyURL))
                features.append(self.topLevelDomainCount(everyURL))
                features.append(self.suspicousWords(everyURL))
                features.append(self.countPunctuation(everyURL))
                features.append(self.digitsInDomain(everyURL))
                features.append(self.getEntropy(everyURL))
                features.append(self.getKLDivergence(everyURL))

                #print columns[originalHeader[0]][count]
                if int(columns[originalHeader[1]][count]) >= threshold:
                    #then phishy
                    count+=1
                    #print 'Phishy ratio'
                    features.append("1")
                else:
                    features.append("0")
                    count+=1
                #print features
                #write these features to the csv File
                w.writerow(features)

    def normalized(self,lst):
        s = sum(lst)
        return map(lambda x: float(x) / s, lst)


# a_normalized =obj.normalized(a)
# print 'done'
# b_normalized = obj.normalized(b)
# print a_normalized
# print b_normalized
# print "%.2f" % scipy.stats.entropy(a_normalized) #low entropy signifies meaningless string. Normalizes internally
# print scipy.stats.entropy(a,b) # Calculates K-L divergence. Normalizes internally