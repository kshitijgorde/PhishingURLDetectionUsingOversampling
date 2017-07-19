from imblearn.under_sampling import AllKNN
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

class Resampling():

    def smoteOversampling(self,featureMatrix, Labels):
        print type(featureMatrix)
        print type(Labels)
        sm = SMOTE(kind='regular')
        #print type(featureMatrix[0][0])
        #print type(Labels[0])
        feature_Resampled, Labels_Resampled = sm.fit_sample(featureMatrix, Labels)
        #print type(feature_Resampled[0][0])
        #print type(Labels_Resampled[0])
        print "Resampling Done"
        return feature_Resampled,Labels_Resampled