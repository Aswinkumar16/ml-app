from .packages import *

def load_data(file_input):
        df = pd.read_csv(file_input,header = 0)
        
        df.drop('id',axis=1,inplace=True)
        df.drop('Unnamed: 32',axis=1,inplace=True)
        df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
        y = df['diagnosis']
        
        X = df.drop('diagnosis', axis=1)

        #Feature Selection
        drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']
        X1 = X.drop(drop_list1,axis = 1 )
        data = train_test_split(X1, y, test_size=0.3, random_state=42)

        return X1,data
