import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle


# reading dataset with pandas lib
df=pd.read_csv("outbreak_detect.csv")

# dropping null values in the dataset
df=df.dropna()


#droping colums not requried
df=df.drop('Positive',axis=1)
df=df.drop('pf',axis=1)


#data processing
from sklearn import preprocessing

#labelencodingg
LE=preprocessing.LabelEncoder()

#fitting it to our dataset

df.Outbreak=LE.fit_transform(df.Outbreak)
df.head()

# loading  the dataset
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1:].values

# splitting the dataset
# spliting he data set into train and test dataset
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

sst=StandardScaler()

X_train = sst.fit_transform(X_train)
X_test = sst.transform(X_test)


clf = LogisticRegression()
clf.fit(X_train, Y_train)


y_pred = clf.predict(X_test)
print(y_pred)

pickle.dump(clf,open('model1.pkl','wb')) #serializing the model by creating "model.pkl
model1 = pickle.load(open('model1.pkl','rb'))  #deserializing reading the file
print("success loaded")
