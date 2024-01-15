import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

data=pd.read_csv('iris.csv')

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

x=StandardScaler().fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

clf = SVC(kernel='rbf')

model = clf.fit(x_train, y_train)

prediction = model.predict(x_test)

accuracy = accuracy_score(prediction, y_test)

print(accuracy)
