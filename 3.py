import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv('breast-cancer.csv')

label_encoder = preprocessing.LabelEncoder()

data['target']=label_encoder.fit_transform(data['diagnosis'])

data.drop('diagnosis',axis=1,inplace=True)

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier()

model = classifier.fit(x_train, y_train)

y_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

print(cm) 
print(ac)