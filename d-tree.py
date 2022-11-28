import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('iris.csv')

X = raw_data.drop('CLASS', axis=1)
y = raw_data['CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=2)

clf = DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X)

predict_file = open('predict.csv', 'w')
predict_file.write('predicted,actual\n')

for predicted, actual in zip(y_pred, y.tolist()):
    x = 0
    y = 0

    if predicted == 'Iris-setosa':
        x = 0
    elif predicted == 'Iris-versicolor':
        x = 1
    else:
        x = 2

    if actual == 'Iris-setosa':
        y = 0
    elif actual == 'Iris-versicolor':
        y = 1
    else:
        y = 2

    predict_file.write(str(x)+','+str(y)+'\n')
