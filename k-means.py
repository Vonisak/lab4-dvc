import pandas as pd
from sklearn.cluster import KMeans

raw_data = pd.read_csv('iris.csv', index_col=4)

raw_data1 = pd.read_csv('iris.csv')
raw_data1.loc[raw_data1['CLASS'] == 'Iris-setosa'] = 0
raw_data1.loc[raw_data1['CLASS'] == 'Iris-versicolor'] = 1
raw_data1.loc[raw_data1['CLASS'] == 'Iris-virginica'] = 2

model = KMeans(n_clusters=3, random_state=2)
model.fit(raw_data)

predict_file = open('predict.csv', 'w')
predict_file.write('predicted,actual\n')

for i, j in zip(model.labels_, raw_data1['CLASS'].tolist()):
    predict_file.write(str(i) + ',' + str(j) + '\n')
