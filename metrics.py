from sklearn import metrics
import pandas as pd

raw_data = pd.read_csv('predict.csv')

print(metrics.accuracy_score(raw_data['actual'].tolist(), raw_data['predicted'].tolist()))
