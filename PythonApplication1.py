import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sdv.tabular import GaussianCopula
from sdv.evaluation import evaluate


items = 50
customer_id = np.random.randint(0, 50, items)

df = pd.DataFrame({"customer_id":customer_id})
df['material'] = np.random.choice(["uo2","zro", "b4c"], size=items, p=[.40,.40,.20])
print(df)
features, output = make_classification(n_samples = 1000,
                                       n_features = 10,
                                       n_informative = 5,
                                       n_redundant = 1,
                                       n_classes = 5,
                                       weights = [.2, .2,.2, .2, .2],
                                       random_state=0
                                      )
print(pd.DataFrame(features).head())
 
data = pd.read_csv('test1.csv')
print(data.head())
model = GaussianCopula()
model.fit(data)
sample = model.sample(10000)
print('synthetic data',sample.head())
evaluate(sample, data, metrics=['CSTest', 'KSTest'], aggregate=False)
print('evaluation first try',evaluate(sample, data, metrics=['CSTest'], aggregate=False))
print('evaluation data ',evaluate(sample, data))
sample.to_csv('raw_data1.csv', index=False)