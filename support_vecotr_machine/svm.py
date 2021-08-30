import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

iris = datasets.load_iris()
for key, value in iris.items():
    try:
        print(key, value.shape)
    except:
        print(key)


x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
#print(x)

y = pd.DataFrame(iris['target'], columns=['target_names'])
#print(y)

iris_data = pd.concat([x, y], axis=1)
print(iris_data)
