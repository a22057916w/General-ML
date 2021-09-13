import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# ---------------- load iris dataset ---------------------
print("------ Loading Iris Dataset")
# type(iris) return <class 'sklearn.utils.Bunch'>
iris = datasets.load_iris()
for key, value in iris.items():
    try:
        print(key, value.shape)
    except:
        print(key)


# type(iris['data']) return <class 'numpy.ndarray'>
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
"""
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                  5.1               3.5                1.4               0.2
1                  4.9               3.0                1.4               0.2
2                  4.7               3.2                1.3               0.2
3                  4.6               3.1                1.5               0.2
4                  5.0               3.6                1.4               0.2
..                 ...               ...                ...               ...
145                6.7               3.0                5.2               2.3
146                6.3               2.5                5.0               1.9
147                6.5               3.0                5.2               2.0
148                6.2               3.4                5.4               2.3
149                5.9               3.0                5.1               1.8
"""

y = pd.DataFrame(iris['target'], columns=['target'])
"""
     target_names
0               0
1               0
2               0
3               0
4               0
..            ...
145             2
146             2
147             2
148             2
149             2
"""

# type(iris_data) return <class 'pandas.core.frame.DataFrame'>
# axis=0是依row合併;axis=1則是依column合併
iris_data = pd.concat([x, y], axis=1)
"""
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target_names
0                  5.1               3.5                1.4               0.2             0
1                  4.9               3.0                1.4               0.2             0
2                  4.7               3.2                1.3               0.2             0
3                  4.6               3.1                1.5               0.2             0
4                  5.0               3.6                1.4               0.2             0
..                 ...               ...                ...               ...           ...
145                6.7               3.0                5.2               2.3             2
146                6.3               2.5                5.0               1.9             2
147                6.5               3.0                5.2               2.0             2
148                6.2               3.4                5.4               2.3             2
149                5.9               3.0                5.1               1.8             2
"""

iris_data = iris_data[["sepal length (cm)", "petal length (cm)", "target"]]
"""
     sepal length (cm)  petal length (cm)  target
0                  5.1                1.4       0
1                  4.9                1.4       0
2                  4.7                1.3       0
3                  4.6                1.5       0
4                  5.0                1.4       0
..                 ...                ...     ...
145                6.7                5.2       2
146                6.3                5.0       2
147                6.5                5.2       2
148                6.2                5.4       2
149                5.9                5.1       2

"""

#print(iris_data["target"].isin([0, 1])) 
"""
[150 rows x 3 columns]
0       True
1       True
2       True
3       True
4       True
       ...
145    False
146    False
147    False
148    False
149    False

"""

iris_data = iris_data[iris_data["target"].isin([0, 1])]
#print(iris_data)
"""    sepal length (cm)  petal length (cm)  target
0                 5.1                1.4       0
1                 4.9                1.4       0
2                 4.7                1.3       0
3                 4.6                1.5       0
4                 5.0                1.4       0
..                ...                ...     ...
95                5.7                4.2       1
96                5.7                4.2       1
97                6.2                4.3       1
98                5.1                3.0       1
99                5.7                4.1       1

[100 rows x 3 columns]
"""


"""
type(iris_data)                         --> <class 'pandas.core.frame.DataFrame'>
type(iris_data["target"])               --> <class 'pandas.core.series.Series'>
type(iris_data["target"].isin([0, 1]))  --> <class 'pandas.core.series.Series'>
"""

# 印頭3個
print(iris_data.head(3))
print("-------------------------------------------------")


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_data[['sepal length (cm)','petal length (cm)']], iris_data[['target']], test_size=0.3, random_state=0)

print(X_train, X_test)
print(y_train, y_test)
