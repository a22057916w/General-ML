import os
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
absolutepath = os.path.abspath(__file__)
fileDirectory = os.path.dirname(absolutepath)

train = pd.read_csv(os.path.join(fileDirectory, 'train.csv'))
test = pd.read_csv(os.path.join(fileDirectory, 'test.csv'))
submission = pd.read_csv(os.path.join(fileDirectory, 'sample_submission.csv'))

X_train, X_val, y_train, y_val = train_test_split(train.drop('label',axis=1), train['label'], test_size=0.1, random_state=33)

xgb_model = XGBClassifier(objective='multi:softprob',
                      num_class= 10)
xgb_model.fit(X_train, y_train)

preds = xgb_model.predict(test).astype(int)
save = pd.DataFrame({'ImageId':submission.ImageId,'label':preds})
save.to_csv('submission.csv',index=False)
