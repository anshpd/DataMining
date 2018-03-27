from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('thoracic.csv', delimiter=',')

feature = df.drop('Class', axis=1)
kelas = df['Class']
print('Original dataset shape {}'.format(Counter(kelas)))
print
print kelas
print

sm = SMOTE(random_state=42)

df_res, kelas_res= sm.fit_sample(feature, kelas)
print('Resampled dataset shape {}'.format(Counter(kelas_res)))
print
x_train_res, x_val_res, y_train_res, y_val_res = train_test_split(df_res,
                                                    kelas_res,
                                                    test_size = .3,
                                                    random_state=12)

clf_rf = RandomForestClassifier(n_estimators=25, random_state=42)
clf_rf.fit(x_train_res, y_train_res)
print "Accuracy : ", clf_rf.score(x_val_res, y_val_res)
