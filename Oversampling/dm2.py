from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
#from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv('thoracic.csv', delimiter=',')

feature = df.drop('Class', axis=1)
kelas = df['Class']

pca = PCA(n_components=2)
X_vis = pca.fit_transform(feature)

print('Original dataset shape {}'.format(Counter(kelas)))
print
print kelas
print

sm = SMOTE(kind='borderline1',random_state=42)

df_res, kelas_res= sm.fit_sample(feature, kelas)

X_res_vis = pca.transform(df_res)
print('Resampled dataset shape {}'.format(Counter(kelas_res)))
print
x_train_res, x_val_res, y_train_res, y_val_res = train_test_split(df_res,
                                                    kelas_res,
                                                    test_size = .3,
                                                    random_state=12)

clf_rf = RandomForestClassifier(n_estimators=25, random_state=42)
clf_rf.fit(x_train_res, y_train_res)
print "Accuracy : ", clf_rf.score(x_val_res, y_val_res)

f, (ax1, ax2) = plt.subplots(1, 2)
c1 = ax1.scatter(X_vis[kelas == 0, 0], X_vis[kelas == 0, 1], label="Class #1", alpha=0.5)
c2 = ax1.scatter(X_vis[kelas == 1, 0], X_vis[kelas == 1, 1], label="Class #2", alpha=0.5)
ax1.set_title('Original set')

ax2.scatter(X_res_vis[kelas_res == 0, 0], X_res_vis[kelas_res == 0, 1], label="Class #1", alpha=0.5)
ax2.scatter(X_res_vis[kelas_res == 1, 0], X_res_vis[kelas_res == 1, 1], label="Class #2", alpha=0.5)
ax2.set_title('SMOTE')

# make nice plotting
for ax in (ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])

plt.figlegend((c1, c2), ('Class #1', 'Class #2'), loc='lower center', ncol=2, labelspacing=0)
plt.tight_layout(pad=3)
plt.show()
