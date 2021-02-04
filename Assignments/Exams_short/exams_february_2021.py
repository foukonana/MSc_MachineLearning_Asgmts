# =============================================================================
# MACHINE LEARNING
# EXAMS - FEBRUARY 2021
# PROGRAMMING PROJECT
# Complete the missing code by implementing the necessary commands.
# =============================================================================

# Libraries to use
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 

bc = load_breast_cancer(as_frame=True)
bc_df = bc['frame']

# split the data into train-test (40% of data as test), with stratify enabled
X = bc_df.drop(columns='target')
y= bc_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=67, stratify=y)

# function to print the metrics 
def print_metric_scores(y_act, y_pred, set_name):
    print(f'The metrics for the {set_name} set are:')
    print(f'Accuracy score: {accuracy_score(y_act, y_pred)*100:0.2f}%')
    print(f'Precision score: {precision_score(y_act, y_pred)*100:0.2f}%')
    print(f'Recall score: {recall_score(y_act, y_pred)*100:0.2f}%')
    print(f'F1 score: {f1_score(y_act, y_pred)*100:0.2f}%')

    df = pd.DataFrame(data={'metric': ['accuracy', 'precision', 'recall', 'f1'],
                         'score': [accuracy_score(y_act, y_pred)*100, 
                                   precision_score(y_act, y_pred)*100,
                                   recall_score(y_act, y_pred)*100,
                                   f1_score(y_act, y_pred)*100],
                         'set': set_name})
    return df

# scale the data as KNN is sensitive to the scale of the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# define the knn classifier and train it and get prediction for train and test data
clf = KNeighborsClassifier(n_neighbors=15, 
                           metric='minkowski', 
                           p=2,
                           n_jobs=-1)
clf.fit(X_train_scaled, y_train)
y_pred_train = clf.predict(X_train_scaled)
y_pred_test = clf.predict(X_test_scaled)

train_res = print_metric_scores(y_train, y_pred_train, 'train')
test_res = print_metric_scores(y_test, y_pred_test, 'test')

res_df = train_res.append(test_res)
sns.barplot(x='metric', y='score', hue='set', data=res_df, palette='ocean')
plt.title('Resulting metrics of train and test set')
plt.show()
