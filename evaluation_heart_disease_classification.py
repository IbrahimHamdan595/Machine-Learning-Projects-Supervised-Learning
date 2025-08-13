import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_val_score


heart_disease = pd.read_csv("../data/heart-disease.csv")

X = heart_disease.drop('target', axis=1)
y = heart_disease['target']

X_train, X_test, y_train, y_test = train_test_split(X,y)

models = {
    "LinearSVC": LinearSVC(),
    "SVC": SVC(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier()
}

results = {}

np.random.seed(42)

for model_name, model in models.items():
    model.fit(X_train, y_train)
    results[model_name] = model.score(X_test, y_test)

print(results)

result_df = pd.DataFrame(results.values(),
                         results.keys(),
                         columns=["accuracy"] 
                         )

result_df.plot.bar()
plt.show()

  
param_dist = {
      'n_estimators': [100, 200, 300],
      'max_features': ['auto', 'sqrt', 'log2'],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4],
      'bootstrap': [True, False]
  }

rf = RandomForestClassifier()
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=10, cv=3)
random_search.fit(X_train, y_train)

print(random_search.best_params_)
print(random_search.best_score_)

# Classification model evaluation

clf = RandomForestClassifier(**random_search.best_params_)
clf.fit(X_train, y_train)

y_preds = clf.predict(X_test)

confusion_matrix(y_test, y_preds)

def plot_conf_mat(y_test, y_preds):
  
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True, #Annotate the boxes
                     cbar=False)
    plt.xlabel('True label')
    plt.ylabel('Predicted label')

    bottom, top= ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

plot_conf_mat(y_test, y_preds)
plt.show()

print(classification_report(y_test, y_preds))

precision_score(y_test, y_preds)
recall_score(y_test, y_preds)
f1_score(y_test, y_preds)

RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.show()
cross_val_score(clf, X, y, scoring='accuracy', cv= 5)

cross_val_acc = np.mean(cross_val_score(clf, X, y, scoring='accuracy', cv= 5))
cross_val_precision = np.mean(cross_val_score(clf, X, y, scoring='precision', cv= 5))
cross_val_recall = np.mean(cross_val_score(clf, X, y, scoring='recall', cv= 5))
cross_val_f1 = np.mean(cross_val_score(clf, X, y, scoring='f1', cv= 5))
