import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, RocCurveDisplay


# Load Data

train = pd.read_csv('../data/titanic survivor/train.csv')
test = pd.read_csv('../data/titanic survivor/test.csv')


# EDA

print(train.shape, test.shape)
print(train.info())
print(train.isna().sum())
print(train['Survived'].value_counts(normalize=True))

sns.countplot(x='Survived', data=train)
plt.title('Survival Count')
plt.show()

sns.countplot(x='Sex', hue='Survived', data=train)
plt.title('Survival by Gender')
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=train)
plt.title('Survival by Ticket Class')
plt.show()

train['AgeGroup'] = pd.cut(train['Age'], bins=[0,12,18,35,50,80], 
                           labels=['Child','Teen','Young Adult','Middle Aged','Senior'])

sns.countplot(x='AgeGroup', hue='Survived', data=train)
plt.title('Survival by Age Group')
plt.show()

# Feature Engineering

for df in [train, test]:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major',
                                       'Rev','Sir','Jonkheer','Dona'],'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')


# Preprocessing Pipelines

num_features = ['Age', 'Fare', 'FamilySize']
cat_features = ['Sex', 'Embarked', 'Title']

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('oneHot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])


# Model

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

X = train.drop(columns=["Survived","PassengerId","Name","Ticket","Cabin","AgeGroup"])
y = train["Survived"]

# Cross-validation 
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"Cross-val Accuracy: {scores.mean():.4f}")

# Train/Test split for evaluation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)

# Metrics
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_pred):.4f}")
print(f"Recall: {recall_score(y_val, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# ROC Curve
RocCurveDisplay.from_estimator(pipeline, X_val, y_val)
plt.show()


# Final Prediction on Test.csv

X_test_final = test.drop(columns=["PassengerId","Name","Ticket","Cabin"])
final_preds = pipeline.predict(X_test_final)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": final_preds
})
submission.to_csv("submission.csv", index=False)
print("submission.csv created!")
