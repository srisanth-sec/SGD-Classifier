# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load Data.

2.Split Dataset into Training and Testing Sets

3.Train the Model Using Stochastic Gradient Descent (SGD).

4.Make Predictions and Evaluate Accuracy

5.Generate accuracy_score, confusion_matrix,classification_report

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Srisanth R
RegisterNumber:  212225240156
*/
import pandas as pd

from sklearn.datasets import load_iris

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

iris = load_iris()

df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

df['target'] = iris.target

print(df.head())

x = df.iloc[:, :-1]

y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/4, random_state = 42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(x_train, y_train)

y_pred = sgd_clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :",accuracy)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix : \n",cm)

report = classification_report(y_test, y_pred, target_names=iris.target_names)

print("\nClassification Report:\n", report)

```

## Output:
<img width="867" height="730" alt="Screenshot 2026-05-11 140755" src="https://github.com/user-attachments/assets/7cb7394c-a820-4da4-96b7-ff005ca3a209" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
