import pickle
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Read the data

with open('processed_data.pickle', 'rb') as f:
    all_parts = pickle.load(f)
#print(f"Here the results: {all_parts}")


X = [np.array(run["Functional"][0:4]).flatten() for participant in all_parts for run in participant]
#y = [[run["Shape"], run["Condition"]] for participant in all_parts for run in participant]
y = [run["Shape"] for participant in all_parts for run in participant]

print('Here the X', len(X[0]), flush=True)
print("Here the y", y[0])

assert len(X) == len(y), "Something wrong, X and Y have to be the same lenght"
print(len(X))
print(len(y))
# Split the data into Train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_test)
print(y_train)

from sklearn.neighbors import KNeighborsClassifier
# Define the classifier.
print("Defining pipeline")
p0 = [("StandardScaler", StandardScaler()),
      ("PCA", PCA(n_components=.9)),
("SVC", SVC( kernel="linear", class_weight="balanced"))]

#NOther Pipeline
#p1 = [("StandardScaler", StandardScaler()),
#      ("PCA", PCA(n_components=.9)), ("SVC", SVC(kernel="linear", class_weight="balanced"))]


#("SVC", KNeighborsClassifier(3))]
#("SVC", SVC(kernel="linear", class_weight="balanced"))]

clf = Pipeline(p0)

print("Training the classifier")
clf.fit(X=X_train, y=y_train)

print("Predicting")
y_pred = clf.predict(X=X_test)

print(accuracy_score(y_test, y_pred))


