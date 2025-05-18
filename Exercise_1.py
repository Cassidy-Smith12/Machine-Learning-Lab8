import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('speedLimits.csv')
X = np.array(df[['Speed']])
y = np.array(df['Ticket'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for i in range(len(y)):
    if y[i] == 'NT':
        y[i] = 'g'
    else:
        y[i] = 'r'

plt.scatter(df['Speed'], df['Ticket'], alpha=0.7, c=y)
plt.title("Speed vs Ticket")
plt.xlabel('Speed')
plt.ylabel('Ticket')
plt.show()

#Define different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
classifiers = {}

#Train SVM models with different kernels
for kernel in kernels:
    clf = SVC(kernel=kernel, C=1.0)
    clf.fit(X_train_scaled, y_train)
    classifiers[kernel] = clf

#Evaluate the models
accuracies = {}
for kernel, clf in classifiers.items():
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[kernel] = accuracy
    print(f"Accuracy with {kernel} kernel: {accuracy:.4f}")

#Find the optimal kernel
optimal_kernel = max(accuracies, key=accuracies.get)
print(f"The optimal kernel is: {optimal_kernel} with accuracy {accuracies[optimal_kernel]:.4f}")
