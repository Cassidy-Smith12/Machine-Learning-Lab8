import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
         'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
df = pd.read_csv('breast-cancer-wisconsin.data.csv', names=names)

df = df.replace('?', np.nan)
df = df.dropna(axis=0)

X = np.array(df.iloc[:, 1:10], dtype=float)
y = np.array(df.iloc[:, 10], dtype=int)

X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(principalComponents, y, test_size=0.25, random_state=42)

modelSVC = SVC(kernel='linear')
modelSVC.fit(X_train, y_train)

y_pred = modelSVC.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Plotting the decision boundary
w = modelSVC.coef_[0]
b = modelSVC.intercept_[0]
a = -w[0] / w[1]
xx = np.linspace(min(principalComponents[:, 0]), max(principalComponents[:, 0]), 100)
yy = a * xx - (b / w[1])

plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y)
plt.plot(xx, yy, 'k-')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.ylim(-2, 4)
plt.title("LSVM with PCA")
plt.show()
