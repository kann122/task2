"""
Original file is located at
    https://colab.research.google.com/drive/1cSebZqnzt2PrfSvLg9asQCNgpoGUJuBa

IMPORTING LIBRARIES
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

"""DOWNLOADING DATASETS"""

flower_data=sns.load_dataset("iris")
flower_data.head()

flower_data.shape

flower_data.info()

flower_data.isnull().sum()

flower_data['species'],categories=pd.factorize(flower_data['species'])
flower_data.head()

flower_data.describe()

"""Hence its time to visualize the data"""

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(flower_data.petal_length,flower_data.petal_width,flower_data.species)
ax.set_xlabel('PetalLengthCm')
ax.set_ylabel('PetalWidthCm')
ax.set_zlabel('species')
plt.title('3D Scatter Plot Example')
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(flower_data.sepal_length,flower_data.sepal_width,flower_data.species)
ax.set_xlabel('SepalLengthCm')
ax.set_ylabel('SepalWidthCm')
ax.set_zlabel('species')
plt.title('3D Scatter Plot Example')
plt.show()

"""Thus 3-D gives the glimpse of species of iris flower is more inclined towards the variables petal length and petal width"""

sns.scatterplot(data=flower_data,x='petal_length',y='petal_width',hue='species')

sns.scatterplot(data=flower_data,x='sepal_length',y='sepal_width',hue='species')

"""Applying Elbow Technique"""

K_rng=range(1,10)
sse=[]
for k in K_rng:
  km=KMeans(n_clusters=k)
  km.fit(flower_data[['petal_length','petal_width']])
  sse.append(km.inertia_)

sse

plt.xlabel('K_rng')
plt.ylabel('Sum of squared error')
plt.plot(K_rng,sse)

"""Applying KMean Algorithm"""

km=KMeans(n_clusters=3,random_state=0)
y_predicted=km.fit_predict(flower_data[['petal_length','petal_width']])
y_predicted

flower_data['cluster']=y_predicted
flower_data.head(150)

"""Accuracy Measure"""

from sklearn.metrics import confusion_matrix
Cm=confusion_matrix(flower_data.species,flower_data.cluster)
Cm

true_labels=flower_data.species
predicted_labels=flower_data.cluster

Cm=confusion_matrix(true_labels,predicted_labels)
class_labels = ['Setosa', 'Versicolor', 'Virginica']

#plot confusion matrix
plt.imshow(Cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)

#Fill matrix with values
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j, i, str(Cm[i][j]),ha='center',va='center',color='white')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
