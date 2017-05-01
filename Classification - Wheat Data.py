# Lab KNN
# Assignment5


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


def plotDecisionBoundary(model, X, y):
  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.6
  resolution = 0.0025
  colors = ['royalblue','forestgreen','ghostwhite']

  # Calculate the boundaris
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix
  # are the predictions of the class at at said location
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  cs = plt.contourf(xx, yy, Z, cmap=plt.cm.terrain)

  # Plot the test original points as well...
  for label in range(len(np.unique(y))):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], label=str(label), alpha=0.8)

  p = model.get_params()
  plt.axis('tight')
  plt.title('K = ' + str(p['n_neighbors']))



# Load up the dataset into a variable called X
X = pd.read_csv('/Users/.../wheat.data', index_col = 0) 
X.head
X.dtypes


# Set "wheat_type" to be y and drop "wheat_type" column from X
y = X.wheat_type
X = X.drop(labels = ['wheat_type'], axis = 1)


# Set "wheat_type" to be ordinal
y = y.astype('category', ordered = True).cat.codes


# Fill each row's nans with the mean of the feature
X.isnull().sum() # Check the number of nans in the dataframe
X.compactness.fillna(X.compactness.mean(), inplace = True)
X.width.fillna(X.width.mean(), inplace = True)
X.groove.fillna(X.groove.mean(), inplace = True)
X.isnull().sum()


# Split X into training and testing data sets 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)


# Create an instance of SKLearn's Normalizer class and then train it
from sklearn import preprocessing
normalizer = preprocessing.Normalizer()
normalizer = normalizer.fit(X_train)


# With your trained pre-processor, transform both your training and testing data.
T_X_train = normalizer.transform(X_train)
T_X_test = normalizer.transform(X_test)



# PCA model
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(T_X_train)
PCA_T_X_train = pca.transform(T_X_train)
PCA_T_X_test = pca.transform(T_X_test)


# Create and train a KNeighborsClassifier. With K=9 neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(PCA_T_X_train, y_train)


# Plot out the result
plotDecisionBoundary(knn, PCA_T_X_train, y_train)


# Accuracy score of your test data/labels, computed by
knn.score(PCA_T_X_test, y_test)


