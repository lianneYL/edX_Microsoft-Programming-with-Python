import random, math
import pandas as pd
import numpy as np
import scipy.io

from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier


# If you'd like to try this lab with PCA instead of Isomap,
# as the dimensionality reduction technique:   
Test_PCA = True    

def plotDecisionBoundary(model, X, y):
    print ("Plotting...")
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    padding = 0.1 
    resolution = 0.1
    
    # (2 for benign, 4 for malignant)
    colors = {2: 'royalblue', 4: 'lightsalmon'}
    
    # Calculate the boundaries 
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = x_max - x_min 
    y_range = y_max - y_min 
    x_min -= x_range * padding
    y_min -= y_range * padding
    x_max += x_range * padding
    y_max += y_range * padding
    
    # Create a 2D Grid Matrix. The values stored in the matrix
    # are the predictions of the class at said location
    import numpy as np
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # What class does the classifier say?
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour map
    plt.contourf(xx, yy, Z, cmap = plt.cm.seismic)
    plt.axis('tight')
    
    # Plot your testing points as well
    for label in np.unique(y):
        indices = np.where(y == label)
        plt.scatter(X[indices, 0], X[indices, 1], c = colors[label], alpha = 0.8)
        
    p = model.get_params()
    plt.title('K = ' + str(p['n_neighbors']))
    plt.show()
    


# TODO: Load in the dataset, identify nans, and set proper headers.
# Be sure to verify the rows line up by looking at the file in a text editor.
cancerDF = pd.read_csv('/Users/.../breast-cancer-wisconsin.data',
                       names = ['Sample', 'Thickness', 'Size', 'Shape', 'Adhesion', 'Epithelial', 'Nuclei', 'Chromatin', 'Nucleoli', 'Mitoses', 'Status'])


 
# TODO: Copy out the status column into a slice, then drop it from the main
# dataframe. You can also drop the sample column, since that doesn't provide
# us with any machine learning power.
Status = cancerDF.Status
cancerDF = cancerDF.drop('Status', axis = 1)
cancerDF = cancerDF.drop('Sample', axis = 1)



# TODO: With the labels safely extracted from the dataset, replace any nan values
# with the mean feature / column value
cancerDF.Nuclei.unique() # Check if columns have ?
cancerDF.Nuclei = cancerDF.Nuclei.replace('?', np.NaN)
cancerDF.Nuclei = pd.to_numeric(cancerDF.Nuclei)
cancerDF.Nuclei = cancerDF.Nuclei.fillna(cancerDF.Nuclei.mean())


# TODO: Do train_test_split. Use the same variable names as on the EdX platform in
# the reading material, but set the random_state=7 for reproduceability, and keep
# the test_size at 0.5 (50%).
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancerDF, Status, train_size = 0.5, random_state = 7)



# TODO: Experiment with the basic SKLearn preprocessing scalers. We know that
# the features consist of different units mixed in together, so it might be
# reasonable to assume feature scaling is necessary. Print out a description
# of the dataset, post transformation.
from sklearn import preprocessing
scaling = preprocessing.MaxAbsScaler()
scaling.fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)




# PCA and Isomap are your new best friends
model = None
if Test_PCA:
  print ("Computing 2D Principle Components")
  
  # TODO: Implement PCA here. save your model into the variable 'model'.
  # You should reduce down to two dimensions.
  from sklearn.decomposition import PCA
  model = PCA(n_components = 2)
  

else:
  print ("Computing 2D Isomap Manifold")
  
  # TODO: Implement Isomap here. save your model into the variable 'model'
  # Experiment with K values from 5-10.
  # You should reduce down to two dimensions.
  from sklearn import manifold
  model = manifold.Isomap(n_neighbors = 4, n_components =2)
  


# TODO: Train your model against data_train, then transform both
# data_train and data_test using your model. You can save the results right
# back into the variables themselves.
model.fit(X_train)
X_train = model.transform(X_train)
X_test = model.transform(X_test)



# TODO: Implement and train KNeighborsClassifier on your projected 2D
# training data here. You can use any K value from 1 - 15, so play around
# with it and see what results you can come up. Your goal is to find a
# good balance where you aren't too specific (low-K), nor are you too
# general (high-K). You should also experiment with how changing the weights
# parameter affects the results.
knn_model = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform')
knn_model.fit(X_train, y_train)



# INFO: Be sure to always keep the domain of the problem in mind! It's
# WAY more important to errantly classify a benign tumor as malignant,
# and have it removed, than to incorrectly leave a malignant tumor, believing
# it to be benign, and then having the patient progress in cancer. Since the UDF
# weights don't give you any class information, the only way to introduce this
# data into SKLearn's KNN Classifier is by "baking" it into your data. For
# example, randomly reducing the ratio of benign samples compared to malignant
# samples from the training set.


# TODO: Calculate + Print the accuracy of the testing set
knn_model.score(X_test, y_test)


plotDecisionBoundary(knn_model, X_test, y_test)



