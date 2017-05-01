# Module 5
# Cluster - Wholesale Customer Data


import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans


PLOT_TYPE_TEXT = False    # If you'd like to see indices
PLOT_VECTORS = True       # If you'd like to see your original features in P.C.-Space


matplotlib.style.use('ggplot') 
c = ['red', 'green', 'blue', 'orange', 'yellow', 'brown']  
 

def drawVectors(transformed_features, components_, columns, plt):
    num_columns = len(columns)
    
        
    # Scale the PCA by the max value in the transformed set belonging to that component
    xvector = components_[0]*max(transformed_features[:,0])
    yvector = components_[1]*max(transformed_features[:,1])
    
    # Visualize projections
    
    # Sort each column by its length - *original columns, not the principal components
    import math
    important_features = {columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns)}
    important_features = sorted(zip(important_features.values(), important_features.keys()), reverse = True)
    print ('Projected Features by importance:\n', important_features)
    
    ax = plt.axes()
    
    for i in range(num_columns):
        plt.arrow(0, 0, xvector[i], yvector[i], color = 'b', width = 0.0005, head_width = 0.02, alpha = 0.75, zorder = 600000)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color = 'b', alpha = 0.75, zorder = 600000)
    return ax


def doKMeans(data, clusters=0):
    model = KMeans(n_clusters = clusters)
    model.fit(data)
    return model.cluster_centers_, model.labels_


# Import data
wholesale_df = pd.read_csv('/Users/.../wholesale customers data.csv')
wholesale_df.isnull().values.any()


# Drop 'Channel' and 'Region' columns
wholesale_df = wholesale_df.drop(labels = ['Channel', 'Region'], axis = 1)


# Take a look into the data
wholesale_df.describe()
# Plot histogram
wholesale_df.plot.hist()


# Remove top 5 and bottom 5 samples for each column:
drop = {}
for col in wholesale_df.columns:
  # Bottom 5
  sort = wholesale_df.sort_values(by = col, ascending = True)
  if len(sort) > 5: sort = sort[:5]
  for index in sort.index: drop[index] = True    
  # Top 5
  sort = wholesale_df.sort_values(by = col, ascending = False)
  if len(sort) > 5: sort = sort[:5]
  for index in sort.index: drop[index] = True 
    
    
# INFO Drop rows by index
#print ('Dropping {0} Outliers...'.format(len(drop)))
wholesale_df.drop(inplace = True, labels = drop.keys(), axis = 0)
print (wholesale_df.describe())


# Try "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "Normalizer"
T = preprocessing.StandardScaler().fit_transform(wholesale_df)
T1 = preprocessing.MinMaxScaler().fit_transform(wholesale_df)
T2 = preprocessing.MaxAbsScaler().fit_transform(wholesale_df)
T3 = preprocessing.Normalizer().fit_transform(wholesale_df)
T4 = wholesale_df # No Change
   
    
# Do KMeans
n_clusters = 3
centroids, labels = doKMeans(T, n_clusters)


# Print out centroids.
print (centroids)


# Project the centroids as well as the samples into the new 2D feature space for visualization purposes
def doPCA(data, dimensions = 2):
    from sklearn.decomposition import RandomizedPCA
    model = RandomizedPCA(n_components = dimensions)
    model.fit(data)
    return model
    
    
display_pca = doPCA(T)
T = display_pca.transform(T)
CC = display_pca.transform(centroids)


# Visualize all the samples and setup cluster labels
fig = plt.figure()
ax = fig.add_subplot(111)
if PLOT_TYPE_TEXT:
   for i in range(len(T)): ax.text(T[i,0], T[i,1], wholesale_df.index[i], color = c[labels[i]], alpha = 0.75, zorder = 600000)
   ax.set_xlim(min(T[:0])*1.2, max(T[:,0])*1.2)
   ax.set_ylim(min(T[:,1])*1.2, max(T[:,1])*1.2)
else:
    # Plot a regular scatter plot
    sample_colors = [c[labels[i]] for i in range(len(T))]
    ax.scatter(T[:, 0], T[:,1], c = sample_colors, marker = 'o', alpha = 0.2)
    
    
# Plot the centroids as X's, and label them
ax.scatter(CC[:,0], CC[:,1], marker = 'x', s = 169, linewidths = 3, zorder = 1000, c = c)
for i in range(len(centroids)): ax.text(CC[i, 0], CC[i,1], str(i), zorder = 50010, fontsize = 18, color = c[i])


# Display feature vectors for investigation:
if PLOT_VECTORS: drawVectors(T, display_pca.components_, wholesale_df.columns, plt)

# Add the cluster label back into the dataframe and display it:
wholesale_df['label'] = pd.Series(labels, index = wholesale_df.index)
print (wholesale_df)

plt.show()




