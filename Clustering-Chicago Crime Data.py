# Module5
# Clustering - Chicago crime data


from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
matplotlib.style.use('ggplot')


def doKMeans(df):
      
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df.Longitude, df.Latitude, marker ='.', alpha = 0.3)
    
    df = df[['Longitude', 'Latitude']]
   
    kmeans_model = KMeans(n_clusters = 7) # K-means, clusters = 7
    kmeans_model.fit(df)
    KMeans(copy_x = True, init = 'k-means++', max_iter = 300, n_clusters = 7, n_init = 10,
       n_jobs = 1, precompute_distances = 'auto', random_state = None, tol = 0.0001, verbose = 0)
    labels = kmeans_model.predict(df)
    
    # Plot the centroids
    centroids = kmeans_model.cluster_centers_
    ax.scatter(centroids[:,0], centroids[:,1], marker = 'x', c = 'red', alpha = 0.5, linewidths = 3, s = 169)
    print (centroids)

    
# Load the dataset
df = pd.read_csv('/Users/.../Crimes_-_2001_to_present.csv')
df = df.dropna(axis = 0) # Remove rows with nan
df.dtypes # Check the data type
df.Date = pd.to_datetime(df.Date) # Transform the Date into datetime64 data type
df.dtypes # Check data type of Date again


doKMeans(df)


# Filter out the data with Date > '2011-01-01' 
df1 = df[df.Date > '2011-01-01']


# Plot your data
doKMeans(df1)
plt.title('Date > 2011-01-01')
plt.show()





