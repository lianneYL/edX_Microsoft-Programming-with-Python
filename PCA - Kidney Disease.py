# PCA - Kidney Disease


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import assignment2_helper as helper


scaleFeatures = True
matplotlib.style.use('ggplot')


#load the dataset
kidney_disease = pd.read_csv('/Users/.../kidney_disease.csv')
kidney_diseaseDF = kidney_disease.dropna(axis = 0) #remove any rows with nan


# Create some color coded labels; the actual label feature
# will be removed prior to executing PCA, since it's unsupervised.
# You're only labeling by color so you can see the effects of PCA
labels = ['red' if i=='ckd' else 'green' for i in kidney_diseaseDF.classification] 
kidney_diseaseDF1 = kidney_diseaseDF[['bgr', 'wc', 'rc']] #Select only 'bgr', 'wc', 'rc' columns
kidney_diseaseDF1.dtypes #Check the data type of each column


#Change each column into numeric
kidney_diseaseDF1[['bgr', 'wc', 'rc']] = kidney_diseaseDF1[['bgr', 'wc', 'rc']].apply(pd.to_numeric)
kidney_diseaseDF1.dtypes
np.var(kidney_diseaseDF1) #Check the variance of each feature
kidney_diseaseDF1.describe()


if scaleFeatures: kidney_diseaseDF1 = scaleFeatures(kidney_diseaseDF1)


#Run PCA on your dataset and reduce it to 2 components
# Ensure your PCA instance is saved in a variable called 'pca',
# and that the results of your transformation are saved in 'T'.
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(kidney_diseaseDF1)
T = pca.transform(kidney_diseaseDF1)


ax = drawVectors(T, pca.components_, kidney_diseaseDF1.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()
