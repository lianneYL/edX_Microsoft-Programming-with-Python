# PCA - Stanford Armadillo 


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData, PlyElement


# Every 100 data samples, we save 1. 
#If things run too slow, try increasing this number. 
#If things run too fast, try decreasing it... =)
reduce_factor = 100
matplotlib.style.use('ggplot')


# Load up the scanned armadillo
plyfile = PlyData.read('/Users/.../stanford_armadillo.ply')
armadillo = pd.DataFrame({
  'x':plyfile['vertex']['z'][::reduce_factor],
  'y':plyfile['vertex']['x'][::reduce_factor],
  'z':plyfile['vertex']['y'][::reduce_factor]
})


def do_PCA(armadillo):
    pca = PCA(n_components = 2)
    pca.fit(armadillo)
    PCA(copy = True, n_components = 2, whiten = False)
    T = pca.transform(armadillo)
    T.shape
    return T

    
def do_RandomizedPCA(armadillo):
    from sklearn.decomposition import RandomizedPCA
    rpca = PCA(n_components = 2)
    rpca.fit(armadillo)
    PCA(copy = True, n_components = 2, whiten = False)
    rT = rpca.transform(armadillo)
    rT.shape
    return rT
    
    
# Render the Original Armadillo
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Armadillo 3D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(armadillo.x, armadillo.y, armadillo.z, c='green', marker='.', alpha=0.75)


# Time the execution of PCA 5000x
t1 = datetime.datetime.now()
for i in range(5000): pca = do_PCA(armadillo)
time_delta = datetime.datetime.now() - t1


# Render the newly transformed PCA armadillo!
if not pca is None :
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title('PCA, build time: ' + str(time_delta))
  ax.scatter(pca[:,0], pca[:,1], c='blue', marker='.', alpha=0.75)

  
# Time the execution of rPCA 5000x
t1 = datetime.datetime.now()
for i in range(5000): rpca = do_RandomizedPCA(armadillo)
time_delta = datetime.datetime.now() - t1


# Render the newly transformed RandomizedPCA armadillo!
if not rpca is None:
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title('RandomizedPCA, build time: ' + str(time_delta))
  ax.scatter(rpca[:,0], rpca[:,1], c='red', marker='.', alpha=0.75)


plt.show()
