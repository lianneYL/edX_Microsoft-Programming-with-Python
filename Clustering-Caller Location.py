# Module5 
# Clustering - Assignment 2
# Find the caller's location


from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
matplotlib.style.use('ggplot')


# Load the data
cdr_df = pd.read_csv('/Users/.../CDR.csv')


def showandtell(title=None):
  if title != None: plt.savefig(title + ".png", bbox_inches='tight', dpi=300)
  plt.show()
  exit()

  
# Convert date and time 
cdr_df.CallDate = pd.to_datetime(cdr_df.CallDate)
cdr_df.CallTime = pd.to_timedelta(cdr_df.CallTime)
cdr_df.Duration = pd.to_timedelta(cdr_df.Duration)
cdr_df.dtypes


# Find unique user phone number
in_numbers = list(cdr_df.In.unique())


# User1 with "In" = 4638472273 
user1 = cdr_df[(cdr_df.In == in_numbers[0])]
# Plot all the call locations
user1.plot.scatter(x='TowerLon', y='TowerLat', c='gray', alpha=0.1, title='Call Locations')


# Select only records that came in on Sat & Sun
user1 = user1[(user1.DOW == 'Sat') | (user1.DOW == 'Sun')]
# Select calls that are came in either before 6AM OR after 10pm (22:00:00).
user1 = user1[(user1.CallTime < '06:00:00') | (user1.CallTime > '22:00:00')]


# Print out the length of user1
print (len(user1))


# Plot the potential location of user1
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(user1.TowerLon,user1.TowerLat, c='g', marker='o', alpha=0.2)
ax.set_title('Weekend Calls (<6am or >10p)')


# Run K-Means with a K=1
kmeans = KMeans(n_clusters = 1)
user1 = pd.concat([user1.TowerLon, user1.TowerLat], axis = 1)
labels = kmeans.fit_predict(user1)
centroids = kmeans.cluster_centers_
ax.scatter(x = centroids[:,0], y = centroids[:,1], c = 'r', marker = 'x', s = 100)
print(centroids)


# Repeat the above steps for all 10 individuals, and record their approximate home locations
locations = []
for i in range(10):
	user = cdr_df[(cdr_df.In == in_numbers[i])]
	user.plot.scatter(x='TowerLon', y='TowerLat', c='purple', alpha=0.12, title='Call Locations', s = 30)
	user = user[(user.DOW == 'Sat') | (user.DOW == 'Sun')]
	user = user[(user.CallTime < "06:00:00") | (user.CallTime > "22:00:00")]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(user.TowerLon, user.TowerLat, c='g', marker='o', alpha=0.2)
	ax.set_title('Weekend Calls (<6am or >10p)')
	kmeans = KMeans(n_clusters = 1)
	user = pd.concat([user.TowerLon, user.TowerLat], axis = 1)
	labels = kmeans.fit_predict(user)

	centroids = kmeans.cluster_centers_
	ax.scatter(x = centroids[:, 0], y = centroids[:, 1], c = 'r', marker = 'x', s = 100)
	locations.append(centroids)

 
 