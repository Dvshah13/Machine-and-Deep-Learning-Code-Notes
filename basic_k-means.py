# one of the most popular clustering algorithm, because it is very simple and easy to implement and it has shown good performance on different tasks.  Belongs to the class of partition algorithms that simultaneously partition data points into distinct groups called clusters. K-means picks k number of points for each cluster known as centroids. Each data point forms a cluster with the closest centroids i.e. k clusters. The main idea behind k-means is to find a partition of data points such that the squared distance between the cluster mean and each point in the cluster is minimized. This method assumes that you know a priori the number of clusters your data should be divided into.
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn import cluster

# loading and reading data
data = pd.read_csv('Data Sets for Code/faithful.csv')

# passing list of columns data
data.columns = ['eruptions', 'waiting']

# setting plot variables and labels
plt.scatter(data.eruptions, data.waiting)
plt.title('Old Faithful Scatterplot')
plt.xlabel('Length of eruption (minutes)')
plt.ylabel('Time between eruptions (minutes)')

# reshaping data into numpy array, necessary for scikit learn to read data
reshaped_data = np.array(data)

# in this basic example setting k=2 since there are 2 distinct groupings that were identified when reading the data and then running kmeans from scikit to our reshaped data
k = 2
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(reshaped_data)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
# creating visualization for cluster groups,
for i in range(k):
    # select only data observations with cluster label == i
    ds = faith[np.where(labels==i)]
    # plot the data observations
    plt.plot(ds[:,0],ds[:,1],'o', markersize=7)
    # plot the centroids
    lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
plt.show()
