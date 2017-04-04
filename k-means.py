# simple k-means example, here we are importing a data set of 300 randomly generated points that are roughly grouped together in 3 regions of higher density visualized as a 2-d scatterplot.  Our goal is to group the samples based on feature similarity by randomly picking k centroids from the sample points as initial cluster centers.  Then assign samples to the nearest centroids.  Then we move the centroids to the center of the samples assigned to it and repeat until the cluster assignment is unchanged or a maximum # of iterations is reached.

# similarity between objects is measured by the squared Euclidean distance between x and y in m-dimensional space.  Therefore the k-means algorithm is essentially a simple optimization problem (minimizing sum of squared errors within cluster, known as cluster inertia).

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples = 300, n_features = 2, centers = 3, cluster_std = 0.5, shuffle = True, random_state = 0)
plt.scatter(X[:,0], X[:,1], c = 'blue', marker = 'o', s = 50)
plt.grid()
plt.show()

# set the number of desired clusters to 3 and specifying the n_init to 10 meaning that the k-means clustering algorithm runs 10 times independently with different random centroids to choose the final model with lowest sum of squared errors/cluster inertia.  Max_iter specifies the maximum number of iterations for each single run, thus k-means only stops early if it converges before the 300 iterations is reached.  Tol is tolerance and we use this to deal with convergence problems, since it's possible that k-means doesn't reach convergence for a particular run.
km = KMeans(n_clusters = 3, init = 'random', n_init = 10, max_iter = 300, tol = 1e-04, random_state = 0)
y_km = km.fit_predict(X)
