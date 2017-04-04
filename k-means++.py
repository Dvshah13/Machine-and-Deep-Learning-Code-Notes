# Algorithm which can lead to better and more consistent results then k-means by placing the initial centroids far away from each other.  Initializing k-means ++ is as follows: 1. initialize an empty set (M) to store the k centroids being selected. 2. Randomly choose the first centroid from the input samples and assign it to M.  3. For each sample x^i that is not in M, find the minimum squared distance to any of the centroids in M. 4. Use a weighted probability distribution to randomly select the next centroid. 5. Repeat steps 2 and 3 until k centroids are chosen and proceed with classic k-means algorithm.

# to use kmeans++ in scikit learn, set the init parameter to k-means++ instead of random.

### Note code is just building upon k-means.py file ###

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples = 300, n_features = 2, centers = 3, cluster_std = 0.5, shuffle = True, random_state = 0)
plt.scatter(X[:,0], X[:,1], c = 'blue', marker = 'o', s = 50)
plt.grid()
plt.show()

# set the number of desired clusters to 3 and specifying the n_init to 10 meaning that the k-means clustering algorithm runs 10 times independently with different random centroids to choose the final model with lowest sum of squared errors/cluster inertia.  Max_iter specifies the maximum number of iterations for each single run, thus k-means only stops early if it converges before the 300 iterations is reached.  Tol is tolerance and we use this to deal with convergence problems, since it's possible that k-means doesn't reach convergence for a particular run.

### K-means++ initialized ###
km = Kmeans(n_clusters = 3, init = 'k-means++', n_init = 10, max_iter = 300, tol = 1e-04, random_state = 0)
y_km = km.fit_predict(X)

# visulalize the clusters k-means identified with cluster centroids
plt.scatter(X[y_km == 0,0], X[y_km == 0,1], s = 100, c = 'blue', marker = 's', label = 'cluster 1')
plt.scatter(X[y_km == 1,0], X[y_km == 1,1], s = 100, c = 'green', marker = 'o', label = 'cluster 2')
plt.scatter(X[y_km == 2,0], X[y_km == 2,1], s = 100, c = 'red', marker = 'v', label = 'cluster 3')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s = 350, marker = '*', c = 'black', label = 'centroids')

plt.legend()
plt.grid()
plt.show()
