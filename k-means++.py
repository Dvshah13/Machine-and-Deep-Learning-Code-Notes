# Algorithm which can lead to better and more consistent results then k-means by placing the initial centroids far away from each other.  Initializing k-means ++ is as follows: 1. initialize an empty set (M) to store the k centroids being selected. 2. Randomly choose the first centroid from the input samples and assign it to M.  3. For each sample x^i that is not in M, find the minimum squared distance to any of the centroids in M. 4. Use a weighted probability distribution to randomly select the next centroid. 5. Repeat steps 2 and 3 until k centroids are chosen and proceed with classic k-means algorithm.

# to use kmeans++ in scikit learn, set the init parameter to k-means++ instead of random.
from sklearn.cluster import Kmeans

km = Kmeans(n_clusters = 3, init = 'k-means++', n_init = 10, max_iter = 300, tol = 1e-04, random_state = 0)
y_km = km.fit_predict(X)
