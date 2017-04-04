# often when working with unsupervised data, you don't want to specify the labels or set specific cluster groups.  Using the elbow method, you can estimate the optimal number of clusters k for a given task.  The idea is to identify the value of k where the distortion begins to increase most rapidly (if k increases the distortion will decrease).

### elbow method ###
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1,11), distortions, marker = 'o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortions')
plt.show()

# you can aso use another metric to evaluate the quality of a clustering known as silhouette analysis.  This can be used as a graphical tool to plot a measure of how tightly grouped the samples in the clusters are.  To do this, we must calculate the cluster cohesion as the average distance between a sample and all other points in the same cluster.  Then calculate the cluster separation from the next closest cluster as the average distance between the sample x^i and all samples in the nearest cluster.  Finally we calculate the silhouette s^i as the difference between cluster cohesion and separation divided by the greater of the two.  The silhouette coefficient is bounded within the range -1 and 1.  The ideal silhouette is close to 1.

# we can get this coefficient using scikit learn's silhouette_samples
km = Kmeans(n_clusters = 3, init = 'k-means++', n_init = 10, max_iter = 300, tol = 1e-04, random_state = 0)
y_km = km.fit_predict(X)

import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_values = silhouette_samples(X, y_km, metric = 'euclidean')

y_axis_lower, y_axis_upper = 0,0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_values = silhouette_values[y_km == c]
    c_silhouette_values.sort()
    y_axis_upper += len(c_silhouette_values)
    color = cm.jet(i/n_clusters)
    plt.barh(range(y_ax_lower, y_axis_upper), c_silhouette_values, height = 1.0, edgecolor = 'none', color = color)
    yticks.append((y_ax_lower + y_axis_upper) / 2)
    y_axis_lower += len(c_silhouette_values)

silhouette_average = np.mean(silhouette_values)
plt.axvline(silhouette_average, color = 'red', linestyle = '--')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette Coefficient')
plt.show()
# look for silhouette coefficients that aren't near zero, closer to 1 and are visibly similar lengths and widths.
