# alternative approach to prototype-based clustering, that has the advantage of allowing us to plot the visualizations of a binary hierarchical clustering (dendrograms) which can help with the interpretation of results by making meaningful classifications.  There are two main approaches to hierarchical clustering, agglomerative and divisive hierarchical clustering. In agglormerative clustering, we start with each sample as an individual cluster and merge the closest pairs of clusters until only one cluster remains.  In divisive the opposite is true, we start with one cluster that encompasses all of our samples and iteratively split the cluster into smaller clusters until each cluster only contains one sample.  With agglomerative clustering, there are two standard algorithms, single linkage and complete linkage.  In single linkage you compute the distances between the most similar members for each pair of clusters and merge the two clusters for which the distance between the most similar
# members is the smallest. In complete linkage, the approach is similar but instead of comparing the most similar members in each pair of clusters, you compare the most dissimilar members to perform the merge.

# how to code, agglormerative clusters using complete linkage: compute the distance matrix of all samples, represent each data point as a singleton cluster, merge the two closest clusters based on distance of most dissimilar members, update similarity matrix and repeat until one single cluster remains
import pandas as import pd
import numpy as np

np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5,3]) * 10
df = pd.DataFrame(X, columns = variables, index = labels)
print df

# calculate the distance matrix as input for the hierarchical clustering algorithm
from scipy.spatial.distance import pdist, squareform
# calculating Euclidean distance between each pair of sample points in our data set based on features X, Y, Z.  Provide the condensed distance matrix, pdist as input to squareform function to create a symetrical matrix of pair-wise distances.
row_dist = pd.DataFrame(squareform(pdist(df, metric = 'euclidean')), columns = labels, index = labels)
print row_dist

# call the linkage function
from scipy.cluster.hierarchy import linkage
row_clusters = linkage(pdist(df, metric = 'euclidean'), method = 'complete') # important note you can use the input sample matrix (df.values) or condensed distance matrix but using the squareform distance matrix would yield different distance values form those expected

# turn into a pandas dataframe for easier viewing
pd.DataFrame(row_clusters, columns = ['row label 1', 'row label 2', 'distance', 'num of items in cluster'], index = ['cluster %d' %(i+1) for i in range(row_clusters.shape[0])])

# visualize results in the form of dendrogram
from scipy.cluster.hierarchy import dendrogram
row_dendrogram = dendrogram(row_clusters, labels = labels)
plt.tight_layout()
plt.ylabel('Euclidean Distance')
plt.show()
