## Often when faced with datasets that contain a large number of features, many of those may be unnecessary.  This is a typical problem where you want to log as much as you can to either get enough information to properly predict the target variable.  Some features are very informative for the prediction, some are somehow related and some are completely unrelated (they only contain noise or irrelevant information)  Dimensionality reduction is the operation of eliminating some features of the input dataset and creating a restricted set of features that contains all the information you need to predict the target variable in a more effective way.  Reducing the number of features usually also reduces the output variability and complexity (as well as the time)
## The main hypothesis behind many algorithms used in the reduction is the one pertaining to Additive White Gaussian Noise (AWGN) noise.  It is an independent Gaussian-shaped noise that is added to every feature of every dataset.  Reducing the dimensionality also reduces the energy of the noise since you're decreasing its span set.

## The Covariance Matrix - gives you an idea about the correlation between all the different pairs of features, usually a good first step of dimensionality analysis because it gives you an idea of the number of features that are strongly related (therefore the number of features you can likely discard) and the ones that are independent.
# Using the Iris dataset, where each observation has four features, can be easily computed and understood with the help of a simple graphical representation
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
cov_data = np.corrcoef(iris.data.T)
print iris.feature_names
print cov_data
# Graphical representation
import matplotlib.pyplot as plt
img = plt.matshow(cov_data, cmap=plt.cm.winter)
plt.colorbar(img, ticks=[-1, 0, 1])
plt.show() # from this image, we notice a high correlation of the first and third, first and fourth, and third and fourth features. We also see that the second feature is almost independent of the others; the others are somehow correlated to each other.  We now have a rough idea about the potential number of features in the reduced set: 2.

## Principle Component Analysis (PCA) - technique that helps you define a smaller and more relevent set of features.  The new features are linear combinations (rotation) of the current features.  After the rotation of the input space, the first vector of the output set contains most of the signal's energy (its variance).  The second is orthogonal to the first and it contains most of the remaining variance/energy; the third is orthogonal to the first two vectors and contains most of the remaining variance/energy and so on.  Ideally, the initial vectors contain all the information of the input signal and the ones towards the end contain only certain noise.
## Since the output basis is orthogonal, you can decompose and synthesize an approximate version of the input dataset.  The key parameter used to decide how many basis vectors one can use is the energy/variance.  Since the algorithm under the hood is for singular value decomposition, eigenvectors (the basis vectors) and eigenvalues (the standard deviation associated to that vector) are the terms that are often referred to with PCA.  The cardinality of the output set is the one that guarantees the presence of 95 percent (sometimes 99 percent) of the variance.
from sklearn.decomposition import PCA
pca_2c = PCA(n_components=2)
X_pca_2c = pca_2c.fit_transform(iris.data)
print X_pca_2c.shape
plt.scatter(X_pca_2c[:,0], X_pca_2c[:,1], c=iris.target, alpha=0.8, edgecolors='none')
plt.show()
print pca_2c.explained_variance_ratio_.sum()
# we can see that after applying the PCA, the output set has only two features.  This is because PCA() was called with n_components = 2.  You can change this and test different values to get the optimal result.  Our output data set contained almost 98 percent of the energy of the input signal
# transformation matrix, comprised of four columns (number of input features) and two rows (number of the reduced ones)
print pca_2c.components_
# sometimes PCA is not effective enough, then you can try to whiten the signal.  This is where the eigenvectors are forced to unit component-wise-variances.  Whitening removes information, but sometimes, it improves the accuracy of the machine learning algoritms.  An example of whitening (note in this case it doesn't change anything except for the scale of the dataset with reduced output)
pca_2cw = PCA(n_components=2, whiten=True)
X_pca_1cw = pca_2cw.fit_transform(iris.data)
plt.scatter(X_pca_1cw[:,0], X_pca_1cw[:,1], c=iris.target, alpha=0.8, edgecolors='none')
plt.show()
print pca_2cw.explained_variance_ratio_.sum()
# project the input dataset on a 1D space generated with PCA
pca_1c = PCA(n_components=1)
X_pca_1c = pca_1c.fit_transform(iris.data)
plt.scatter(X_pca_1c[:,0], np.zeros(X_pca_1c.shape), c=iris.target, alpha=0.8, edgecolors='none')
plt.show()
print pca_1c.explained_variance_ratio_.sum() # here the output energy is lower (92.4 percent of the original signal) and the output points are added to the monodimensional Euclidean space.  This might not be a great feature reduction step since many points with different labbels are mixed together.
## A nice trick to use to ensure you generate an output set containing at least 95 percent of the input energy, you can just specify this value to the PCA object during its first call.  A result equal to the one with two vectors can be obtained with the following code:
pca_95pc = PCA(n_components=0.95)
X_pca_95pc = pca_95pc.fit_transform(iris.data)
print pca_95pc.explained_variance_ratio_.sum()  # get roughly 97.7 percent
print X_pca_95pc.shape  # (150,2)

## A variation of PCA for big data - RandomizedPCA
## The main issue of PCA is the complexity of the underlying Singular Value Decomposition (SVD) algorithm.  There is a faster algorithm in Scikit-Learn based on Randomized SVD.  It is a lighter but approximate iterative decomposition method.  With Randomized SVD, the full-rank reconstruction is not perfect and the basis vectors are locally optimized in every iteration.  On the other hand, to get a good approximation, it only requires a few steps making it faster than the classical SVD algorithms.  Therefore it's a great choice if the training dataset is big.
from sklearn.decomposition import RandomizedPCA
rpca_2c = RandomizedPCA(n_components=2)
X_rpca_2c = rpca_2c.fit_transform(iris.data)
plt.scatter(X_rpca_2c[:,0], X_rpca_2c[:,1], c=iris.target, alpha=0.8, edgecolors='none')
plt.show()
print rpca_2c.explained_variance_ratio_.sum() # get around 97.7 again but on larger datasets the results could vary significantly

## Latent Factor Analysis (LFA) - another technique to reduce dimensionality of the dataset.  Idea is similar to PCA, however in this case there's no orthogonal decomposition of the input signal and therefore no output basis.  Some data scientists believe that LFA is a generalization of PCA that removes the constraint of orthogonality.  LFA is generally used when a latent factor or a construct is in the system and all the features are observations of the variables of the latent factor that is linearly transformed and which has an Arbitrary Waveform Generator (AWG) noise.  It's generally assumed that the latent factor has a Gaussian distribution and a unitary covariance.  Therefore, instead of collapsing the energy/variance of the signal, the covariance among the variables is explained in the output dataset.  Scikit-Learn implements an iterative algorithm making it great for large datasets
## Here is code to lower dimensionality of the iris dataset by assuming two latent factors in the system
from sklearn.decomposition import FactorAnalysis
fact_2c = FactorAnalysis(n_components=2)
X_factor = fact_2c.fit_transform(iris.data)
plt.scatter(X_factor[:,0], X_factor[:,1], c=iris.target, alpha=0.8, edgecolors='none')
plt.show()

## Linear Discriminant Analysis (LDA) - LDA is strictly speaking a classifier but is often used for dimensionality reduction.  Since it's a supervised approach, it requires the label set to optimize the reduction step.  LDA outputs linear combinations of the input features, trying to model the difference between the classes that best discriminate them (since LDA uses label information). Compared to PCA, the output dataset that is obtained with the help of LDA contains neat distinction between classes.  Cannot be used in regression problems, only used in classification.
from sklearn.lda import LDA
lda_2c = LDA(n_components=2)
X_lda_2c = lda_2c.fit_transform(iris.data, iris.target)
plt.scatter(X_lda_2c[:,0], X_lda_2c[:,1], c=iris.target, alpha=0.8, edgecolors='none')
plt.show()

## Latent Semantical Analysis (LSA) - typically applied to text after it is processed with TfidfVectorizer or CountVectorizer.  Compared to PCA, it applies SVD to the input dataset (which is usually a sparse matrix) producing semantic sets of words usually associated with the same concept. This is why LSA is used when the features are homogeneous (that is, all the words in the documents) and are present in large numbers
# Example with fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups
categories = ['sci.med', 'sci.space']
twenty_sci_news = fetch_20newsgroups(categories=categories)
from sklearn.feature_extraction.text import TfidfVectorizer
tf_vect = TfidfVectorizer()
word_freq = tf_vect.fit_transform(twenty_sci_news.data)
from sklearn.decomposition import TruncatedSVD
tsvd_2c = TruncatedSVD(n_components=50)
tsvd_2c.fit(word_freq)
print np.array(tf_vect.get_feature_names())[tsvd_2c.components_[20].argsort()[-10:][::-1]]

## Independent Component Analysis (ICA) - An approach where you try to derive independent components from the input signal.  In fact, ICA is a technique which allows you to create maximally independent additive subcomponents from the multivariate input signal.  The main hypothesis of this technique focuses on the statistical independence of the subcomponents and their non-Gaussian distribution. A typical use case is blind source separation, for example is you have two or more microphones picking up a person speaking and a song playing, ICA is able to separate the two sounds into two output features.  Scikit-Learn has a faster version of this algorithm FastICA, similar to the other techniques

## kernel PCA - uses a kernel to map the signal on a (typically) nonlinear space and makes it linearly separable (or close to).  It's an extension of PCA where the mapping is an actual projection on a linear subspace.  There are many well-known kernels, most used are: linear, poly, RBF, sigmoid and cosine.  All serve different configurations of input datasets and can only linearize selected types of data.
# Example using a disk-shaped dataset
def circular_points(radius, N):
    return np.array([[np.cos(2*np.pi*t/N)*radius, np.sin(2*np.pi*t/N)*radius] for t in xrange(N)])
N_points = 50
fake_circular_data = np.vstack([circular_points(1.0, N_points), circular_points(5.0, N_points)])
fake_circular_data += np.random.rand(*fake_circular_data.shape)
fake_circular_target = np.array([0]*N_points + [1]*N_points)
plt.scatter(fake_circular_data[:,0], fake_circular_data[:,1], c=fake_circular_target, alpha=0.8, edgecolors='none')
plt.show()
# with this input dataset, all the linear transforms will fail to separate blue and red dots since the dataset contains circumference-shaped classes.  We'll use the RBF kernel and see what happens
from sklearn.decomposition import KernelPCA
kpca_2c = KernelPCA(n_components=2, kernel='rbf')
X_kpca_2c = kpca_2c.fit_transform(fake_circular_data)
plt.scatter(X_kpca_2c[:,0], X_kpca_2c[:,1], c=fake_circular_target, alpha=0.8, edgecolors='none')
plt.show() # here we see a separation, blue dots on the right and red dots on the left

## Restricted Boltzmann Machine (RBM) - composed of linear functions (usually called hidden units or neurons), creates a nonlinear transformation of the input data.  The hidden units represents the status of the system and the output dataset is actually the status of that layer.  The main hypothesis is that the input dataset is composed of features that represent probability (binary values or real values in the [0,1] range) since RBM is a probabilistic approach.
# Here we'll feed the RBM with binarized pixels of images as features (1=white, 0=black) and we will print out the latent components of the system.  These components represent different generic faces that appear in the original images
from sklearn import preprocessing
from sklearn.neural_network import BernoulliRBM
n_components = 64 # can try this with 64, 100, 144
olivetti_faces = datasets.fetch_olivetti_faces()
X = preprocessing.binarize(preprocessing.scale(olivetti_faces.data), 0.5)
rbm = BernoulliRBM(n_components=n_components, learning_rate = 0.01, n_iter=100)
rbm.fit(X)
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(int(np.sqrt(n_components+1)), int(np.sqrt(n_components+1)), i + 1)
    plt.imshow(comp.reshape((64, 64)), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle(str(n_components) + ' components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
plt.show()
Scikit-Learn contains just the base layer of RBM processing.  On a big dataset, you're better off using GPU-based toolkits (like ones built on the top of CUDA or OpenCL) since RBMs are highly parallelizable
