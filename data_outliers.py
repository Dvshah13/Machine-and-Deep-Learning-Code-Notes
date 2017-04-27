## Outliers can derail the core learning process in data science.  3 general causes of a point being an outlier: 1. point represents a rare occurence, 2. point represents the usual occurrence of another distribution, 3. point is clearly some kind of mistake (just remove)

## To explain the reason behind a data point being an outlier, you are required to first locate the possible outliers in your data.  There are a good number of approaches - some univariate (observe each singular variable at once), others multivariate (consider more variables at the same time).  The univariate methods are usually based on EDA and visualizations such as boxplots.

## A couple of rules of thumb to keep in mind when working with single variables.  Both are based on the fact that outliers may be spotted as extreme values:  if you are observing Z-scores, observations with scores higher than 3 in absolute values have to be considered as suspect outliers.  if you are observing a description of data, you can take as suspect outliers the observations that are smaller than the 25th percentile value minus the IQR(interquartile range), the difference between the 75thand 25th percentile values * 1.5 and those greater than the 75th percentile value plus the IQR * 1.5, usually done with help of boxplot
# using Z-score and Boston House Prices dataset
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()
continuous_variables = [n for n in range(np.shape(boston.data)[1]) if n!=3]
# standardize all the continuous variables using the StandardScaler function from sklearn.  Our target is the fancy indexing of boston.data boston.data[:,continuous_variables] in order to create another array containing all the variables except the previous one that sas indexed 3
# StandardScaler automatically standardizes the zero mean and unit variance.  This is a necessary routine operation that should be performed before feeing the data to the learning phase. Otherwise the algorithm won't work properly.
# let's locate values that are above the absolute value of 3 standard deviations
from sklearn import preprocessing
normalized_data = preprocessing.StandardScaler().fit_transform(boston.data[:,continuous_variables])
outliers_rows, outliers_columns = np.where(np.abs(normalized_data)>3) # the outliers_row and outliers_columns variables contain the row and column indexes of the suspect outliers.  we can print the index of the examples:
print outliers_rows
# we can display the tuple of the row/column coordinates in the arry:
print(list(zip(outliers_rows, outliers_columns)))
# the univariate approach can reveal quite a lot of potential outliers.  It won't disclose an outlier that does not have an extreme value.  However it will reveal the outlier if it finds an unusual combination of values in two or more variables.  Often in such cases, the values of the involved variables may not even be extreme and therefore the outlier may slip away unnoticed.
# In order to discover cases where this happens, you can use a dimensionality reduction algorithm, such as PCA and then check the absolute values of the components that are beyond three standard deviations
# Scikit-Learn offers a couple of classes that can automatically work for you straight out of the box and signal all suspect classes: the covariance.EllipticEnvelope class fits a robust distribution estimation of your data, pointing out outliers that might be contaminating your dataset because they are the extreme points in the general distribution of the data. svm.OneClassSVM class is a support vector machine algorithm that can approximate the shape of your data and find out if any new instances provided should be considered as a novelty (acts as novelty detector by default, assumes no outlier in data), by modifying its parameters can work on dataset where outliers are present providing an even more robust and reliable outlier detection system than EllipticEnvelope

## EllipticEnvelope - a function that tries to figure out the key parameters of your data's general distribution by assuming that your entire data is an expression of an underlying multivariate Gaussian distribution.  We can say that it checks the distance of each observation with respect to a grand mean that takes into account all the variables in your dataset.  Thus it is able to spot both univariate and multivariate outliers.
# The only parameter you have to take into account when using this function from the covariance module is the contamination parameter, which can take a value of up to 0.5.  Situations may vary from dataset to dataset, however as a starting figure, we suggest a value from 0.01-0.02 since it is the percentage of observations that should fall over the absolute value 3 in the Z score distance from the mean in a standardize normal distribution.  For this reason, we deem the default value of 0.1 as too high.
# create an artifical distribution made of blobs
from sklearn.datasets import make_blobs
blobs = 1 # The number of distributions (parameter centers) is related to the user-defined variable blobs, which is initially set to 1
blob = make_blobs(n_samples=100, n_features=2, centers=blobs, cluster_std=1.5, shuffle=True, random_state=5) # creates a certain number of distributions into a bidimensional space for a total of 100 examples (n_samples parameter).
# Robust Covariance Estimate
from sklearn.covariance import EllipticEnvelope
robust_covariance_est = EllipticEnvelope(contamination=.1).fit(blob[0]) # running EllipticEnvelope with a contamination rate of 10 percent helps you find out the most extreme values in the distribution.  The model deploys first fit by using the .fit() method on the EllipticEnvelope class.
detection = robust_covariance_est.predict(blob[0]) # Now the model prediction is obtained by using the predict() method on the data that was used for fit.
outliers= np.where(detection == -1) # results corresponding to a vector of values 1 and -1, -1 being the mark for anomalous examples
inliers = np.where(detection == 1)
# the distinction between inliers and outliers is recorded in the variable's outliers and inliers which contain the indexes of the examples
# Draw the distribution and detected outliers
import matplotlib.pyplot as plt # just the distribution
plt.plot(blob[0][:,0], blob[0][:,1], 'x', markersize=10, color='black', alpha=0.8)
plt.show()
# The distribution and the outliers
a = plt.plot(blob[0][inliers,0],blob[0][inliers,1],'x',markersize=10,color='black',alpha=0.8,label='inliers')
b = plt.plot(blob[0][outliers,0],blob[0][outliers,1],'o',markersize=6,color='black',alpha=0.8,label='outliers')
plt.legend((a[0],b[0]),('inliers', 'outliers'), numpoints=1, loc='lower right')
plt.show() # in the case of a unique underlying multivariate distribution (when the variable blobs=1), the EllipticEnvelope algorithm has successfully located 10 percent of the observations on the fringe of the distribution and has consequently signaled all suspect outliers.
# a limitation of the EllipticEnvelope is when multiple distributions are present in the data as if there were two or more natural clusters, the algorithm, trying to fit a unique general distribution, tends to locate the potential outliers on just the most remote cluster, thus ignoring other areas of data that might be potentially affected by outlying cases, which is a situation that could occur with real data.
# Boston data to use in data that's real
from sklearn.decomposition import PCA
# Normalized data relative to continuous variables
continuous_variables = [n for n in range(np.shape(boston.data)[1]) if n != 3]
normalized_data = preprocessing.StandardScaler().fit_transform(boston.data[:,continuous_variables]) # standardize the data
# Just for visualization purposes pick the first 2 PCA components
pca = PCA(n_components=2) # 2 components account for 62 percent of the initial variance expressed by the 12 continuous variables available in the dataset, sufficient for visualization purposes but you normally need more than 2 since target is to have enough to account for at least 95 percent of the total variance.
Zscore_components = pca.fit_transform(normalized_data)
vtot = 'PCA Variance explained ' + str(round(np.sum(pca.explained_variance_ratio_),3))
v1 = str(round(pca.explained_variance_ratio_[0],3))
v2 = str(round(pca.explained_variance_ratio_[1],3))
# Robust Covariance Estimate
robust_covariance_est = EllipticEnvelope(store_precision=False, assume_centered=False, contamination=.05) # code estimates that EllipticEnvelope (assuming a low contamination eqivalent to .05) predicts the outlier and stores them in an array just as it also stores the inliers.
robust_covariance_est.fit(normalized_data)
detection = robust_covariance_est.predict(normalized_data)
outliers = np.where(detection == -1)
regular = np.where(detection == 1)
# Draw the distribution and the detected outliers, remember the two components account for about 62 percent of the variance in the data, therefore it seems there should be two main distinct clusters of house prices in Boston corresponding to the high and low end units in the market.  This is a nonoptimal situation for EllipticEnvelope estimations.
from matplotlib import pyplot as plt
a = plt.plot(Zscore_components[regular,0], Zscore_components[regular,1], 'x', markersize=2, color='black', alpha=0.6, label='inliers')
b = plt.plot(Zscore_components[outliers,0], Zscore_components[outliers,1], 'o', markersize=6, color='black', alpha=0.8, label='outliers')
plt.xlabel('1st component ('+v1+')')
plt.ylabel('2nd component ('+v2+')')
plt.xlim([-7,7])
plt.ylim([-6,6])
plt.legend((a[0],b[0]),('inliers','outliers'), numpoints=1, loc='best')
plt.title(vtot)
plt.show()
# In line with what was noticed while experimenting with the synthetic blobs the algorithm in this instance pointed out the outliers on just a cluster - the lesser one.  There is strong reason to believe given this that we just recieved a partial response and some further investigation will be required for the same.  Scikit Learn integrates the robust covariance estimation method which is a statistical approach with another methodology that is well rooted in machine learning, the OneClassSVM.

## OneClassSVM - machine learning algorithm that learns from the data what the data distribution should be and it is therefore applicable in a bigger variety of datasets.  It is great if you have a clean dataset and have it fit perfectly.  OneClassSVM can be summoned to check if any new example fits in the historical distribution and if it doesn't, it will signal a novel example, which might be both an error or some new, previously unseen situation.  OneClassSVM can spot things that are different rather than just try to fit the new example into existing topic categorization.  OneClassSVM can also be used to spot existing outliers.  If a distribution could not be modeled by this SVM class and it lies at the margins then there if something most likely fishy with it.
# In order to have it work as an outlier detector, you need to work on its core parameters.  OneClassSVM requires you to define the kernel, degree, gamma, and nu.
# Kernel and Degree - interconnected, usually the values that we suggest are the default ones.  The value of the kernel should be rbf and its degree should be 3.  Such parameters will inform OneClassSVM to create a series of classification bubbles that span through three dimensions, allowing you to model even the most complex multidimensional distribution forms.
# Gamma - parameter connected to the rbf kernel.  We suggest that you keep it as low as possible.  A good rule of thumb should be to assign it a minimum value that lies between the inverse of the number of cases and the variables.  Higher gamma values tend to lead the algorithm to follow the data more to define the shape of the classification bubble.
# Nu - parameter that determines whether we have to fit the exact distribution or we should try to keep a certain generalization by not adapting too much to the present data (necessary if outliers are present).  It can be determined by the following formula: nu_estimate = 0.95 * outliers_fraction + 0.05.  If the value of the outliers fraction is very small, Nu will be small and the SVM algorthm will try to fit the contour of the data points.  On the other hand, if the fraction is high, so will be the parameter, forcing a smooth boundary of the inliers' distributions.
# Example on Boston housing dataset
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import svm
# Normalized data relative to continuous variables
continuous_variables = [n for n in range(np.shape(boston.data)[1]) if n != 3]
normalized_data = preprocessing.StandardScaler().fit_transform(boston.data[:,continuous_variables])
# Just for visualization purposes pick the first 5 PCA components
pca = PCA(n_components=5)
Zscore_components = pca.fit_transform(normalized_data)
vtot = 'PCA Variance explained ' + str(round(np.sum(pca.explained_variance_ratio_),3))
# OneClassSVM fitting and estimates
outliers_fraction = 0.02 # if you change this to a larger value such as 0.1, you may experience the result when supporting a larger incidence of anomalous cases in your data
nu_estimate = 0.95 * outliers_fraction + 0.05 # calculating nu
machine_learning = svm.OneClassSVM(kernel="rbf", gamma=1.0/len(normalized_data), degree=3, nu=nu_estimate)
machine_learning.fit(normalized_data)
detection = machine_learning.predict(normalized_data)
outliers = np.where(detection == -1)
regular = np.where(detection == 1)
# Draw the distribution and the detected outliers
from matplotlib import pyplot as plt
for r in range(1,5):
    a = plt.plot(Zscore_components[regular,0], Zscore_components[regular,r], 'x', markersize=2, color='blue', alpha=0.6, label='inliers')
    b = plt.plot(Zscore_components[outliers,0], Zscore_components[outliers,r], 'o', markersize=6, color='red', alpha=0.6, label='inliers')
    plt.xlabel('Component 1 (' +str(r+1) + '(' + str(round(pca.explained_variance_ratio_[r],3)) + ')')
    plt.xlim([-7,7])
    plt.ylim([-6,6])
plt.legend((a[0],b[0]),('inliers','outliers'),numpoints=1, loc='best')
plt.title(vtot)
plt.show() # this approach modeled the distribution of the house price data better and spotted a few extreme values on the borders of the distribution.  You can use these outlier methods or even use both to further scrutinize the characteristics of the outliers in order to figure out a reason for them (could make you reflect on the generative processes underlying your data) and trying to build machine learning models; including under-weighting or excluding the outlying observations.
