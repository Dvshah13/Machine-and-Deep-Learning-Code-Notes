## Technique used when in the situation where features and target variables are not really related.  You can modify the input dataset, apply linear or nonlinear transforms that can improve the accuracy and other various techniques.

## Example using the california housing dataset, you're trying to predict the value of a house and you just know the height, width and the length of each room.  You can artificially build a feature that represents the volume of the house.  This is strictly not an observed feature but it's a feature built on top of the existing ones.
import numpy as np
from sklearn import datasets # import the dataset, see that it is a regression problem (house prices are a real value)
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor # importing a simple regressor, will calculate Mean Absolute Error
cali = datasets.california_housing.fetch_california_housing()
X = cali['data']
Y = cali['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8) # defining train and test sets

regressor = KNeighborsRegressor()
regressor.fit(X_train, Y_train)
Y_est = regressor.predict(X_test)
print "Mean Absolute Error =", mean_squared_error(Y_test, Y_est) # MAE is around 1.15, which is good but we can try to do better using different techniques
## Normalize the input features using Z-score and compare the regression tasks on this new feature set.  Z-normalization is simply the mapping of each feature to a new one with a null mean and unitary variance.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
regressor = KNeighborsRegressor()
regressor.fit(X_train_scaled, Y_train)
Y_est = regressor.predict(X_test_scaled)
print "Mean Absolute Error =", mean_squared_error(Y_test, Y_est) # just by normalizing we dropped the error to only .38
## Now add a nonlinear modification to a specific feature.  We can assume that the output is roughly related to the number of occupiers of a house.  In fact, there is a big difference between the price of a house occupied by a single person and the price for three people staying in the same house.  However, the difference between 10 people and 12 people is not great (difference but much smaller).  We'll add another feature built as a nonlinear transform of another one
non_linear_feat = 5 # Average Ocupancy
# The creating process is, created new feature (square root of it), attach it to the dataset, do it for both the train and test set
X_train_new_feat = np.sqrt(X_train[:,non_linear_feat])
X_train_new_feat.shape = (X_train_new_feat.shape[0], 1)
X_train_extended = np.hstack([X_train, X_train_new_feat])

X_test_new_feat = np.sqrt(X_test[:,non_linear_feat])
X_test_new_feat.shape = (X_test_new_feat.shape[0], 1)
X_test_extended = np.hstack([X_test, X_test_new_feat])

scaler = StandardScaler()

X_train_extended_scaled = scaler.fit_transform(X_train_extended)
X_test_extended_scaled = scaler.fit_transform(X_test_extended)
regressor = KNeighborsRegressor()
regressor.fit(X_train_extended_scaled, Y_train)
Y_est = regressor.predict(X_test_extended_scaled)
print "Mean Absolute Error =", mean_squared_error(Y_test, Y_est) # we've further reduced the MAE to .34 and obtained a fairly statisfying regressor.  There are other methods you can use but this is a good starting point
