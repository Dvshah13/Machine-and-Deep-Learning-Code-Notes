## You'll be working often with categorical data.  A plus point is the values are Booleans, they can be seen as the presence or absence of a feature or on the other side the probability of a feature having an exhibit (has displayed, has not displayed).  Since many ML algos don't allow the input to be categorical, boolean features are often used to encode categorical features as numerical values.
## This process maps a feature with each level of the categorical feature.  On;y one binary feature reveals the presence of the categorical feature, the others remain 0.  This operation is called dummy coding.
import pandas as pd
categorical_features = pd.Series(['sunny', 'rainy', 'cloudy', 'snowy'])
mapping = pd.get_dummies(categorical_features)
print mapping
# this output is a DataFrame that contains the categorical levels as column lables and the respective binary features along the column.

## Here you can map a categorical value to a list of numerical ones
print mapping['sunny']
print mapping['cloudy']

## you can use Scikit learn to do the same operation although the process is a bit more complex as you must convert text to categorical indices but the result is the same
## SK lean example
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ohe = OneHotEncoder()
levels = ['sunny', 'rainy', 'cloudy', 'snowy'] # your features
fit_levs = le.fit_transform(levels) # maps the text to 0-to-N integer number
ohe.fit([[fit_levs[0]], [fit_levs[1]], [fit_levs[2]], [fit_levs[3]]]) # mapped to four binary variables
print ohe.transform([le.transform(['sunny'])]).toarray()
print ohe.transform([le.transform(['cloudy'])]).toarray()
