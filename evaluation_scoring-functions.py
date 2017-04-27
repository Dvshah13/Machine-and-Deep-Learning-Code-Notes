## Scoring functions - used to evaluate the performance of the system and to check how close you are to the objective that you have in mind by scoring the outcome.  Typically, different scoring functions are used to deal with binary classification, multilabel classification, regression or a clustering problem.

## multilabel classification - when your task is to predict more than a single label.  It's very popular and many performance metrics exist to evaluate classifiers.
# Example using iris data
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.50, random_state=4)
# use a very bad multiclass classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print iris.target_names
# measures used in multilabel classification
# Confusion Matrix - a table that gives us an idea about what the misclassifications are for each class, ideally in perfect classification all the cells that are not on the diagonal should be 0.
from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print cm
# Accuracy - portion of the predicted labels that are exactly equal to the real ones.  Percentage of correctly classified labels
print "Accuracy:", metrics.accuracy_score(Y_test, Y_pred)
# Precision - counts the number of relevant results in the result set.  In classification it counts the number of correct labels in each set of classified labels.
print "Precision:", metrics.precision_score(Y_test, Y_pred)
# Recall - counts the number of relevant results in the result set, compared to all the relevant labels in the dataset.  In classification tasks, that's the amount of correctly classified labels in the set divided by the total count of labels for that set.
print "Recall:", metrics.recall_score(Y_test, Y_pred)
# F1 Score - harmonic average of precision and recall
print "F1 Score:", metrics.f1_score(Y_test, Y_pred)
# There is a convenient method that shows a report on these measures through Scikit Learn.  It includes precision, recall, f1-score and support. Support is simply the number of observations with that label.  It's pretty useful to understand whether a dataset is balanced (that is, whether it has the same support for every class) or not.
from sklearn.metrics import classification_report
print classification_report(Y_test, Y_pred, target_names=iris.target_names)
# In practice, Precision and Recall are used more extensively than Accuracy as most datasets in Data Science tend to be unbalanced.  To account for this imbalance, data scientists often present their results in terms of Precision, Recall, and F1 tuple.
# Also Accuracy, Precision, Recall and F1 assume values in the [0.0, 1.0] range.  Perfect classifiers achieve the score of 1.0 for all these measures.

## Binary Classification - in addition to the ones metrics shown previously, in binary classification (where you only have two output classes), there are additional measures that can be used.  The most used, since it's very informative, is the area under the Receiver Operating Characteristics curve (ROC) or area under a curve (AUC).  The ROC is a graphical way to express how the performances of the classifier change over all the possible classification thresholds (that is, the change in outcomes when its parameters change).  Specifically, the performances have a true positive (or hit) rate, and a false positive (or miss) rate.  The first is the rate of the correct positive results, and the second is the rate of the incorrect ones. The area under that curve represents how well the classifier performs with respect to a random classifier (whose AUC is 0.50).  AUC closer to 1.0 is preferred.
# The function used to compute the AUC with Python is sklearn.metrics import roc_auc_score()

## Regression - here any of the measures to score are functions derived from Euclidean algebra.
# Mean Absolute Error or MAE - is the mean L1 (lasso) norm of the difference vector of the predicted and real values
from sklearn.metrics import mean_absolute_error
print mean_absolute_error([1.0, 0.0, 0.0], [0.0, 0.0, -1.0])
# Mean Squared Error or MSE - is the mean L2 (ridge) norm of the difference vector of the predicted and real values
from sklearn.metrics import mean_squared_error
print mean_squared_error([-10.0, 0.0, 0.0], [0.0, 0.0, 0.0])
# R2 Score - also known as the Coefficient of Determination, R2 determines how good a linear fit that exists between the predictors and the target variables.  It takes a value between 0 and 1 (inclusive); the higher it is, the better the model.  The function to use in this case is sklearn.metrics import r2_score
