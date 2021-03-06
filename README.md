# Machine and Deep Learning Code and Notes
#### Code and Notes on Machine and Deep Learning Algorithms/Techniques using various python libraries. 
#### Credit must be given to Python Machine Learning by Sebastian Raschka, Machine Learning Mastery by Jason Brownlee, Scikit Learn Documentation, Introduction to Machine Learning with Python by Andreas C Muller and Sarah Guido, Python Data Science Essentials by Alberto Boschetti and Luca Massaron, Building Machine Learning System by Willi Richert and Luis Pedro Coelho, Tensorflow Site and Documentation.  These resources were instrumental in my learning process.
### Please Note:  Most of the python scripts are labeled with algorithm(s) and libraries used.  Notes and my thought process are commented at the top and between code.  Data sets can be found in MNIST and Data Sets for Code folder (provided by UCI data repository, Kaggle and Project Gutenburg).  Table of contents will indicate which algorithm(s), description, libraries and data set is being used for algorithm.
### Table of Contents
#### Data Munging Techniques, Transformation, Analysis, Visualization and Overall Evaluation
1. data_munging_basics-loading-data.py - Showing how to load, extract information and checking shape of datasets using various methods (Pandas, NumPy) and dealing with issues that may arise such as missing data, bad lines, parsing through dates, dealing with larger data sets
2. data_loading_storage_file_formats.py - Examples of loading data from JSON Data, XML/HTML Web Scrapping, Binary Data Formats and HDF5, Web APIs and Databases (SQL and no-SQL)
3. data_munging_data-preprocessing.py - Going through preprocessing steps, including mask function, mask target to change label and getting information grouped in different ways
4. data_munging_basics-categorical.py - More preprocessing steps this time with categorical data using Pandas and Scikit Learn to map categorical data into binary features
5. data_munging_basics-text.py - Preprocessing steps with text data, getting word counts, lists, frequencies, n-grams and hashing.  Using Scikit learn's CountVectorizer, Tfidf, HashingVectorizer
6. data_munging-numpy-preprocessing.py - Preprocessing NumPy ndarray's so they can them be used in ML functions with various python libraries.  Processes touched on include creating NumPy arrays, controlling memory size, working with heterogeneous lists, converting lists to multidimensional arrays, resizing arrays, arrays derived from NumPy functions, getting an array directly from a file, extracting data with Pandas, NumPy operation and computations, matrix operations, slicing and indexing NumPy arrays and stacking NumPy arrays
7. data_exploratory-data-analysis.py - Starting the data science process, showing how to understand the dataset better, check its features and shape, through the data and visually by using different plotting methods and getting a preliminary idea about the next steps to take
8. data_feature-creation.py - Utilizing a built in housing dataset that is a regression problem, shows by using a simple regressor and calculating mean absolute error (MAE) how different feature creation techniques and transforms can minimize the MAE error and yield results that are satisfactory
9. data_dimensionality-reduction.py - Examples using a variety of dimensionality reduction techniques found within Scikit-Learn, including the coraviance matrix, principle component analysis (PCA), randomized PCA, latent factor analysis, linear discriminant analysis, latent semantical analysis, independent component analysis, kernal PCA and restricted boltzmann machine
10. data_outliers.py - Dealing with the detection and treatment of outliers, methods include univariate outlier detection, EllipticEnvelope and OneClassSVM
11. evaluation_scoring-functions.py - Examples of various scoring functions you can use to evaluate performance of the system used.  Includes metrics and measures used for multilabel classification, binary classification and regression such as confusion matrix, accuracy, precision, recall, F1 score, ROC/AUC, mean absolute error (MAE), mean squared error (MSE) and R2 score
12. evaluation_testing-validation.py - Validation and testing best practices, including basic primer on cross validation
13. evaluation_cross-validation.py - Cross-validation in detail, including using various iterators to build your own cross-validation processes
14. evaluation_sampling-bootstrapping.py - Utilizing both sampling and bootstrap techniques as variations on cross-validation to improve model validation 
15. evaluation_hyper-parameters-optimization.py - Extending cross-validation to find the best hyper-parameters to generalize our test dataset
16. evaluation_custom-scoring-function-grid-search-runtime.py - Building custom scoring functions when the predefined functions may not perform well in your domain and how to reduce grid search runtime in instances where the data or grid search space is large by using RandomizedGridSearchCV
17. data_evaluation_feature-selection.py - Demonstrating various techniques of feature selection including, univariate selection, recursive elimination, stability and L1-based selection
18. data_wrangling_transform.py - Different methods of transforming data and discretization and binning examples
#### Data Science Libraries Basic Functionality
1. data_pandas_series_primer.py - Different use cases with Pandas Series that typically come up in the Data Science Pipeline
2. data_pandas_dataframe_primer.py - Different use cases with Pandas DataFrame that typically come up in the Data Science Pipeline
3. data_pandas_essential_features - Overview in code of essential features utilized in Pandas throughout the Data Science Pipeline
#### Machine Learning Scripts
1. linear_regression_sgd.py - Linear Regression with Stoichastic Gradient Descent using the red_wine_quality.csv data set (provided by UCI data repository)
2. logistic_regression_sgd.py - Logistic Regression with Stoichastic Gradient Descent using the diabetes.csv data set (provided by UCI data repository)
3. decision_tees.py - Decision Tree using the signals_sonar_classify.csv data set (provided by UCI data repository)
4. decision_trees_xg_boosting.py - Demonstration of XG Boosting and plotting with scikit learn, pandas, numpy and matplotlib using the train_xgboost.csv (provided by Kaggle)
5. bagging_decision_trees.py - Demonstration of bagging and bootstrap aggregation using the signals_sonar_classify.csv data set (provided by UCI data repository)
6. random_forests.py - Demonstration of the random forest algorithm using the signals_sonar_classify.csv data set (provided by UCI data repository)
7. naive_bayes.py - Naive Bayes algorithm using the diabetes.csv data set (provided by UCI data repository)
8. k-nearest_neighbors.py - K-Nearest Neighbors algorithm using the iris.data classic data set
9. basic_k-means.py - K-Means clustering demonstrated using the faithful.csv data set 
10. k-means.py - K-Means clustering demonstration using scikit learn module data set
11. k-means++.py - Demonstration of K-Means++ algorithm building upon k-means.py
12. k-means++_elbow_silhouette.py - Building upon concepts of K-Means++, demonstrating methods to estimate optimal number of clusters (elbow) and evaluation of the quality of clustering (silhouette)
13. hierarchical_clustering.py - Demonstration of alternative type of clustering, hierarchical clustering with scipy
14. basic_perceptron.py - Perception algorithm using the signals_sonar_classify.csv data set (provided by UCI data repository)
15. classification_tensorflow.py - Classification with tensorflow using the MNIST data set filtered and obtained with the input_data.py file
16. ex1_tensorflow.py - Basic file outlining structure (weights, biases, tensorflow structure) using tensorflow
#### Deep Learning Scripts
1. basic_imageclass.py - Image Classification using Convolutional Neural Network with tensorflow using the MNIST data set filtered and obtained with the input_data.py file
2. basic_ann_tensforflow.py - First neural network using tensorflow, numpy and plotted with matplotlib
3. optimizers_notes_tensorflow.py - Notes on using optimizers in tensorflow
4. overfitting_tensorflow.py - Dealing with overfitting data using the dropout function in tensorflow
5. placeholders_tensorflow.py - How to use placeholder variables and pass values on after output
6. cnn_rnn_lstm_sequence_classification_keras.py - Convolutional Neural Net with LSTM Recurrent Neural Net for sequence classification using the IMDB movie review data set which is accessed using a built in Keras function
7. rnn_lstm_text_generation_keras.py - LSTM Recurrent Neural Net text generation using keras and sherlock_asib.txt data set (provided by Project Gutenburg) 
8. rnn_lstm_time_series_keras.py - LSTM Recurrent Neural Net time series prediction with keras, pandas, scikit learn, numpy and matplotlib using DJIA.csv data set
9. rnn_classification_tensorflow - Classification with Recurrent Neural Net using tensorflow and MNIST data set
10. rnn_regression_tensorflow - Regression with Recurrent Neural Net using tensorflow and MNIST data set
11. save_restore_tensorflow - Saving and restoring variables using tensorflow
12. session_init_tensorflow - Basic session initialization using tensorflow
13. variables_tensorflow - dealing with and initializing variables using tensorflow
14. visualize_net_basic_tensorflow.py - basic version of visualizing network using tensorflow's built in tool called tensorboard
15. visualizing_net_added_features_tensorflow.py - includes some added features, visualizing network using tensorflow's built in tool called tensorboard
#### Data Munging Data Sets
1. istanbul_market_data.csv - Data collected from imkb.gov.tr and finance.yahoo.com and available on UCI data set repository, contains data from Istanbul's stock exchange and other global exchanges 
2. istanbul_data_bad.csv - Using a smaller version of istanbul_market_data deleted data from data set and used to show how to deal with this situation
3. istanbul_data_bad_lines.csv - Using a smaller version of istanbul_market_data deleted data from data set and used to show how to deal with this situation
4. istanbul_market_excel.xlsx - istanbul_market_data in excel format to show how to read varying file formats
5. istanbul_market_nodates_nolabels.csv - istanbul_market_data without dates and labels used in extracting data from Pandas and converting into NumPy array

#### Data Sets for Model
1. MNIST - Developed by Yann LeCun, Corinna Cortes, Google Labs, Christopher J.C. Burges, contains a database of handwritten digits, a training set of 60,000 examples, and a test set of 10,000 examples. 
2. red_wine_quality.csv - Developed by Paulo Cortezo, A. Cerdeira, F. Almeida, T. Matos and J. Reis, contains twelve attributes including wine quality score from 0-10
3. diabetes.csv - Developed by National Institute of Diabetes and Digestive and Kidney Diseases, contains nine attributes including class variable 0-1 (0 negative and 1 positive for diabetes)
4. signals_sonar_classify.csv - Developed by Terry Sejnowski, each pattern is a set of 60 numbers in the range 0.0 to 1.0. Each number represents the energy within a particular frequency band, integrated over a certain period of time
5. train_xgboost.csv - Developed by the Otto Group, describes the 93 obfuscated details of more than 61,000 products grouped into 10 product categories
6. iris.data - Developed by R.A. Fisher, contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other
7. faithful.csv - Developed by Azzalini and Bowman, containing Old Faithful Geyser Data in two attributes, eruptions - eruption time in minutes and waiting - waiting time to next eruption also in minutes
8. sherlock_asib.txt - Written by Sir Arthur Conan Doyle and prepared by Project Gutenburg, contains text for A Scandal in Bohemia
9. DJIA.csv - Dow Jones Industrial Average over a period of 10 years, only attributes considered are date and price
#### Notes 
1. Deep Learning Notes.pdf - More simplified
2. DeepLearning_Notes_Detailed.pdf - More detailed
3. Data Munging Notes - Detailed set of notes on data munging techniques and practices 
4. Adding spark notes here
5. Adding hadoop notes here
