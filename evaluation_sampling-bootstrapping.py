# Sampling - different from cross-validation in that you don't split the training set, but you subsample or bootstrap it.
# Subsampling - performed when you randomly select a part of the available data, obtaining a smaller dataset than the initial one.  It's very useful when you need to extensively test your hypothesis but you prefer not to obtain your validation from extremely small test samples (so, you can opt out of a leave-one-out approach or a KFold using a large number of folds):
from sklearn import cross_validation
# subsampling = cross_validation.ShuffleSplit(n = 100, n_iter = 10, test_size = 0.1, random_state = 1) # the n parameter, instructs the iterator to perform the folding on 100 indexes, n_iter will set the number of subsamples, test_size the percentage (if a float is given) or the number of observations to be used as a test
# for train_idx, validation_idx in subsampling:
    # print train_idx, validation_idx

# Bootstrap, as a resampling method, works randomly - choosing observations and allowing repetitions - until a new dataset, which is the same size as the original one is built.  But since bootstrapping works by sampling with replacement (allowing repetition of the same observation), there are some issues that may arise such as: cases that may appear both on the training and test set (you just have to use out-of-bootstrap sample observations for test purposes) and there is less variance and more bias than the cross-validation estimations due to nondistinct observations resulting from sampling with replacement.
# Example of bootstrap, which is called by a for loop.  It generates a sample bootstrap of the same size as the input data (the length of the indexes) and a list of the excluded indexes (out of sample) that could be used for testing purposes:
import random
def Bootstrap(n, n_iter = 3, random_state = None):
    # Random sampling with replacement cross-validation generator.  For each iter a sample bootstrap of the indexes [0, n] is generated and the function returns the obtained sample and a list of all the excluded indexes
    if random_state:
        random.seed(random_state)
    for j in range(n_iter):
        bts = [random.randint(0, n-1) for i in range(n)]
        out_bts = list({i for i in range(n)} - set(bts))
        yield bts, out_bts

boot = Bootstrap(n = 100, n_iter = 10, random_state = 1) # function performs subsampling and accepts the parameter n for the n_iter index to draw the bootstrap and the random_state index for repeatability
for train_idx, validation_idx in boot:
    print train_idx, validation_idx
