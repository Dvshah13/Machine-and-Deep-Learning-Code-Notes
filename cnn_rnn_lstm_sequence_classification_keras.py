# A LSTM (long-short term memory) network is a recurrent neural network that is trained using backpropagation or btt and overcomes the vanishing gradient problem.  It can be used to create large (stacked) recurrent networks, that in turn can be used to address difficult sequence problems in machine learning and achieve state-of-the-art results. Instead of neurons, LSTM networks have memory blocks that are connected into layers. A block has components that make it smarter than a classical neuron and a memory for recent sequences. A block contains gates that manage the block’s state and output. A unit operates upon an input sequence and each gate within a unit uses the sigmoid activation function to control whether they are triggered or not, making the change of state and addition of information flowing through the unit conditional.

# A convolutional neural network is a network which preserves the spatial structure of the problem and has three layers (convolutional layer, pooling layer and fully connected layer).  They were developed for object recognition tasks such as handwritten digit recognition and are popular because people are achieving state-of-the-art results on difficult computer vision and natural language processing tasks.

# LSTM and CNN for sequence classification in the IMDB dataset provided by Stanford, Keras already has built in access through the imdb.load_data() function allows you to load the dataset in a format that is ready for use in neural network and deep learning models

# We will map each movie review into a real vector domain, a popular technique when working with text called word embedding. This is a technique where words are encoded as real-valued vectors in a high dimensional space, where the similarity between words in terms of meaning translates to closeness in the vector space
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest constraining the dataset to the top 5,000 words and split the dataset into train (50%) and test (50%) sets
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad the input sequences so that they are all the same length for modeling. The model will learn the zero values carry no information so indeed the sequences are not the same length in terms of content but same length vectors is required to perform the computation in Keras
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# define CNN model whihc is used to pick out features for good and bad sentiment. This learned spatial features may then be learned as sequences by an LSTM layer. Here we add a one-dimensional CNN and max pooling layers after the Embedding layer which then feed the consolidated features to the LSTM. We can use a small set of 32 features with a small filter length of 3 with the relu activation function. The pooling layer can use the standard length of 2 to halve the feature map size.  
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
# define, compile and fit our LSTM model. First layer is the Embedded layer that uses 32 length vectors to represent each word. The next layer is the LSTM layer with 100 memory units. Because this is a classification problem we use a Dense output layer with a single neuron and a sigmoid activation function to make 0 or 1 predictions for the two classes (good and bad) in the problem. Since it is a binary classification problem, log loss is used as the loss function (binary_crossentropy in Keras). The efficient ADAM optimization algorithm is used. The model is fit for only 3 epochs because it quickly overfits the problem. A large batch size of 64 reviews is used to space out weight updates
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model, estimate the performance of the model on unseen reviews
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
