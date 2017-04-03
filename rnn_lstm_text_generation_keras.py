# A LSTM (long-short term memory) network is a recurrent neural network that is trained using backpropagation or btt and overcomes the vanishing gradient problem.  It can be used to create large (stacked) recurrent networks, that in turn can be used to address difficult sequence problems in machine learning and achieve state-of-the-art results. Instead of neurons, LSTM networks have memory blocks that are connected into layers. A block has components that make it smarter than a classical neuron and a memory for recent sequences. A block contains gates that manage the block’s state and output. A unit operates upon an input sequence and each gate within a unit uses the sigmoid activation function to control whether they are triggered or not, making the change of state and addition of information flowing through the unit conditional.

# Here is an example of a simple LSTM Recurrent Neural Network to learn sequences of characters from Sherlock Holmes, "A Scandal in Bohemia" and generate a new sequence of characters.

# import dependencies
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load ascii text and covert to lowercase to reduce vocabulary network must learn
filename = "/Data Sets for Code/sherlock_asib.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# prepare the data for modeling by the neural network. Since we cannot model the characters directly, we must convert the characters to integers. We can do this by first creating a set of all of the distinct characters in the book, then creating a map of each character to a unique integer
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the data set
n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab

# prepare the data set of input to output pairs encoded as integers and define the training data for the network. Here we will split the book text up into subsequences with a fixed length of 100 characters, an arbitrary length. We could also split the data up by sentences and pad the shorter sequences and truncate the longer ones. Each training pattern of the network is comprised of 100 time steps of one character (X) followed by one character output (y). When creating these sequences, we slide this window along the whole book one character at a time, allowing each character a chance to be learned from the 100 characters that preceded it (except the first 100 characters of course)
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns

# transform the list of input sequences into the form [samples, time steps, features] expected by an LSTM network.
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# rescale the integers to the range 0-to-1 to make the patterns easier to learn by the LSTM network that uses the sigmoid activation function by default
X = X / float(n_vocab)

# convert the output patterns (single characters converted to integers) into a one hot encoding. This is so that we can configure the network to predict the probability of each of different characters in the vocabulary
y = np_utils.to_categorical(dataY)

# define a single hidden LSTM layer with 256 memory units. The network uses dropout with a probability of 20. The output layer is a Dense layer using the softmax activation function to output a probability prediction for each of the characters between 0 and 1. The problem is really a single character classification problem and as such is defined as optimizing the log loss (cross entropy), here using the ADAM optimization algorithm for speed.
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# the network is slow to train, a characteristic of all lstm networks, because of this and our optimization requirements, we will use model checkpoints to record all of the network weights to file each time an improvement in loss is observed at the end of the epoch
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model, using 20 epochs and  batch size of 128 patterns
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

### Now to generate the text ###
# load the network weights
# filename = "checkpoint_filepath" # this is the network weights, loaded from a checkpoint file, first train then run
# model.load_weights(filename)
# model.compile(loss='categorical_crossentropy', optimizer='adam')
# # pick a random seed, the easiest way to use the Keras LSTM model to make predictions is to first start off with a seed sequence as input, generate the next character then update the seed sequence to add the generated character on the end and trim off the first character. This process is repeated for as long as we want to predict new characters  We can pick a random input pattern as our seed sequence, then print generated characters as we generate them.
# start = numpy.random.randint(0, len(dataX)-1)
# pattern = dataX[start]
# print "Seed:"
# print "\"", ''.join([int_to_char[value] for value in pattern]), "\""
# # generate characters, create a reverse mapping that we can use to convert the integers back to characters so that we can understand the predictions
# for i in range(1000):
# 	x = numpy.reshape(pattern, (1, len(pattern), 1))
# 	x = x / float(n_vocab)
# 	prediction = model.predict(x, verbose=0)
# 	index = numpy.argmax(prediction)
# 	result = int_to_char[index]
# 	seq_in = [int_to_char[value] for value in pattern]
# 	sys.stdout.write(result)
# 	pattern.append(index)
# 	pattern = pattern[1:len(pattern)]
# print "\nDone."
