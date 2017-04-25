## Having used Pandas commands to upload and preprocess your data in memory or in smaller batches, you'll have to work on it in order to prepare a suitable data matrix for your supervised and unsupervised learning procedures

## As best practice, divide the task between a phase of your work when your data is still heterogeneous (mix of numerical and symbolic values) and another phase when it is turned into a numeric table of data arranged in rows that represent your examples, and columns that contain the characteristics observed values of your examples, which are your variables.  In doing so, you'll have to wrangle between two key Python packages for scientific analysis, Pandas and NumPy, and their two pivotal data structures, DataFrame and ndarray.  The target data structure is a NumPy ndarry.

## NumPy offers an ndarray object class (n-dimensional array) that has the following attributes: 1. it is memory optimal, 2. allows faster linear algebra computations and element-wise operations, 3. is the data structure that critical libraries such as SciPy and Scikit-learn expect as an input for their functions.  They have some drawbacks such as: 1. they usually store only elements of a single, specific data type that you can define beforehand (there's a way to define complex data and heterogeneous data types, though they could be very difficult to handle for analysis purpose), 2. after they are initialized, their size is fixed.

## Basics of NumPy ndarray objects: since type (and memory space it occupies in terms of bytes) of an array should be defined from the beginning, the array creation procedure can reserve the exact memory space to contain all the data.  The access, modification and computation of the elements of an array are therefore quite fast, though this also consequenly implies that the array is fixed and cannot be structurally changed.  It is therefore important to understand that when we are viewing an array, we have called a procedure that allows us to immediately convert the data into something else (but the sourcing array has been unaltered), when we are copying an array, we are creating a new array with a different structure (thus occupying new fresh memory)

## Creating NumPy arrays, more than one way to creating NumPy array: 1. transforming an existing data structure into an array, 2. creating an array from scratch and populating it with default or calculated values, 3. uploading some data form a disk into an array...if you're going to transform an existing data structure, the odds are in favor of you working with a structured list or a Pandas DataFrame

## From list to unidimensional arrays
import numpy as np
# list_of_ints = [1,2,3]
# Array_1 = np.array(list_of_ints) # transforms a list into a uni-dimensional array
# print Array_1
# print Array_1[1] # can still use like a normal python list
# print type(Array_1) # will see numpy.ndarray
# print Array_1.dtype # will see int32
## controlling memory size, can calculate memory space that array is taking up
# print Array_1.nbytes
## in order to save memory, you can specify beforehand the type that best suits your array
# Array_1 = np.array(list_of_ints, dtype = 'int8')
# print Array_1.nbytes
## if an array has a type that you want to change, you can easily create a new array by casting a new specified type:
# Array_1b = Array_1.astype('float32')
# print Array_1b
# print Array_1b.dtype

## Heterogeneous lists, can be a bit more complex
# complex_list = [1,2,3] + [1.,2.,3.] + ['a','b','c']
# Array_2 = np.array(complex_list[:3]) # at first the input list is just ints
# print 'complex_list[:3]', Array_2.dtype
# Array_2 = np.array(complex_list[:6]) # now ints and floats
# print 'complex_list[:6]', Array_2.dtype
# Array_2 = np.array(complex_list) # now ints and floats and strings (all added)
# print 'complex_list', Array_2.dtype
# what we see is that float types prevail over int types and strings prevail over everything else, when using lists of different elements be sure and check dtype since later you might find it impossible to operate certain operations on your resulting array and incur an unsupported operand type error
# check to see if NumPy array is of the desired numeric type
# print isinstance(Array_2[0], np.number) # checking this will make sure we transformed all the variables into numeric ones

## Lists to multidimensional arrays
# a_list_of_lists = [[1,2,3],[4,5,6],[7,8,9]]
# Array_2d = np.array(a_list_of_lists) # transform a list into a bidimensional array
# print Array_2d
# print Array_2d[1,1] # you can call out single values with indices (one for row (axis 0) and one for column dimension (axis 1))

## three dimensional array
a_list_of_lists_of_lists = [[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]]
Array_3D = np.array(a_list_of_lists_of_lists)
print Array_3D
# to access single elements of a three-dimensional array, you just have to point our a tuple of three indices
print Array_3D[0,2,0] # access 5th element
