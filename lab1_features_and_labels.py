# So overall I have learned:
# 1) Create dataset using tensorflow
# 2) Window creation (with size and shift specified)
# 3) How to visualize the windows
# 4) drop_remainder, flat_map(removing ',')
# 5) Creating features and labels, shuffle, batching


import tensorflow as tf

dataset = tf.data.Dataset.range(10)

for val in dataset:
    print(val.numpy())

dataset = dataset.window(size = 5, shift = 1)
for window_dataset in dataset:
    print(window_dataset)

for window_dataset in dataset:
    print([item.numpy() for item in window_dataset])


# Use of drop remainder
dataset = tf.data.Dataset.range(10)

dataset = dataset.window(size = 5, shift = 1, drop_remainder=True)

for window_dataset in dataset:
    print([item.numpy() for item in window_dataset])



# Flatten the windows (basically removing the " , " between the numbers)

dataset = tf.data.Dataset.range(10)

dataset = dataset.window(size=5, shift=1, drop_remainder=True)

dataset = dataset.flat_map(lambda window: window.batch(5))

for window in dataset:
    print(window.numpy())


# Group into features and labels
dataset = tf.data.Dataset.range(10)

dataset = dataset.window(size=5, shift=1, drop_remainder=True)

dataset = dataset.flat_map(lambda window: window.batch(5))

dataset = dataset.map(lambda window: (window[:-1], window[-1])) # create tuples with features and labels

for x, y in dataset:
    print('x = ', x.numpy())
    print('y = ', y.numpy())
    print() # this is for creating an empty line


# Shuffle the data

dataset = tf.data.Dataset.range(10)

dataset = dataset.window(size=5, shift=1, drop_remainder=True)

dataset = dataset.flat_map(lambda window: window.batch(5))

dataset = dataset.map(lambda window: (window[:-1], window[-1]))

dataset = dataset.shuffle(buffer_size = 10) # buffer size should be equal to or greater than the number of windows

for x, y in dataset:
    print('x = ', x.numpy())
    print('y = ', y.numpy())
    print()


# Create batches for training

dataset = tf.data.Dataset.range(10)

dataset = dataset.window(size=5, shift=1, drop_remainder=True)

dataset = dataset.flat_map(lambda window: window.batch(5))

dataset = dataset.map(lambda window: (window[:-1], window[-1]))

dataset = dataset.shuffle(buffer_size = 10)

dataset = dataset.batch(2).prefetch(1) # group your windows into batches # prefetch optimizes the execution time by prefetch the next one batch in advance

for x, y in dataset:
    print('x = ', x.numpy())
    print('y = ', y.numpy())
    print()



