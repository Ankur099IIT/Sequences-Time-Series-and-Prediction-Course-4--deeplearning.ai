# In this lab i am able to build a model (single NN) model to predict the series next values
#First of all we created the same series for 4 years and 1 day including trend seasonality and noise
# then we divided the series into training and validation set by defining split size
# we have prepared features and labels after that.
# we are feeding the series in the batch of 32 with 20 (window_size) columns at a time and train the model
# further we have calculated the forecast value which is 20 less than the length of the series because it is not possible to forecast for first 20 values
# later on we sliced our forecast list to come in shape with the validation set so that we can compare and get some metrics
# forecast is a list so we need to convert it into array and reduce its dimension accordingly (x_valid)

import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt

def plot_series(time, series, format='-', start=0, end=None):

    plt.figure(figsize=(10,6))

    if type(series) is tuple:
        for series_num in series:
            plt.plot(time[start:end], series_num[start:end], format)
    else:
        plt.plot(time[start:end], series[start:end], format)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

def trend(time, slope):
    trend = time * slope
    return trend

def seasonal_pattern(time_season):

    data_pattern = np.where(time_season<0.4, np.cos(time_season*2*np.pi), 1/np.exp(3*time_season))

    return data_pattern

def seasonality(time, period, amplitude=1):

    time_season = (time%period)/period
    data_pattern = seasonal_pattern(time_season) * amplitude

    return data_pattern

def noise(time, noise_level, seed):
    rnd = np.random.RandomState(seed)

    noise = rnd.randn(len(time)) * noise_level

    return noise

# Generating the synthetic data

time = np.arange(4*365 +1, dtype = 'float32')
baseline = 10
amplitude = 40
slope = 0.05
noise_level= 5

# Create the series

series = baseline + trend(time, slope)+ seasonality(time, period = 365, amplitude=amplitude) + noise(time, noise_level, seed=42)

plot_series(time, series)

split_time = 1000

time_train = time[:split_time]
x_train = series[:split_time]

time_valid = time[split_time:]
x_valid = series[split_time:]

plot_series(time_train, x_train)
plot_series(time_valid, x_valid)

# Prepare features and labels

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000


def windowed_dataset(series, window_size, batch_size, shuffle_buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)

    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)

    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))

    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.batch(batch_size).prefetch(1)  # group your windows into batches # prefetch optimizes the execution time by prefetch the next one batch in advance

    return dataset

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

# Print properties os a single batch
for windows in dataset.take(1):
    print(f'data type: {type(windows)}')
    print(f'number of elements in tuple: {len(windows)}')
    print(f'shape of first element: {windows[0].shape}')
    print(f'shape of first element: {windows[1].shape}')


# Build and compile the model

layer = tf.keras.layers.Dense(1, input_shape = [window_size])

model = tf.keras.Sequential([layer])

print('layers weight: \n {} \n'.format(layer.get_weights()))
model.summary


model.compile(loss = 'mse',
              optimizer = tf.keras.optimizers.SGD(learning_rate=1e-6, momentum = 0.9))

model.fit(dataset, epochs = 100)

print('layer weights {}'.format(layer.get_weights()))

# Model Prediction

print(f'shape of series[:20]: {series[:20].shape}')

print(f'shape of series[:20][np.newaxis]: {series[:20][np.newaxis].shape}') # adding a batch dimension

print(f'shape of series[:20][np.newaxis]: {np.expand_dims(series[:20], axis = 0).shape}') # alternative to add batch dimension

print(f'model prediction: {model.predict(series[:20][np.newaxis])}')

forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time+window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]

print(f'length of the forecast list: {len(forecast)}')
print(f'shape of the validation set: {x_valid.shape}')

print(f'shape after converting to numpy array: {np.array(forecast).shape}') # converting the list to numpy array (for compatibility with plot_series function as x_valid is an array and forecast is a list)
print(f'shape after squeezing: {np.array(forecast).squeeze().shape}') # reducing one dimension (batch wala)

results = np.array(forecast).squeeze()

plot_series(time_valid, (x_valid, results))

# metrics calculation

print(tf.keras.metrics.mean_squared_error(x_valid, results).numpy())
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())












