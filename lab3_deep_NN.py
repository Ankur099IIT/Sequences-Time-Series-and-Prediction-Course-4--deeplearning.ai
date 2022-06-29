# Deep NN is used to train the model
# Auto Tune the Learning rate and plotted Loss vs Learning rate and chose the optimum learning rate and retrain the model (model_tune) and forecast again and hence comapre the metrics with x_valid


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

# Build the model
model_baseline = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape = [window_size], activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dense(1)
])

model_baseline.summary()

model_baseline.compile(loss = 'mse',
              optimizer = tf.keras.optimizers.SGD(learning_rate=1e-6, momentum = 0.9))

model_baseline.fit(dataset, epochs = 100)

forecast = []

forecast_series = series[split_time-window_size:]

for time in range(len(forecast_series)-window_size):
    forecast.append(model_baseline.predict(forecast_series[time:time + window_size][np.newaxis]))

results = np.array(forecast).squeeze()

plot_series(time_valid, (x_valid, results))

print(tf.keras.metrics.mean_squared_error(x_valid, results).numpy())
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())

# Tune the learning rate

model_tune = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape = [window_size], activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dense(1)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch/20)
)

model_tune.compile(loss = 'mse',
              optimizer = tf.keras.optimizers.SGD(momentum = 0.9))

history = model_tune.fit(dataset, epochs = 100, callbacks = [lr_schedule])

# Plot loss vs learning rate

lrs = 1e-8 * (10**(np.arange(100)/20))

plt.figure(figsize=(10,6))
plt.grid(True)

plt.semilogx(lrs, history.history['loss'])
plt.tick_params('both', length = 10, width=1, which = 'both')
plt.axis([1e-8, 1e-3, 0, 300])


model_retune = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape = [window_size], activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dense(1)
])

model_retune.summary()

model_retune.compile(loss = 'mse',
              optimizer = tf.keras.optimizers.SGD(learning_rate=4e-6, momentum = 0.9))

history = model_retune.fit(dataset, epochs = 100)


forecast = []

forecast_series = series[split_time-window_size:]

for time in range(len(forecast_series)-window_size):
    forecast.append(model_retune.predict(forecast_series[time:time + window_size][np.newaxis]))

results = np.array(forecast).squeeze()

plot_series(time_valid, (x_valid, results))

print(tf.keras.metrics.mean_squared_error(x_valid, results).numpy())
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())

