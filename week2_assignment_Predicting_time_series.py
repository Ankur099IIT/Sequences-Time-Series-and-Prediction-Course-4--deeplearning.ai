# Adding a bit of my understanding here: so there 1461 elements in series. and we are creating a window of 21 elements each.
# so wee will be having 1441 sets of window (each having 21 elements)
# Now as we split each window into features (window_size=20 elements) and labels (1 element)
# Also we have defined the batch size = 32 so feature shape (32, 20) and labels shape (32,) and there will be total of 45 such sets for this series and one still left making the total of 1441 sets of 21


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_series(time, series, format, start, end):
    plt.figure(figsize=(10,6))

    if type(series) is tuple:
        for series_num in series:
            plt.plot(time[start:end], series_num[start:end], format)
    else:
        plt.plot(time[start:end], series[start:end], format)

    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

def trend(time, slope):
    trend = time * slope
    return trend

def seasonal_pattern(time_season):
    data_pattern = np.where(time_season<0.1,
                            np.cos(time_season * 6 * np.pi),
                            2/np.exp(9*time_season))
    return data_pattern

def seasonality(time, period, amplitude):
    time_season = (time%period)/period
    data_pattern = seasonal_pattern(time_season)*amplitude
    return data_pattern

def noise(time, noise_level, seed):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

# Generating the synthetic data

time = np.arange(4 * 365 + 1, dtype = 'float32')
baseline = 10
amplitude = 50
slope = 0.005
noise_level= 3
period=365
seed=42
format = '-'
start = 0
end = len(time)
split_time = 1100
window_size = 20
shuffle_buffer = 1000
batch_size = 32

# Create the series

series = baseline + trend(time, slope) + seasonality(time, period, amplitude) + noise(time, noise_level, seed)

plot_series(time, series, format, start, end)

def train_val_split(time, series, split_time):
    time_train = time[:split_time]
    x_train = series[:split_time]

    time_valid = time[split_time:]
    x_valid = series[split_time:]
    return time_train, x_train, time_valid, x_valid

time_train, x_train, time_valid, x_valid = train_val_split(time, series, split_time)
plot_series(time_train, x_train, format, start, end)
plot_series(time_valid, x_valid, format, start, end)

# Processing the data
def windowed_dataset(series, window_size, shuffle_buffer, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size = window_size+1, shift = 1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

dataset = windowed_dataset(series, window_size, shuffle_buffer, batch_size)

# for dataset visualization
for x, y in dataset:
    print('x = ', x.numpy())
    print('y = ', y.numpy())
    print()

# Defining the model architecture
def create_model(window_size):

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape = [window_size], activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss = 'mse',
                  optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-6, momentum = 0.9))

    return model

dataset = windowed_dataset(x_train, window_size, shuffle_buffer, batch_size)

model = create_model(window_size)

model.fit(dataset, epochs = 100) # dataset contains both features and labels


# Evaluating the forecast
def generate_forecast():
    forecast = []

    for time in range(len(series)-window_size):
        forecast.append(model.predict(series[time:time+window_size][np.newaxis]))

    forecast = forecast[split_time-window_size:]
    results = np.array(forecast).squeeze()
    return results

results = generate_forecast()
plot_series(time_valid, (x_valid, results), format, start, end)

print('MSE: ', tf.keras.metrics.mean_squared_error(x_valid, results).numpy())
print('MAE: ',tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())
