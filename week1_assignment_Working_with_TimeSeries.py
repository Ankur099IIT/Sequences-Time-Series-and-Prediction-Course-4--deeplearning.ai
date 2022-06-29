import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def trend(time, slope):
    trend = time * slope
    return trend

def seasonality_series(time_season, period, amplitude):
    time_season = (time_season%period)/period
    seasonality_series = np.where(time_season<0.1, np.cos(time_season*7*np.pi),
                           1/np.exp(5*time_season))
    return seasonality_series*amplitude

def plot_series(time, series, format, x_label, y_label):
    plt.figure(figsize=(10,6))

    if type(series) is tuple:
        for series_num in series:
            plt.plot(time, series_num, format)
    else:
        plt.plot(time, series, format)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

def noise(time, noise_level = 1, seed = None):
    rnd = np.random.RandomState(seed)
    noise = rnd.randn(len(time)) * noise_level

    return noise

TIME = np.arange(4*365+1, dtype = 'float32')
y_intercept = 10
slope = 0.01


SERIES = trend(TIME, slope) + y_intercept + seasonality_series(time_season=TIME, period=365, amplitude=40) + noise(TIME, noise_level = 1, seed = None)

plot_series(TIME, SERIES, format = '-', x_label = 'Time', y_label = 'Value')

# Split the series

SPLIT_TIME = 1100

def train_val_split(time, series, time_step = SPLIT_TIME):
    time_train = time[:time_step]
    series_train = series[:time_step]
    time_valid = time[time_step:]
    series_valid = series[time_step:]

    return time_train, series_train, time_valid, series_valid

time_train, series_train, time_valid, series_valid = train_val_split(TIME, SERIES)
plot_series(time_train, series_train, format='-', x_label='training_time', y_label = 'training_value')
plot_series(time_valid, series_valid, format='-', x_label='validation_time', y_label = 'validation_value')

# Evaluation metrics

def compute_metrics(true_series, forecast):
    mse = tf.keras.metrics.mean_squared_error(true_series, forecast).numpy()
    mae = tf.keras.metrics.mean_absolute_error(true_series, forecast).numpy()

    return mse, mae

# Test your function

# Define some dummy series

zeros = np.zeros(5)
ones = np.ones(5)

mse, mae = compute_metrics(zeros, ones)
print(f'mse and mae values for the series of zeros and forecast series of ones is {mse} and {mae}')

mse, mae = compute_metrics(ones, ones)
print(f'mse and mae values for the series of ones and forecast series of ones is {mse} and {mae}')

# Forecasting

#Naive forecast

naive_forecast = SERIES[SPLIT_TIME-1: -1]


plot_series(time_valid, series_valid, format='-', x_label='time', y_label = 'validation_value')
plot_series(time_valid, naive_forecast, format='-', x_label='time', y_label = 'Naive_forcast')

mse, mae = compute_metrics(series_valid, naive_forecast)

print(f'mse and mae for naive forecasting is {mse} and {mae}')

# Moving Average

def moving_average(series, window_size):

    forecast = []

    for time in range(len(series)-window_size):
        forecast.append(series[time:time+window_size].mean())

    np_forecast = np.array(forecast)

    return np_forecast

print(f'whole series has {len(SERIES)}elements so the moving average forecast (np_forecast) should have {len(SERIES)-30} elements')

# Test your moving average function

moving_avg = moving_average(SERIES, window_size=30)
print(f'moving average forecast with whole series has shape {moving_avg.shape}\n')

# slice it so it matches the validation period

moving_avg = moving_avg[SPLIT_TIME-30:]
print(f'moving average forecast after slicing has shape{moving_avg.shape}\n')

plt.figure(figsize=(10,6))
plot_series(time_valid, series_valid, format = '-', x_label = 'time', y_label = 'valid_Series')
plot_series(time_valid, moving_avg, format = '-', x_label = 'time', y_label = 'moving_avg_sliced_Series')

# Differencing

diff_series = (SERIES[365:] - SERIES[:-365])
diff_time = TIME[365:]

print(f'whole series has {len(SERIES)} elements so the differencing should have {len(SERIES)-365}elements\n')
print(f'diff series has shape{diff_series.shape}\n')
print(f'X-cordinate of difference seeries has {diff_time.shape}')

plt.figure(figsize=(10,6))
plot_series(diff_time, diff_series, format='-', x_label = 'diff_time', y_label='diff_series')
plt.show()

# Great the trend and seasonality seem to be gone so now we can retry using the moving average. diff_series is the ground truth while
# diff_moving_avg is the prediction array. we wil slice these accordingly to the validation set time steps before comparing.

diff_moving_avg = moving_average(diff_series, window_size=50)
print(f'moving average forecast with diff series has shape{diff_moving_avg.shape}\n')

diff_moving_avg = diff_moving_avg[SPLIT_TIME-365-50:]
diff_series = diff_series[SPLIT_TIME-365:]

# Now, lets bring back the trend and seasonality

past_series = SERIES[SPLIT_TIME-365:-365]
print(f'past seriess has shape{past_series.shape}')

diff_moving_avg_plus_past = past_series + diff_moving_avg



mse_diff, mae_diff = compute_metrics(series_valid, diff_moving_avg_plus_past)
print(f'mse and mae for the diff_moving_avg_plus_past are {mse_diff} and {mae_diff}\n')



