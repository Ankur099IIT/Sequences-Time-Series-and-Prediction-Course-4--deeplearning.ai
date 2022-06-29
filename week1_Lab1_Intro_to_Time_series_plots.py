import matplotlib.pyplot as plt
import numpy as np

time = np.arange(365) # for 1 year
slope = 0.1

series = slope * time # for trend

plt.figure(figsize = (10,6))

plt.plot(time, series, '-')
plt.xlabel('Time')
plt.ylabel('Trend')
plt.grid(True)
plt.show()

time_season = np.arange(4*365+1) # for four years
period = 365 # repetition period
amplitude = 40 # just to remove decimal values to integer

season_time = (time_season%period)/period #period inside bracket takes care if time_seasom value exceeds 365 it return only remainder
# period at the denominator is to divide the time into various discrete points
data_pattern = np.where(season_time<0.4,
                        np.cos(season_time*2*np.pi),
                        1/np.exp(3*season_time))

data_pattern = amplitude*data_pattern

plt.figure(figsize = (10,6))

plt.plot(time_season, data_pattern, '-')
plt.xlabel('Extended_Time')
plt.ylabel('Seasonality')
plt.grid(True)
plt.show()

extended_series = slope*time_season # earlier series extended from 365 days to new time of 1461 days
slope_seasonality = extended_series + data_pattern # sum of extended slope series and seasonality
plt.plot(time_season, slope_seasonality, '-')
plt.xlabel('Extended_Time')
plt.ylabel('Seasonality+Trend')
plt.grid(True)
plt.show()

noise_level = 5  #amplitude of noise

rnd = np.random.RandomState(seed=42)

noise = rnd.randn(len(time_season)) * noise_level #generate random numbers

plt.plot(time_season, noise, '-')
plt.xlabel('Extended_Time')
plt.ylabel('Noise')
plt.grid(True)
plt.show()

complete_series = slope_seasonality + noise #this series include trend, seasonality and noise
plt.plot(time_season, complete_series, '-')
plt.xlabel('Extended_Time')
plt.ylabel('Trend+Seasonality+Noise')
plt.grid(True)
plt.show()

