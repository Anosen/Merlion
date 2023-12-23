import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# thresholds_list = np.linspace(0,10,30)
# Define the desired range (0 to 10)
start = 0
end = 10
num_values = 30

# Calculate the mean and standard deviation to achieve the desired range
mean = (start + end) / 2
stddev = (end - start) / 4  # Using 1/4 of the range as the standard deviation

# Generate a list of random numbers following a Gaussian distribution
#thresholds_list = np.clip(np.random.normal(mean, stddev, 30), start, end)

# Generate a gaussian distributed list of thresholds
mean = 3.0  # Mean of the distribution
std_dev = 0.1  # Standard deviation of the distribution
# Generate equally spaced percentiles between 0 and 100
percentiles = np.linspace(0, 100, 100)

# Find the values at the specified percentiles from the Gaussian distribution
thresholds_list = np.percentile(np.random.normal(mean, std_dev, num_values), percentiles)

# Clip the values to the desired range (0 to 10)
#thresholds_list = np.clip(thresholds_list, start, end)

thresholds_list.sort()
plt.plot(thresholds_list)
plt.show()
plt.clf()