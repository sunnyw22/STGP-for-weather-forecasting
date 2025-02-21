# STGP-for-weather-forecasting
Goal: Applying spatial-temporal Gaussian Process Regression for weather forecasting (data resides at the same altitude).

We use a separable kernel in space and time. We test our model using the WeatherBench data. 
# Checklist:
Main goals:
- [ ] 1. Apply Kalman-Filtering approach
- [ ] 2. Build our Kernel on a sphere that takes the spherical polar coordinates as input (theta, phi)
- [ ] 3. Detrend the seasonal data
Lower priority:
- [ ] 4. Plotting the distribution of hyperparams after training on split data sets (i.e. split the full training datasets into yearly data, and examine if there is a pattern for the hyperparams.)
- [ ] 5. Adding in wind velocity (weather deviations tend to propagate along the direction of the wind)
- [ ] 6. Points near the north poles are more "important" (unclear exactly what this means and how to exploit this)
