from hlfr import LinearFourRates, StreamingConfusionMatrix
import numpy as np
import matplotlib.pyplot as plt
import cProfile



T = 10000
changepoint = 5000
cp_1 = np.array([[0.4, 0.1], [0.1, 0.4]])
cp_2 = np.array([[0.3, 0.1], [0.2, 0.4]])

np.random.seed(1234)

# Generate sme synthetic data
possibilities = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
cp1 = [0.4, 0.1, 0.1, 0.4]
cp2 = [0.3, 0.1, 0.2, 0.4]
first_sample_idx = np.random.choice(len(possibilities), changepoint, replace=True, p=cp1)
second_sample_idx = np.random.choice(len(possibilities), changepoint, replace=True, p=cp2)

sampled_data = np.vstack((possibilities[first_sample_idx], possibilities[second_sample_idx]))
y_true = sampled_data[:, 0]
y_pred = sampled_data[:, 1]

K = 100
pr = cProfile.Profile()
pr.enable()
lfr = LinearFourRates(warn_level=0.01, detect_level=1 / (100.0 * K), decay=0.9)
lfr.detect_drift_points(y_obs=y_true[:20], y_pred=y_pred[:20])
pr.disable()
pr.print_stats(sort="calls")
print(lfr.concept_shift_times)

#plot out the four metrics for the simulated data
fig, ax = plt.subplots(1, 1)
time = np.arange(1, 22)
ax.plot(time, lfr.metrics['tpr'].metric_value, 'y-')
ax.plot(time, lfr.metrics['tnr'].metric_value, 'r-')
ax.plot(time, lfr.metrics['ppv'].metric_value, 'b-')
ax.plot(time, lfr.metrics['npv'].metric_value, 'c-')
fig.savefig('/Users/thomas.huijskens/code/hlfr/true-rates.png')


"""
fig, ax = plt.subplots(1, 1)
time = np.arange(1, T + 1)
for metric_name, metric in lfr.metrics.items():
    ax.plot(time, metric._P)
ax.set_xlabel('Time')
ax.set_ylabel('Test statistic')
"""