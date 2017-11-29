"""
Hierarchical Linear Four Rates 

HLFR is a concept drift detection algorithm that consists of two layers:

    1. Layer-1 hypothesis test that detects a potential drift point T.
    2. Layer-2 hypothesis test that confirms the potentiality of drift point T.
    
If a drift point T is confirmed, the underlying model can be reconfigurated and the procedure restarts.
"""
import numpy as np
import itertools
import math

from scipy.stats import bernoulli


def compute_bounds(p_hat, decay, n, alpha, n_sim=1000):
    bernoulli_samples = bernoulli.rvs(p_hat, size=n * n_sim).reshape(n_sim, n)
    # TODO: Check if shapes match

    #empirical_bounds = (1 - decay) * (bernoulli_samples * (n - np.arange(1, n + 1))).sum(axis=1)
    empirical_bounds = (1 - decay) * np.matmul(bernoulli_samples, decay ** (n - np.arange(1, n + 1)).reshape(n, 1)).sum(axis=1)
    lb, ub = np.percentile(empirical_bounds, q=[alpha * 100, (1 - alpha) * 100])

    return lb, ub


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side='left')
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]


class BoundsTable(object):
    def __init__(self, decay, alpha_range, p_hat_range, n_range, n_sim=1000):
        self.decay = decay
        self.alpha_range = alpha_range
        self.p_hat_range = p_hat_range
        self.n_range = n_range
        self.n_sim = n_sim

        self.bounds_table = {}

    def compute_bounds_table(self, rng_seed=123321):
        np.random.seed(rng_seed)
        grid = itertools.product(self.p_hat_range, self.n_range, self.alpha_range)
        # TODO: Is it safe to store dictionary keys as floating point values?
        self.bounds_table = {(p, n, alpha): compute_bounds(p_hat=p, alpha=alpha, decay=self.decay, n=n, n_sim=self.n_sim)
                             for (p, n, alpha) in grid}

        return self

    def lookup_bounds(self, p, n, alpha):
        # We assume here that n and alpha can be exactly matched
        p_nearest = find_nearest(self.p_hat_range, p)
        return self.bounds_table[(p_nearest, n, alpha)]


class LinearFourRates(object):
    """
    The way we code LFR will be that it takes the whole data set, and detects all potential drift points in the time
    series.
    """
    def __init__(self, decay, warn_level, detect_level, bounds_table=None):
        self.decay = decay
        self.warn_level = warn_level
        self.detect_level = detect_level
        self.bounds_table = bounds_table

        self.metrics = {metric_name: PerformanceMetric(metric_name, decay)
                        for metric_name in ['tpr', 'tnr', 'ppv', 'npv']}
        self.warn_time = 0
        self.confusion_matrix = StreamingConfusionMatrix()
        self.concept_shift_times = []

    def _compute_bounds_table(self, n_samples, rng_seed=123321):
        alpha_range = np.array([self.warn_level, self.detect_level])
        p_hat_range = np.arange(1, 100) / 100.0
        n_range = np.arange(2, n_samples + 1)

        self.bounds_table = BoundsTable(self.decay, alpha_range, p_hat_range, n_range, n_sim=self.n_sim)
        self.bounds_table.compute_bounds_table(rng_seed=rng_seed)

        return self.bounds_table

    def detect_drift_points(self, y_obs, y_pred):
        """
        Data API. This takes the model and the data, and computes the performance metrics and confusion matrix
        before passing these on to _detect_drift_point
        """
        n_samples = y_obs.shape[0]

        if self.bounds_table is None:
            self._compute_bounds_table(n_samples)

        for t, (y, y_hat) in enumerate(zip(y_obs, y_pred)):
            self.confusion_matrix.update_confusion_matrix(y, y_hat)

            warnings = []
            detections = []

            for metric in self.metrics.values():
                n, p_hat, r_hat = metric.update_metric(self.confusion_matrix, y, y_hat)

                lb_warn, ub_warn = self.bounds_table.lookup_bounds(p=p_hat, n=n, alpha=self.warn_level)
                lb_detect, ub_detect = self.bounds_table.lookup_bounds(p=p_hat, n=n, alpha=self.detect_level)

                warn_shift = (r_hat <= lb_warn) or (r_hat >= ub_warn)
                detect_shift = (r_hat <= lb_detect) or (r_hat >= ub_detect)

                # NOTE: lb = ub = 0.0. So r_hat >= ub_warn is satisfied... bounds table computation is wrong
                print("Sample %i: metric %s, R: %.3f, Warn LB: %.3f Warn UB: %.f, Detect LB: %.3f, Detect UB: %.3f, warn: %s detect: %s"
                      % (t, metric.metric_name, r_hat, lb_warn, ub_warn, lb_detect, ub_detect, warn_shift, detect_shift))

                warnings.append(warn_shift)
                detections.append(detect_shift)

            if any(warnings) and self.warn_time is None:
                self.warn_time = t
            elif all([not warning for warning in warnings]) and self.warn_time is not None:
                self.warn_time = None

            if any(detections):
                self.concept_shift_times.append(t)

                # Reset all metrics and confusion matrix
                self.confusion_matrix.reset_internals()
                self.warn_time = 0
                for metric in self.metrics.values():
                    metric.reset_internals()

            if t % 100 == 0:
                print("Sample %i" % t)

        return self
