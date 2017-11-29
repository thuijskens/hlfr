import numpy as np

def _compute_tpr(confusion_matrix):
    return confusion_matrix.tp / (confusion_matrix.tp + confusion_matrix.fn)


def _compute_tnr(confusion_matrix):
    return confusion_matrix.tn / (confusion_matrix.tn + confusion_matrix.fp)


def _compute_ppv(confusion_matrix):
    return confusion_matrix.tp / (confusion_matrix.tp + confusion_matrix.fp)


def _compute_npv(confusion_matrix):
    return confusion_matrix.tn / (confusion_matrix.tn + confusion_matrix.fn)


METRICS_FUNCTION_MAPPING = {
    'tpr': _compute_tpr,
    'tnr': _compute_tnr,
    'ppv': _compute_ppv,
    'npv': _compute_npv
}


class StreamingConfusionMatrix(object):
    def __init__(self):
        self.confusion_matrix = np.ones((2, 2))
        self.tp = 1
        self.fp = 1
        self.tn = 1
        self.fn = 1

    def update_confusion_matrix(self, y_true, y_pred):
        self.confusion_matrix[y_pred, y_true] += 1
        self._update_statistics()
        return self.confusion_matrix

    def reset_internals(self):
        self.confusion_matrix = np.ones((2, 2))
        self.tp = 1
        self.fp = 1
        self.tn = 1
        self.fn = 1

    def _update_statistics(self):
        self.tp = self.confusion_matrix[1, 1]
        self.fp = self.confusion_matrix[1, 0]
        self.tn = self.confusion_matrix[0, 0]
        self.fn = self.confusion_matrix[0, 1]


class PerformanceMetric(object):
    def __init__(self, metric_name, decay):
        """
        decay is a function

        :param metric_name: 
        :param decay: 
        """
        if metric_name not in ['tpr', 'tnr', 'ppv', 'npv']:
            raise ValueError('metric_name must be one of tpr, tnr, ppv, or npv, got %s' % metric_name)

        self.metric_name = metric_name
        self.decay = decay
        self.metric_value = [0.5]

        self._R = [0.5]
        self._P = [0.5]

    def reset_internals(self):
        self._R[-1] = 0.5
        self._P[-1] = 0.5

    def update_metric(self, confusion_matrix, y_true, y_pred):

        self.metric_value.append(METRICS_FUNCTION_MAPPING[self.metric_name](confusion_matrix))

        r_hat = self.update_decay(y_true, y_pred)
        n, p_hat = self.update_stats(confusion_matrix.confusion_matrix)

        return n, p_hat, r_hat

    def metric_changed(self):
        return abs(self.metric_value[-1] - self.metric_value[-2]) > 0

    def update_decay(self, y_true, y_pred):
        if self.metric_changed():
            self._R.append(self.decay * self._R[-1] + (1 - self.decay) * int(y_true == y_pred))
        else:
            self._R.append(self._R[-1])

        return self._R[-1]

    def update_stats(self, confusion_matrix):
        if self.metric_name in ['tpr', 'tnr']:
            tpr_indicator = int(self.metric_name == 'tpr')
            n = int(confusion_matrix[:, tpr_indicator].sum())

            self._P.append(confusion_matrix[tpr_indicator, tpr_indicator] / n)
        else:
            ppv_indicator = int(self.metric_name == 'ppv')
            n = int(confusion_matrix[ppv_indicator, :].sum())

            self._P.append(confusion_matrix[ppv_indicator, ppv_indicator] / n)

        return n, self._P[-1]
