from hlfr import StreamingConfusionMatrix
import unittest


class TestStreamingConfusionMatrix(unittest.TestCase):

    def test_update(self):
        confusion_matrix = StreamingConfusionMatrix()
        confusion_matrix.update_confusion_matrix(1, 1)
        confusion_matrix.update_confusion_matrix(1, 0)

        self.assertEqual(confusion_matrix.tp, 2.0)
        self.assertEqual(confusion_matrix.fn, 2.0)

    def test_reset(self):
        confusion_matrix = StreamingConfusionMatrix()
        confusion_matrix.update_confusion_matrix(1, 1)
        confusion_matrix.update_confusion_matrix(1, 0)

        confusion_matrix.reset_internals()

        self.assertEqual(confusion_matrix.tp, 1.0)
        self.assertEqual(confusion_matrix.fp, 1.0)
        self.assertEqual(confusion_matrix.tn, 1.0)
        self.assertEqual(confusion_matrix.fn, 1.0)

if __name__ == '__main__':
    unittest.main()
