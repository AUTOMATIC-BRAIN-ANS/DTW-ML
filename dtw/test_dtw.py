"""
@author: Radoslaw Plawecki
"""

import unittest
import numpy as np
from dtw import DTW
from unittest.mock import patch


class TestDTW(unittest.TestCase):
    def setUp(self):
        x = [0, 2, 0, 1, 0, 0]
        y = [0, 0, 0.5, 2, 0, 1, 0]
        self.dtw = DTW(x, y)

    def test_fill_matrix(self):
        result = self.dtw.fill_matrix()
        expected_result = np.array([
            [0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
            [np.inf, 0, 0, 0.5, 2.5, 2.5, 3.5, 3.5],
            [np.inf, 2, 2, 1.5, 0.5, 2.5, 3.5, 5.5],
            [np.inf, 2, 2, 2, 2.5, 0.5, 1.5, 1.5],
            [np.inf, 3, 3, 2.5, 3, 1.5, 0.5, 1.5],
            [np.inf, 3, 3, 3, 4.5, 1.5, 1.5, 0.5],
            [np.inf, 3, 3, 3.5, 5, 1.5, 2.5, 0.5]
        ])
        np.testing.assert_array_equal(result, expected_result)

    def test_traceback(self):
        result = self.dtw.traceback()
        expected_result = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        np.testing.assert_array_equal(result, expected_result)

    def test_calc_alignment_cost_d_method(self):
        result = np.round(self.dtw.calc_alignment_cost(method='d-method'), 4)
        self.assertEqual(result, 0.0385)

    def test_calc_alignment_cost_td_method(self):
        result = np.round(self.dtw.calc_alignment_cost(method='td-method'), 3)
        self.assertEqual(result, 0.375)

    def test_calc_alignment_cost_c_method(self):
        result = np.round(self.dtw.calc_alignment_cost(method='c-method'), 3)
        self.assertEqual(result, 0.375)

    def test_calc_alignment_cost_error(self):
        with self.assertRaises(ValueError):
            self.dtw.calc_alignment_cost(method='N/A')

    def test_find_mean_alignment_cost(self):
        x = [0, 3, 6, 2, 4, 1, 1, 1, 1, 1, 9, 0]
        y = [0, 1, 4, 2, 1, 6, 9, 1, 4, 6, 5, 5]
        dtw = DTW(x, y)
        result = np.round(dtw.find_alignment_cost(method='td-method', look_for="MEAN", window_size=5, step=5), 2)
        self.assertEqual(result, 8.5)

    @patch("matplotlib.pyplot.savefig")
    def test_find_min_alignment_cost_without_filename(self, mock_savefig):
        x = [0, 3, 6, 2, 4, 1, 1, 1, 1, 1, 9, 0]
        y = [0, 1, 4, 2, 1, 6, 9, 1, 4, 6, 5, 5]
        dtw = DTW(x, y)
        with patch("matplotlib.pyplot.show"):
            result = dtw.find_alignment_cost(method='d-method', look_for="MIN", window_size=5, step=5)
        self.assertEqual(result, 0.7)

    @patch("matplotlib.pyplot.savefig")
    def test_find_min_alignment_cost_with_filename(self, mock_savefig):
        x = [0, 3, 6, 2, 4, 1, 1, 1, 1, 1, 9, 0]
        y = [0, 1, 4, 2, 1, 6, 9, 1, 4, 6, 5, 5]
        dtw = DTW(x, y)
        with patch("matplotlib.pyplot.show"):
            result = dtw.find_alignment_cost(method='c-method', look_for="MIN", window_size=5, step=5,
                                             filename="test_plot")
        self.assertEqual(result, 0)

    @patch("matplotlib.pyplot.savefig")
    def test_find_max_alignment_cost_without_filename(self, mock_savefig):
        x = [0, 3, 6, 2, 4, 1, 1, 1, 1, 1, 9, 0]
        y = [0, 1, 4, 2, 1, 6, 9, 1, 4, 6, 5, 5]
        dtw = DTW(x, y)
        with patch("matplotlib.pyplot.show"):
            result = dtw.find_alignment_cost(method='td-method', look_for="MAX", window_size=5, step=5)
        self.assertEqual(result, 13.6)

    @patch("matplotlib.pyplot.savefig")
    def test_find_max_alignment_cost_with_filename(self, mock_savefig):
        x = [0, 3, 6, 2, 4, 1, 1, 1, 1, 1, 9, 0]
        y = [0, 1, 4, 2, 1, 6, 9, 1, 4, 6, 5, 5]
        dtw = DTW(x, y)
        with patch("matplotlib.pyplot.show"):
            result = dtw.find_alignment_cost(method='d-method', look_for="MAX", window_size=5, step=5,
                                             filename="test_plot")
        self.assertEqual(result, 2.1)

    def test_find_alignment_cost_look_for_error(self):
        x = [0, 3, 6, 2, 4, 1, 1, 1, 1, 1, 9, 0]
        y = [0, 1, 4, 2, 1, 6, 9, 1, 4, 6, 5, 5]
        dtw = DTW(x, y)
        with self.assertRaises(ValueError):
            dtw.find_alignment_cost(method='td-method', look_for="N/A", window_size=5, step=5)

    def test_find_alignment_cost_window_error(self):
        x = [0, 3, 6, 2, 4, 1, 1, 1, 1, 1, 9, 0]
        y = [0, 1, 4, 2, 1, 6, 9, 1, 4, 6, 5, 5]
        dtw = DTW(x, y)
        with self.assertRaises(ValueError):
            dtw.find_alignment_cost(method='td-method', look_for="MIN", window_size=2, step=5)

    def test_find_alignment_cost_step_error(self):
        x = [0, 3, 6, 2, 4, 1, 1, 1, 1, 1, 9, 0]
        y = [0, 1, 4, 2, 1, 6, 9, 1, 4, 6, 5, 5]
        dtw = DTW(x, y)
        with self.assertRaises(ValueError):
            dtw.find_alignment_cost(method='td-method', look_for="MEAN", window_size=5, step=0)

    @patch("matplotlib.pyplot.savefig")
    def test_plot_signals_without_filename(self, mock_savefig):
        with patch("matplotlib.pyplot.show") as mock_show:
            self.dtw.plot_signals()
            mock_show.assert_called_once()
        mock_savefig.assert_not_called()

    @patch("matplotlib.pyplot.savefig")
    def test_plot_signals_with_filename(self, mock_savefig):
        with patch("matplotlib.pyplot.show") as mock_show:
            self.dtw.plot_signals(filename="test_plot")
            mock_savefig.assert_called_once_with("test_plot.pdf", format="pdf")
            mock_show.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    def test_plot_cost_matrix_without_filename(self, mock_savefig):
        self.dtw.traceback()
        with patch("matplotlib.pyplot.show") as mock_show:
            self.dtw.plot_cost_matrix()
            mock_show.assert_called_once()
        mock_savefig.assert_not_called()

    @patch("matplotlib.pyplot.savefig")
    def test_plot_cost_matrix_with_filename(self, mock_savefig):
        self.dtw.traceback()
        with patch("matplotlib.pyplot.show") as mock_show:
            self.dtw.plot_cost_matrix(filename="test_plot")
            mock_savefig.assert_called_once_with("test_plot.pdf", format="pdf")
            mock_show.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    def test_plot_alignment_without_filename(self, mock_savefig):
        self.dtw.traceback()
        with patch("matplotlib.pyplot.show") as mock_show:
            self.dtw.plot_alignment()
            mock_show.assert_called_once()
        mock_savefig.assert_not_called()

    @patch("matplotlib.pyplot.savefig")
    def test_plot_alignment_with_filename(self, mock_savefig):
        self.dtw.traceback()
        with patch("matplotlib.pyplot.show") as mock_show:
            self.dtw.plot_alignment(filename="test_plot")
            mock_savefig.assert_called_once_with("test_plot.pdf", format="pdf")
            mock_show.assert_called_once()


if __name__ == '__main__':
    unittest.main()
