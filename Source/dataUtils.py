import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List


class Data:
    def __init__(
        self,
        model_input: Union[np.ndarray, pd.DataFrame],
        index: list = None,
        name_data_set: str = "Data",
        model_output: Union[np.ndarray, pd.DataFrame] = None,
        x_labels: List[str] = None,
        y_labels: List[str] = None,
    ):
        """Inialize the data Object
        :param model_input: vector or matrix of the model input or
                x values of shape (samples, features)
        :param index: Index for splitting the data into a test and training set
        :param name_data_set: Name of the data set
        :param model_output: vector or matrix of the model output or
                y values of shape (samples, features). Optional,
        :param x_labels: labels for the input features
        :param y_labels: labels for the output features
        """

        convert = np.vectorize(lambda x: float(str(x).replace(",", ".")))
        self._time_series_input = convert(model_input)
        self._time_series_output = convert(model_output)
        if model_output is None:
            self._time_series_input = convert(self._time_series_input[:-1])
            self._time_series_output = convert(self._time_series_input[1:])

        assert (
            self._time_series_input.shape[0] == self._time_series_output.shape[0]
        ), "Input and output time series must have same length in dimension 0 "

        self._no_data_points = self._time_series_input.shape[0]
        self._name_data_set = name_data_set
        self._index = index or list(np.arange(0, self._no_data_points))

        self._start_point_index = 0
        self._split_index = self._time_series_input.shape[0]

        self._data_scaled: bool = False
        self._scaler = None

        self._x_labels = x_labels or []
        self._y_labels = y_labels or []
        if type(model_input) == pd.DataFrame:
            self._x_labels = model_input

        self._x_train: np.ndarray = np.array([])
        self._x_test: np.ndarray = np.array([])
        self._y_train: np.ndarray = np.array([])
        self._y_test: np.ndarray = np.array([])
        self.__set_data()

    def __set_data(self) -> None:
        """Apply data split"""
        self._x_train = self._time_series_input[
            self._start_point_index : self._split_index - 1
        ]
        self._x_test = self._time_series_input[self._split_index : -1]

        output_data = (
            self._time_series_output
            if self._time_series_output is not None
            else self._time_series_input
        )
        self._y_train = output_data[self._start_point_index + 1 : self._split_index]
        self._y_test = output_data[self._split_index + 1 :]

        return

    @property
    def no_input_series(self) -> int:
        return self._time_series_input.shape[1]

    @property
    def no_output_series(self) -> int:
        return (
            self._time_series_output.shape[1]
            if self._time_series_output is not None
            else self._time_series_input.shape[1]
        )

    @property
    def no_data_points(self) -> int:
        return self._no_data_points

    @property
    def name(self) -> str:
        return self._name_data_set

    @property
    def index(self) -> list:
        return self._index

    @property
    def x_train(self) -> np.ndarray:
        return self._x_train

    @property
    def y_train(self) -> np.ndarray:
        return self._y_train

    @property
    def x_test(self) -> np.ndarray:
        return self._x_test

    @property
    def y_test(self) -> np.ndarray:
        return self._y_test

    @property
    def scaler(self):
        return self._scaler

    def visualize(self, data_points: int = 500, show_out_put: bool = False):
        """Visualize the data input/output
        :param data_points: amount of data points to plot
        :param show_out_put: plot the output instead of the input time series
        """
        def plot_graph(ts_data):
            if not show_out_put:
                fig, axs = plt.subplots(2, 2)
                k = 0
                for i in [0, 1]:
                    for j in [0, 1]:
                        axs[i, j].plot(ts_data[-data_points:, k])
                        k += 1
                plt.tight_layout()
                plt.show()

        if not show_out_put:
            plot_graph(self._time_series_input)
        else:
            plot_graph(self._time_series_output)

        return

    def split_data_by_index(
        self, split_index: int, start_point_index: int = 0
    ) -> (str, str):
        """Splits the data by index into a training and test set
        :param split_index: index where the split is located
        :param start_point_index: determine the start index of the training set
        :return: testing start index, testing end index
        """
        self._split_index = split_index
        self._start_point_index = start_point_index

        testing_start_label = self._index[self._split_index]
        testing_end_label = self._index[-1]

        self.__set_data()
        return testing_start_label, testing_end_label

    def split_data_by_label(
        self, split_label: str, start_label: str = None
    ) -> (str, str):
        """Splits the data by label into a training and test set
        :param split_label: label where the split is located
        :param start_label: determine the start label of the training set
        :return: testing start label, testing end label
        """
        assert split_label in list(self._index), "Label not found"
        self._split_index = list(self._index).index(split_label)
        try:
            self._start_point_index = list(self._index).index(start_label)
        except ValueError:
            self._start_point_index = 0

        testing_start_label = self._index[self._split_index]
        testing_end_label = self._index[-1]
        self.__set_data()
        return testing_start_label, testing_end_label

    def scale_data(self, sk_learn_scaler):
        """Scales the data
        :param sk_learn_scaler: an instance of an unfitted sklearn scaler
        """
        self._x_train = sk_learn_scaler.fit_transform(self._x_train)
        self._y_train = sk_learn_scaler.transform(self._y_train)
        self._x_test = sk_learn_scaler.transform(self._x_test)
        self._y_test = sk_learn_scaler.transform(self._y_test)

        self._scaler = sk_learn_scaler
        self._data_scaled = True

    def rescale_data(self):
        """Rescales the data
        """
        assert self._data_scaled == True, "Data is not scaled"

        self._x_train = self.scaler.inverse_transform(self._x_train)
        self._y_train = self.scaler.inverse_transform(self._y_train)
        self._x_test = self.scaler.inverse_transform(self._x_test)
        self._y_test = self.scaler.inverse_transform(self._y_test)

        self._data_scaled = False
