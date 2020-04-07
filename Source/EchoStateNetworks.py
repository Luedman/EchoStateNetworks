# The code in this file is inspired by an ESN implementation in MATLAB by H. Jaeger
# Jaeger, Herbert, "The echo state approach to analysing and training recurrent neural networks-with an erratum note",
# Bonn, Germany: German National Research Center for Information Technology GMD Technical Report 148, 34 (2001), pp. 13.

# Modules
import dataUtils

# Python Packages
import numpy as np
from scipy import sparse, stats
from bayes_opt import BayesianOptimization
from itertools import product as cart_product
from functools import partial
from typing import Callable
from pprint import pprint
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class EchoStateNetwork:
    def __init__(
        self, no_input_nodes, no_output_nodes, no_internal_nodes, **hyper_parameter
    ):
        """
        Initialize the ESN
        :param no_input_nodes: Number of input nodes
        :param no_output_nodes: Number of input output
        :param no_internal_nodes: Size of the reservoir
        :param hyper_parameter: Optional hyper parameter
                Default values are
                model_name: "ESN_model"
                regression_lambda: 1e-10
                spectral_radius: 0.5
                leaking_rate: 0.0 (has no effect)
                connectivity: 10/no_internal_nodes
                input_mask: array of 1 (has no effect)
                input_scaling: 1.0 (has no effect)
                input_shift: 0.0 (has no effect)
                seed: 1
        """

        self._no_input_nodes: int = no_input_nodes
        self._no_output_nodes: int = no_output_nodes
        self._no_internal_nodes: int = no_internal_nodes

        self._model_name: str = hyper_parameter.get("model_name") or "ESN_model"

        self._spectral_radius: float = hyper_parameter.get("spectral_radius") or 0.5
        self._leaking_rate: float = hyper_parameter.get("leaking_rate") or 0.0
        self._input_scaling: np.ndarray = hyper_parameter.get("input_scaling") or 1.0
        self._input_shift: np.ndarray = hyper_parameter.get("input_shift") or 0.0
        self._regression_lambda: float = hyper_parameter.get(
            "regression_lambda"
        ) or 1e-10
        self._connectivity: float = hyper_parameter.get("connectivity") or min(
            [10 / self._no_internal_nodes, 1]
        )
        self._input_mask: np.ndarray = hyper_parameter.get("input_mask") or np.ones(
            [self._no_internal_nodes, self._no_input_nodes]
        )

        self._model_type: str = "ESN"
        self._network_trained: bool = False
        self._no_forget_points: int = 0
        self._reservoir_matrix: np.ndarray = np.ndarray([])
        self._reservoir_readout: np.ndarray = np.ndarray([])
        self._model_residual_matrix: np.ndarray = np.ndarray([])
        self._collected_state_matrix_training: np.ndarray = np.ndarray([])
        self._seed = hyper_parameter.get("seed") or 1
        np.random.seed(self._seed)

        self.__check_input_args(**hyper_parameter)

        # Create reservoir matrix
        success = False
        while success is False:
            try:
                rvs = stats.norm(loc=0, scale=1).rvs
                self._reservoir_matrix = sparse.random(
                    self._no_internal_nodes,
                    self._no_internal_nodes,
                    density=self._connectivity,
                    data_rvs=rvs,
                ).A
                eigenvalues = sparse.linalg.eigs(self._reservoir_matrix, 1, which="LM")

                max_val = max(abs(eigenvalues[1]))
                self._reservoir_matrix = self._reservoir_matrix / (1.25 * max_val)
                success = True
            except (ValueError, sparse.linalg.ArpackNoConvergence):
                pass

        self._reservoir_matrix *= self._spectral_radius

        assert self._reservoir_matrix.shape == (
            self._no_internal_nodes,
            self._no_internal_nodes,
        )

        self._collectedStateMatrix = np.zeros([self._no_internal_nodes, 1])

    def __check_input_args(self, **hyper_parameter) -> None:
        """Check whether the the input parameters
            correspond to an class attribute
        :param hyper_parameter: input parameters when initilizing the class
        :return: None
        """

        for arg in hyper_parameter:
            assert "_" + arg in self.__dict__.keys(), arg + " This argument is unknown"

    def __repr__(self) -> str:
        result = (
            "Name\t\t"
            + self._model_name
            + "\nNo. Inputs\t"
            + str(self._no_input_nodes)
            + "\nNo. Outputs\t"
            + str(self._no_output_nodes)
            + "\nReservoir Size\t"
            + str(self._no_internal_nodes)
            + "\nSpectral Radius\t"
            + str(self._spectral_radius)
            + "\nReg. Lambda\t"
            + str(self._regression_lambda)
            + "\nInput Shift \t"
            + str(self._input_shift)
            + "\nInput Scaling\t"
            + str(self._input_scaling)
            + "\nConnectivity\t"
            + str(self._connectivity)
            + "\nTrained\t\t"
            + str(self._network_trained)
        )

        return str(result)

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def type(self) -> str:
        return self._model_type

    @property
    def trained(self) -> bool:
        return self._network_trained

    @staticmethod
    def __activation_function(
        input_vector: np.ndarray, activation_type: str
    ) -> np.ndarray:
        """Applies the activation function
        :param input_vector: input vector
        :param activation_type: [Linear, Tanh, Sigmoid] Type of activation function
        :return: input vector with activation function applied to it
        """

        assert activation_type.upper() in [
            "SIGMOID",
            "TANH",
            "LINEAR",
        ], "Activation Type unknown"
        if activation_type.upper() == "SIGMOID":
            sigmoid_activation = 1.0 / (1.0 + np.exp(-input_vector))
            return sigmoid_activation
        elif activation_type.upper() == "TANH":
            tanh_activation = np.tanh(input_vector)
            return tanh_activation
        elif activation_type.upper() == "LINEAR":
            linear_activation = np.array(input_vector)
            return linear_activation

    @staticmethod
    def __rmse(x: np.ndarray, x_hat: np.ndarray) -> float:
        """Calculate the root mean squared error"""
        rmse = float(np.sqrt(np.mean(np.power(np.subtract(x, x_hat), 2))))
        return rmse

    def __activate_state(
        self, input_vector: np.ndarray, activation_type: str = "Sigmoid"
    ) -> np.ndarray:
        """Applies an activation function to the internal state
        :param input_vector: ESN internal state
        :param activation_type: [Linear, Tanh, Sigmoid] Type of activation function
        :return: the internal state with an activation function applied to it
        """

        result = np.array(
            list(
                map(
                    partial(
                        self.__activation_function, activation_type=activation_type
                    ),
                    np.array(input_vector),
                )
            )
        )
        return result

    def __activate_output(
        self, output_vector: np.ndarray, activation_type: str = "Linear"
    ) -> np.ndarray:
        """Applies an activation function to the ESN output
        :param output_vector: ESN output
        :param activation_type: [Linear, Tanh, Sigmoid] Type of activation function
        :return: ESN output with an activation function applied to it
        """

        output_activated = self.__activation_function(output_vector, activation_type)
        return output_activated

    def __reservoir_state(self, prev_output, prev_reservoir_state) -> np.ndarray:
        """Calculate the next reservoir state for t+1
        :param prev_output: The ESN output at t-1
        :param prev_reservoir_state: The ESN state at t-1
        :return: The current reservoir state
        """

        prev_reservoir_state = prev_reservoir_state.reshape(self._no_internal_nodes, 1)

        activation = (
            np.matmul(self._reservoir_matrix, prev_reservoir_state)
            + self._input_scaling
            * np.matmul(self._input_mask, prev_output).reshape(
                self._no_internal_nodes, 1
            )
            + self._input_shift
        )

        reservoir_state_result = self.__activate_state(activation, "Sigmoid")
        reservoir_state_result = (
            -self._leaking_rate * prev_reservoir_state + reservoir_state_result
        )

        assert reservoir_state_result.shape == (self._no_internal_nodes, 1)

        return reservoir_state_result

    def __collect_state_matrix(self, x_input: np.ndarray) -> np.ndarray:
        """Calculate a series of reservoir states and stores them in
            the collected sate matrix
        :param x_input: a time series vector of the inputs fro the ESN
        :return: the collected states for each input value
        """

        for i in range(self._collectedStateMatrix.shape[1] - 1, x_input.shape[0]):
            self._collectedStateMatrix = np.concatenate(
                (
                    self._collectedStateMatrix,
                    self.__reservoir_state(
                        x_input[i], self._collectedStateMatrix[:, -1]
                    ),
                ),
                axis=1,
            )

        return self._collectedStateMatrix[:, self._no_forget_points + 1 :]

    def fit(self, data: dataUtils.Data, no_forget_points: int) -> None:
        """Fit the ESN to the training data
        :param data: Data object
        :param no_forget_points: number of data points to ignore at the
                beginning of the training when collecting the reservoir states.
                This intends to use only the reservoir states with sufficient backlog.
        :return: None
        """

        self._no_forget_points = int(no_forget_points)
        self.__check_data_properties(data)

        collected_state_matrix = self.__collect_state_matrix(data.x_train)

        gamma = np.matmul(
            collected_state_matrix, collected_state_matrix.T
        ) + self._regression_lambda * np.eye(self._no_internal_nodes)

        cov = np.matmul(collected_state_matrix, data.y_train[self._no_forget_points :])

        try:
            self._reservoir_readout = np.matmul(np.linalg.inv(gamma), cov).T
            self._collected_state_matrix_training = collected_state_matrix
            self._network_trained = True
        except np.linalg.linalg.LinAlgError:
            self._reservoir_readout = np.zeros(
                (data.y_train.shape[1], self._no_internal_nodes)
            )
            print("Failed to train Network - Collected state matrix is not invertibel")

        assert self._reservoir_readout.shape == (
            data.y_train.shape[1],
            self._no_internal_nodes,
        )

        output_sequence = self.__activate_output(
            np.matmul(self._reservoir_readout, collected_state_matrix)
        ).T
        output_sequence = (
            output_sequence - np.ones(output_sequence.shape) * self._input_shift
        ) / self._input_scaling

        self._model_residual_matrix = (
            data.y_train[self._no_forget_points :] - output_sequence
        )

        return

    def __check_data_properties(self, data: dataUtils.Data) -> None:
        """Check data properties. Raises assertion error if there is a
            mismatch between ESN and data properties
        :param data: Data object
        :return: None
        """

        assert (
            data.no_input_series == self._no_input_nodes
            and data.no_output_series == self._no_output_nodes
        ), (
            "Data format does not match ESN nodes. Input (%d/%d) - Output (%d/%d)"
            % (
                data.no_input_series,
                self._no_input_nodes,
                data.no_output_series,
                self._no_output_nodes,
            )
        )

        assert self._no_forget_points < data.y_train.shape[0], (
            "No Forget Points (%d) must be smaller than training set size (%d)"
            % (self._no_forget_points, data.y_train.shape[0])
        )

        return

    def one_step_ahead_forecasts(self, x_input: np.ndarray) -> np.ndarray:
        """Generates a series of outputs, given an input series
        :param x_input: input series
        :return: ESN output for the given input
        """

        assert self._network_trained is True, "Network isn't trained yet"

        collected_state_matrix = self.__collect_state_matrix(x_input)

        output_sequence = self.__activate_output(
            np.matmul(self._reservoir_readout, collected_state_matrix)
        )
        output_sequence -= np.ones(output_sequence.shape) * self._input_shift
        output_sequence /= self._input_scaling

        return output_sequence.T

    def evaluate_fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        error_function: Callable[[np.ndarray, np.ndarray], float] = None,
    ) -> (float, np.ndarray):
        """Evaluate whether the ESN fits the training data well
        :param x_train: input values of the training set
        :param y_train: output values of the training set
        :param error_function a alternative error function
        :return: error measure, the ESN output sequence
        """

        output_sequence = self.one_step_ahead_forecasts(x_train)

        if error_function is not None:
            assert (
                type(error_function) is Callable[[np.ndarray, np.ndarray], float]
            ), "Error function must be (np.ndarray, np.ndarray) -> float"
            error = error_function(y_train[self._no_forget_points :], output_sequence)
            return error, output_sequence
        else:
            rmse = self.__rmse(y_train[self._no_forget_points :], output_sequence)
            return rmse, output_sequence

    def visualize_fit(
        self, x_train: np.ndarray, y_train: np.ndarray, scaler=None
    ) -> (np.ndarray, np.ndarray):
        """Visualize the training fit (generates a chart per output time series)
        :param x_train: The training set
        :param y_train: The test set
        :param scaler: an fitted sklearn scaler instance for rescaling the data
        :return: one step ahead forecasts on the training set,
                the true output values of the training set
        """

        assert (
            x_train.shape[0] == y_train.shape[0]
        ), "X and Y should be of same length (shape[0])"

        average_test_rmse, y_hat = self.evaluate_fit(x_train, y_train)

        if scaler is not None:
            y_train = scaler.inverse_transform(y_train)
            y_hat = scaler.inverse_transform(y_hat)

        y_plot, y_hat_plot = (0.0, 0.0)
        for i in range(0, y_hat.shape[1]):
            y_plot = y_train[-y_hat[:, i].shape[0] :, i]
            y_hat_plot = y_hat[:, i]

            plt.plot(y_plot, label="y", color="red", linestyle="-", markersize=8)
            plt.plot(
                y_hat_plot, label="y_hat", color="blue", linestyle=":", markersize=8
            )

            plt.ylim(0)
            plt.title("Output %i RMSE %.4f" % (i, self.__rmse(y_plot, y_hat_plot)))
            plt.legend()
            plt.show()

        return y_plot, y_hat_plot

    def forecast(
        self, data, forecast_horizon, no_samples=10
    ) -> (np.ndarray, np.ndarray):
        """Create a multi step ahead forecast from the beginning of the test set
        :param data: Data object
        :param forecast_horizon: int, time steps that the iterative forecast will reach into the future
        :param no_samples: int, Number of samples to be randomly drawn from the residual matrix
        :return: (y_hat, y)
        """

        assert (
            self._network_trained is True
        ), "ESN needs to be trained before you can forecast"

        actual = data.y_test[:forecast_horizon]

        random_start_indices = np.random.randint(
            0,
            self._model_residual_matrix.shape[0] + 1 - forecast_horizon,
            size=no_samples,
        )
        random_residuals_matrix = np.array(
            [
                self._model_residual_matrix[
                    randomIndex : randomIndex + forecast_horizon, :
                ]
                for randomIndex in random_start_indices
            ]
        )

        prev_reservoir_state = self._collected_state_matrix_training[:, -1].reshape(
            -1, 1
        )

        multi_step_forecast = np.zeros((forecast_horizon, self._no_output_nodes))
        multi_step_forecast[-1] = data.y_test[0]

        for i in range(0, random_residuals_matrix.shape[1]):

            one_step_forecasting_samples = list()

            for residualVector in random_residuals_matrix:

                reservoir_state = self.__reservoir_state(
                    multi_step_forecast[i - 1] + residualVector[i], prev_reservoir_state
                )

                one_step_forecast = self.__activate_output(
                    np.matmul(self._reservoir_readout, prev_reservoir_state)
                )
                one_step_forecast = (
                    one_step_forecast - self._input_shift
                ) / self._input_scaling

                prev_reservoir_state = reservoir_state

                if all(np.absolute(one_step_forecast)) < 1.01:
                    one_step_forecasting_samples.append(one_step_forecast)

            if one_step_forecasting_samples:
                multi_step_forecast[i] = np.average(
                    one_step_forecasting_samples, axis=0
                ).reshape(-1)
            else:
                multi_step_forecast[i] = multi_step_forecast[i - 1]

        if data.scaler is not None:
            multi_step_forecast = data.scaler.inverse_transform(multi_step_forecast)
            actual = data.scaler.inverse_transform(actual)

        return multi_step_forecast, actual

    def moving_window_forecasts(
        self,
        data: dataUtils.Data,
        index_base: int,
        index_end: int = None,
        forecast_horizon=30,
        window_mode="Expanding",
        window_size=20,
    ) -> dict:
        """Refit the ESN with past data and forecast from the most recent point of the known data.
        :param data: Data Object
        :param index_base: int, the index that marks the the most recent point of the 'known' data
        :param index_end: int, the function produces forecast for every index until this index is reached
        :param forecast_horizon: int, time steps that the iterative forecast will reach into the future
        :param window_mode: [Expanding, Rolling, Fixed] Determines the data on which the
                ESN is retrained at every point in time. Expanding uses all past data,
                Rolling uses a fixed window size meaning part of the data isn't used as
                'index' moves in the future
        :param window_size: int, only relevant if window_mode='Rolling'. Determines the amount
                of data that is being used
        :return:  multi_step_forecast, actual
        """

        if index_end is None:
            index_end = index_base + 1
        else:
            assert index_base > index_end, "index_end has to be larger that index"

        forecast_collection = dict()
        for i in range(index_base, index_end):
            assert self._network_trained is True
            if window_mode.upper() == "EXPANDING":
                data.split_data_by_index(i, start_point_index=0)
                self.fit(data, no_forget_points=self._no_forget_points)
            elif window_mode.upper() == "ROLLING":
                data.split_data_by_index(i, i - window_size)
                self.fit(data, no_forget_points=self._no_forget_points)
            elif window_mode.upper() == "FIXED":
                data.split_data_by_index(i, start_point_index=0)

            multi_step_forecast, actual = self.forecast(data, forecast_horizon)
            forecast_collection.update({i: [multi_step_forecast, actual]})

        return forecast_collection


def __evaluate_esn_model(data: dataUtils.Data, hyperparameter_esn) -> float:
    """see class EchoStateNetwork
    :return: the error value to be optimized
    """

    internal_nodes = hyperparameter_esn.pop("internal_nodes")

    test_esn = EchoStateNetwork(
        no_input_nodes=data.no_input_series,
        no_output_nodes=data.no_output_series,
        no_internal_nodes=internal_nodes,
        **hyperparameter_esn,
    )

    hyperparameter_esn.update({"internal_nodes": internal_nodes})

    test_esn.fit(data, 100)
    error, _ = test_esn.evaluate_fit(data.x_train, data.y_train)
    del test_esn

    return error * -1


def hyper_parameter_grid_search(
    data: dataUtils.Data,
    split_index: int,
    hyper_parameter_space: dict,
    show_status: bool = False,
):
    """Perform hyperparamter grid search optimization
    :param data: Data object
    :param split_index: split the data into training and test set
    :param hyper_parameter_space: a dictionary containing the limitis of the hyper parameter search
            of the form {"hyperparamter": min, max, no. steps}
    :param show_status: determine whether you want the result of each run printed to the console or not.
    :return: The set of optimal hyperparameters
    """

    print("Run Grid Search")
    data.split_data_by_index(split_index)

    search_points = dict()
    for param in hyper_parameter_space.keys():
        p_start = hyper_parameter_space[param][0]
        p_stop = hyper_parameter_space[param][1]
        p_num = hyper_parameter_space[param][2]

        search_points[param] = np.linspace(start=p_start, stop=p_stop, num=p_num)
        if param in ["internal_nodes", "seed"]:
            search_points[param] = list(map(lambda x: int(x), search_points[param]))

    parameter_combinations = cart_product(
        *[search_points[param] for param in sorted(search_points.keys())]
    )

    min_error: float = float("inf")
    optimal_hyper_parameter: dict = dict()
    run_id: int = 0

    for combination in parameter_combinations:
        run_id += 1

        hyper_parameter_sample = dict(zip(sorted(search_points.keys()), combination))
        test_error = __evaluate_esn_model(data, hyper_parameter_sample)

        if np.abs(test_error) < min_error:
            min_error = np.abs(test_error)
            optimal_hyper_parameter = hyper_parameter_sample

        if show_status:
            status = " %.d " % run_id
            for value in combination:
                status += " %.2f" % value

            print(
                status
                + "  Error: %.4f Global Optimum: %.4f" % (np.abs(test_error), min_error)
            )

    print("Grid Search Completed. Min Error %f" % min_error)
    pprint(optimal_hyper_parameter)

    return min_error, optimal_hyper_parameter


def hyper_parameter_bayesian_optimization(
    data: dataUtils.Data, split_index: int, hyper_parameter_space: dict
):
    """Perform bayesian hyper parameter optimization on the training set
    :param data: Data object
    :param split_index: split the data into training and test set
    :param hyper_parameter_space: a dictionary containing the limitis of the hyper parameter search
            of the form {"hyperparamter": min, max}
    :return: The set of optimal hyperparameters
    """

    print("Run Bayesian Optimization")
    pprint(hyper_parameter_space)

    warnings.filterwarnings("ignore")
    data.split_data_by_index(split_index)

    optimizer = BayesianOptimization(
        f=__evaluate_esn_model, pbounds=hyper_parameter_space, random_state=1
    )

    optimizer.maximize(init_points=1000, n_iter=1000)
    print(optimizer.max)

    return optimizer.max


if __name__ == "__main__":
    pass
