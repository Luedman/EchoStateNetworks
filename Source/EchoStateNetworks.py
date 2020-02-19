# The code in this file is inspired by an ESN implementation in MATLAB by H. Jaeger
# Jaeger, Herbert, "The echo state approach to analysing and training recurrent neural networks-with an erratum note",
# Bonn, Germany: German National Research Center for Information Technology GMD Technical Report 148, 34 (2001), pp. 13.

# Python Packages
import numpy as np
import scipy as sc
import warnings
import datetime as dt
import pprint
from matplotlib import pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind

# Project Scripts
import dataUtils

warnings.filterwarnings("ignore")
np.random.seed(1)
counter = 0


class EchoStateNetwork:
    def __init__(
        self, no_input_nodes, no_output_nodes, no_internal_nodes, **hyper_parameter
    ):

        np.random.seed(1)
        self.no_input_nodes: int = no_input_nodes
        self.no_output_nodes: int = no_output_nodes
        self.no_internal_nodes: int = no_internal_nodes

        self.model_name: str = hyper_parameter.get("modelName") or "ESN"
        self.regression_lambda: float = hyper_parameter.get(
            "regression_lambda"
        ) or 1e-10
        self.spectral_radius: float = hyper_parameter.get("spectral_radius") or 0.5
        self.leaking_rate: float = hyper_parameter.get("leaking_rate") or 0.0
        self.connectivity: float = hyper_parameter.get("connectivity") or min(
            [10 / self.no_internal_nodes, 1]
        )
        self.input_mask: np.ndarray = hyper_parameter.get("input_mask") or np.ones(
            [self.no_internal_nodes, self.no_input_nodes]
        )
        self.input_scaling: float = hyper_parameter.get("input_scaling") or 1.0
        self.input_shift: np.ndarray = hyper_parameter.get("input_shift") or 0.0

        self.model_type: str = "ESN"
        self.network_trained: bool = False
        self.reservoir_matrix: np.ndarray = None
        self.reservoir_readout: np.ndarray = None
        self.model_residual_matrix: np.ndarray = None
        self.collected_state_matrix_training: np.ndarray = None

        self.__check_input_args(**hyper_parameter)

        # Create reservoir matrix
        success = 0
        while success == 0:
            try:
                rvs = sc.stats.norm(loc=0, scale=1).rvs
                self.reservoir_matrix = sc.sparse.random(
                    self.no_internal_nodes,
                    self.no_internal_nodes,
                    density=self.connectivity,
                    data_rvs=rvs,
                ).A
                eigs = sc.sparse.linalg.eigs(self.reservoir_matrix, 1, which="LM")

                max_val = max(abs(eigs[1]))
                self.reservoir_matrix = self.reservoir_matrix / (1.25 * max_val)
                success = 1
            except ValueError:
                pass

        self.reservoir_matrix *= self.spectral_radius

        assert self.reservoir_matrix.shape == (
            self.no_internal_nodes,
            self.no_internal_nodes,
        )

        self.collectedStateMatrix = np.zeros([self.no_internal_nodes, 1])

    def __check_input_args(self, **hyper_parameter):
        for arg in hyper_parameter:
            assert arg in self.__dict__.keys(), arg + " This argument in unknown"

    def __repr__(self):
        result = (
            "Name\t\t"
            + self.model_name
            + "\nNo. Inputs\t"
            + str(self.no_input_nodes)
            + "\nNo. Outputs\t"
            + str(self.no_output_nodes)
            + "\nReservoir Size\t"
            + str(self.no_internal_nodes)
            + "\nSpectral Radius\t"
            + str(self.spectral_radius)
            + "\nReg. Lambda\t"
            + str(self.regression_lambda)
            + "\nInput Shift \t"
            + str(self.input_shift)
            + "\nInput Scaling\t"
            + str(self.input_scaling)
            + "\nConnectivity\t"
            + str(self.connectivity)
            + "\nTrained\t\t"
            + str(self.network_trained)
        )

        return str(result)

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def type(self) -> str:
        return self.model_type

    @property
    def trained(self) -> bool:
        return self.network_trained

    @staticmethod
    def __activation_function(input_vector, function="Sigmoid"):
        def sigmoid_activation(x):
            return 1.0 / (1.0 + np.exp(-x))

        def tanh_activation(x):
            return np.tanh(x)

        if function.upper() == "SIGMOID":
            result = np.array(list(map(sigmoid_activation, np.array(input_vector))))
        elif function.upper() == "TANH":
            result = np.array(list(map(tanh_activation, np.array(input_vector))))
        else:
            raise NameError('Argument "function" for __activationFunction not found.')

        return result

    @staticmethod
    def __output_activation_function(input_vector):
        result = np.array(input_vector)
        return result

    def __reservoir_state(self, prev_output, prev_reservoir_state):
        # Calculate the reservoir state for at given t

        prev_reservoir_state = prev_reservoir_state.reshape(self.no_internal_nodes, 1)

        activation = (
            np.matmul(self.reservoir_matrix, prev_reservoir_state)
            + self.input_scaling
            * np.matmul(self.input_mask, prev_output).reshape(self.no_internal_nodes, 1)
            + self.input_shift
        )

        reservoir_state_result = self.__activation_function(activation, "Sigmoid")
        reservoir_state_result = (
            -self.leaking_rate * prev_reservoir_state + reservoir_state_result
        )

        assert reservoir_state_result.shape == (self.no_internal_nodes, 1)

        return reservoir_state_result

    def __collect_state_matrix(self, input_vector, no_forget_points):
        # Calculate a series of reservoir states and store in collected sate matrix

        for i in range(self.collectedStateMatrix.shape[1] - 1, input_vector.shape[0]):

            self.collectedStateMatrix = np.concatenate(
                (
                    self.collectedStateMatrix,
                    self.__reservoir_state(
                        input_vector[i], self.collectedStateMatrix[:, -1]
                    ),
                ),
                axis=1,
            )

        return self.collectedStateMatrix[:, no_forget_points + 1 :]

    def fit(self, data: dataUtils.Data, no_forget_points: int):
        # Fit the ESN to the training data

        # ToDo: implement ts mapping of different shapes
        assert (
                data.no_input_series == self.no_input_nodes
        ), "Data format does not match ESN properties"

        collected_state_matrix = self.__collect_state_matrix(
            data.x_train, no_forget_points
        )

        gamma = np.matmul(
            collected_state_matrix, collected_state_matrix.T
        ) + self.regression_lambda * np.eye(self.no_internal_nodes)

        cov = np.matmul(collected_state_matrix, data.y_train[no_forget_points:])

        try:
            self.reservoir_readout = np.matmul(np.linalg.inv(gamma), cov).T
            self.collected_state_matrix_training = collected_state_matrix
            self.network_trained = True
        except np.linalg.linalg.LinAlgError:
            self.reservoir_readout = np.ones(
                (data.y_train.shape[1], self.no_internal_nodes)
            )
            self.network_trained = False
            print("Failed to train Network")

        assert self.reservoir_readout.shape == (
            data.y_train.shape[1],
            self.no_internal_nodes,
        )

        output_sequence = self.__output_activation_function(
            np.matmul(self.reservoir_readout, collected_state_matrix)
        ).T
        output_sequence = (
            output_sequence - np.ones(output_sequence.shape) * self.input_shift
        ) / self.input_scaling

        self.model_residual_matrix = data.y_train[no_forget_points:] - output_sequence

        return

    def test(self, x_train, no_forget_points=100):
        # Check whether the ESN fits the training data well

        assert self.network_trained == True, "Network isn't trained yet"

        collected_state_matrix = self.__collect_state_matrix(x_train, no_forget_points)

        output_sequence = self.__output_activation_function(
            np.matmul(self.reservoir_readout, collected_state_matrix)
        )
        output_sequence = (
            output_sequence - np.ones(output_sequence.shape) * self.input_shift
        ) / self.input_scaling

        average_rmse = np.mean(
            np.power(x_train[no_forget_points:] - output_sequence.T, 2)
        )

        return average_rmse

    def evaluate(self, data, show_plot=False):
        # Evaluate the output of ESN.test()

        assert (
            data.x_test.shape[1] == data.y_test.shape[1]
        ), "X and Y should be of same lenght (shape[1])"

        output = self.test(data.x_test, 10)
        y_hat = x

        try:
            error_rmse = np.sqrt(
                mean_squared_error(data.y_test[-y_hat.shape[0] :], y_hat)
            )
        except:
            error_rmse = float("inf")
            print("Error when calculating RSME")

        if show_plot:
            plt.plot(data.y_test[-y_hat.shape[0]:], label="Var")
            plt.plot(y_hat, label="ESN")
            plt.legend()
            plt.show()

        return error_rmse, y_hat

    def multi_step_ahead_forecast(
        self, data, forecast_horizon, index, window_mode="Expanding", window_size=0, no_samples=10
    ):
        # Refit the ESN and create a multi step ahead forecast

        if window_mode.upper() == "EXPANDING":
            data.split_data_by_index(index, start_point_index=0)
            self.fit(data, no_forget_points=50)
        elif window_mode.upper() == "ROLLING":
            data.split_data_by_index(index, index - window_size)
            self.fit(data, no_forget_points=50)
        elif window_mode.upper() == "FIXED":
            data.split_data_by_index(index, start_point_index=0)
            assert self.network_trained is True

        actual = data.y_test[:forecast_horizon]

        random_start_indices = np.random.randint(
            0,
            self.model_residual_matrix.shape[0] + 1 - forecast_horizon,
            size=no_samples,
        )
        random_residuals_matrix = np.array(
            [
                self.model_residual_matrix[
                    randomIndex : randomIndex + forecast_horizon, :
                ]
                for randomIndex in random_start_indices
            ]
        )

        prev_reservoir_state = self.collected_state_matrix_training[:, -1].reshape(
            -1, 1
        )

        multi_step_forecast = np.zeros((forecast_horizon, self.no_output_nodes))
        multi_step_forecast[-1] = data.y_test[0]

        for i in range(0, random_residuals_matrix.shape[1]):

            one_step_forecasting_samples = []

            for residualVector in random_residuals_matrix:

                reservoir_state = self.__reservoir_state(
                    multi_step_forecast[i - 1] + residualVector[i], prev_reservoir_state
                )

                one_step_forecast = self.__output_activation_function(
                    np.matmul(self.reservoir_readout, prev_reservoir_state)
                )
                one_step_forecast = (
                    one_step_forecast - self.input_shift
                ) / self.input_scaling

                if all(np.absolute(one_step_forecast)) < 1.01:
                    one_step_forecasting_samples.append(one_step_forecast)

            if one_step_forecasting_samples:
                multi_step_forecast[i] = np.average(one_step_forecasting_samples)
                prev_reservoir_state = reservoir_state
            else:
                multi_step_forecast[i] = multi_step_forecast[i - 1]
                prev_reservoir_state = reservoir_state

        def show_forecast():
            # Debug Function
            ax1 = plt.subplot(2, 1, 1)
            ax1.set_title("Actual vs Forecast")
            ax1.plot(multi_step_forecast, label="ESN")
            ax1.plot(data.x_test[index : index + forecast_horizon], label="actual")
            ax1.legend()
            ax2 = plt.subplot(2, 1, 2)
            ax2.set_title("Reservoir Readout")
            ax2.bar(
                list(range(0, self.no_internal_nodes)),
                self.reservoir_readout.reshape(-1),
            )
            plt.tight_layout()
            plt.show()

        return multi_step_forecast, actual


def bayesianOptimization(dataPath="./Data/"):
    # Hyperparameter Bayesian Search
    print("Bayesian Optimization")

    # Disable Warnings (especially overflow)
    warnings.filterwarnings("ignore")

    # USER INPUT
    assetList = ["AAPL"]

    daysAhead = 30
    trainingFraction = 0.8

    data = dataUtils.loadScaleDataMultivariate(
        assetList, dataPath, endDate=dt.datetime(2008, 12, 31)
    )
    splitIndex = int(list(data.index).index(dt.datetime(2006, 1, 3)))
    data.split_data_by_index(splitIndex)

    def esn_evaluation(
        internal_nodes, spectral_radius, regression_lambda, connectivity, leaking_rate
    ):

        hyperparameter_esn = {
            "internalNodes": internal_nodes,
            "inputScaling": 1,
            "inputShift": 0,
            "spectralRadius": spectral_radius,
            "regressionLambda": regression_lambda,
            "connectivity": connectivity,
            "leakingRate": leaking_rate,
            "seed": 1,
        }

        test_esn = EchoStateNetwork(
            nInputNodes=data.noTimeSeries,
            nOutputNodes=data.noTimeSeries,
            hyper_parameter=hyperparameter_esn,
        )

        test_esn.fit(data, 100)
        error = test_esn.test(data.x_train)

        del test_esn
        return error * -1

    pbounds = {
        "internalNodes": (30, 100),
        "spectralRadius": (0.1, 0.5),
        "regressionLambda": (0.001, 1e-5),
        "connectivity": (0.01, 0.1),
        "leakingRate": (0, 0.3),
    }
    optimizer = BayesianOptimization(f=esn_evaluation, pbounds=pbounds, random_state=1)

    optimizer.maximize(init_points=2000, n_iter=2000)
    print(optimizer.max)


if __name__ == "__main__":
    pass
