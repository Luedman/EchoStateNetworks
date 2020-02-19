import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import expm as matrix_exponential
from scipy.linalg import logm as matrix_logarithm
from matplotlib import pyplot as plt
from itertools import combinations_with_replacement
import datetime as dt
from os import getpid
from typing import Union

class Data:
    def __init__(self, time_series_data_input: np.ndarray, index: list=None, name_data_set ="Data",
                 time_series_data_output: np. ndarray=None):

        self._time_series_data_input = time_series_data_input
        self._time_series_data_output = time_series_data_output
        if time_series_data_output is not None:
            assert self._time_series_data_input.sahpe[0] == self._time_series_data_output.shape[0], \
                "Input and output time series must have same lenght in dimension 0 "
        self._no_data_points = self._time_series_data_input.shape[0]
        self._name_data_set = name_data_set
        if index is not None:
            self._index = index
        else:
            self._index = list(np.arange(0, self._no_data_points))

        self._max_look_back = 0
        self._split_index = self._time_series_data_input.shape[0] - self._max_look_back
        self._start_point_index = 0

    @property
    def no_input_series(self) -> int:
        return self._time_series_data_input.shape[1]

    @property
    def no_output_series(self) -> int:
        if self._time_series_data_output is not None:
            return self._time_series_data_output.shape[1]
        else:
            return self._time_series_data_input.shape[1]

    @property
    def no_data_points(self) -> int:
        return self._no_data_points

    @property
    def name(self) -> str:
        return self._name_data_set

    @property
    def x_train(self) -> np.ndarray:
        x_train = self._time_series_data_input[self._start_point_index + self._max_look_back: self._split_index - 1]
        return x_train

    @property
    def y_train(self) -> np.ndarray:
        if self._time_series_data_output is not None:
            y_train = self._time_series_data_output[self._start_point_index + self._max_look_back + 1: self._split_index]
        else:
            y_train = self._time_series_data_input[self._start_point_index + self._max_look_back + 1: self._split_index]
        return y_train

    @property
    def x_test(self) -> np.ndarray:
        x_test = self._time_series_data_input[self._split_index - 1 + self._max_look_back: -1]
        return x_test

    @property
    def y_test(self) -> np.ndarray:
        if self._time_series_data_output is not None:
            y_test = self._time_series_data_output[self._split_index + 1 + self._max_look_back:]
        else:
            y_test = self._time_series_data_input[self._split_index + 1 + self._max_look_back:]
        return y_test

    def set_max_look_back(self, look_back: int =0):
        self._max_look_back = look_back

    def split_data_by_index(self, split_index: int, start_point_index: int=0) -> (str, str):
        self._split_index = split_index
        self._start_point_index = start_point_index

        testing_start_label = self._index[self._split_index]
        testing_end_label = self._index[-1]
        return testing_start_label, testing_end_label

    def split_data_by_label(self, split_label: str, start_label:str=None) -> (str, str):
        assert split_label in list(self._index), "Label not found"
        self._split_index = list(self._index).index(split_label)
        try:
            self._start_point_index = list(self._index).index(start_label)
        except ValueError:
            self._start_point_index = 0

        testing_start_label = self._index[self._split_index]
        testing_end_label = self._index[-1]
        return testing_start_label, testing_end_label


class ErrorMetrics:
    def __init__(
        self,
        errorVectors,
        errorMatrices,
        modelName="NA",
        modelType="NA",
        testSetSize=0,
        oneDayAheadError=[],
    ):

        self.errorVector = errorVectors
        self.errorMatrix = errorMatrices
        self.modelName = modelName
        self.modelType = modelType
        self.testSetSize = testSetSize
        self.oneDayAheadError = oneDayAheadError


def createVarianceVector(data, assetList, dateIndex):
    # Create a covariance vector

    assetList = list(set(assetList))
    assetList.sort()

    date = data.index[dateIndex]

    assetCombos = list(combinations_with_replacement(assetList, 2))
    assetCombos = [combo[0] + "-" + combo[1] for combo in assetCombos]

    varianceVector = data.loc[date][assetCombos].values

    assert not any(np.isinf(varianceVector)), "Inf"

    return np.array(varianceVector)


def covMatFromVector(varianceVector, noAssets):
    # Convert a covariance matrix from a flattend matrix

    covarianceMatrix = np.zeros((noAssets, noAssets))
    covarianceMatrix.T[np.tril_indices(noAssets, 0)] = varianceVector

    ilower = np.tril_indices(noAssets, -1)
    covarianceMatrix[ilower] = covarianceMatrix.T[ilower]

    return covarianceMatrix


def varVectorFromCovMat(covarianceMatrix):
    # Flatten a covariance matrix to a vector

    assert covarianceMatrix.shape[0] == covarianceMatrix.shape[1]

    ilower = np.tril_indices(covarianceMatrix.shape[0], 0)
    varianceVector = covarianceMatrix[ilower]

    return varianceVector


def loadScaleDataMultivariate(
    assetList,
    loadpath,
    startDate=dt.datetime(1999, 1, 6),
    endDate=dt.datetime(2008, 12, 31),
):
    def getHeader(loadpath):

        header = pd.read_excel(loadpath + "no_trade.xls")
        header = header.set_index("date")
        header = header.drop(["BTI", "GSK", "ITT", "TM", "UVV"], axis=1)
        try:
            header = header.drop(["Unnamed: 0"], axis=1)
        except:
            pass

        columnNames = []
        for asset in header.columns.tolist()[:-1]:
            index = list(header.columns).index(asset)
            columnNames.append(asset + "-" + asset)
            for crossAsset in header.columns[index + 1 :].tolist():
                columnNames.append(asset + "-" + crossAsset)

        columnNames.append(header.columns[-1] + "-" + header.columns[-1])
        columnNames.sort()

        return columnNames

    noAssets = len(assetList)
    data = pd.read_csv(
        loadpath + "RVOC_6m.csv", engine="python", skiprows=[1], index_col="Var1"
    )
    data = data.drop("Unnamed: 0", axis=1)
    data.columns = getHeader(loadpath)
    data.index = pd.to_datetime(data.index, format="%Y%m%d")
    assert startDate in list(data.index), str(startDate) + " not in index"
    assert endDate in list(data.index), str(endDate) + " not in index"
    data = data[pd.Timestamp(startDate) : pd.Timestamp(endDate)]

    varVectorList = []
    for i in range(1, len(data.index)):
        varVector = createVarianceVector(data, assetList, i)
        covMat = np.real(matrix_logarithm(covMatFromVector(varVector, noAssets)))
        varVector = varVectorFromCovMat(covMat).reshape(1, -1, order="C")
        varVectorList.append(varVector)
    varVectorData = np.concatenate(varVectorList, axis=0)

    scaler = MinMaxScaler(feature_range=(0.0, 1))
    scaledTimeSeries = scaler.fit_transform(varVectorData)

    return Data(scaledTimeSeries, scaler, noAssets, "Multivariate", data.index)


def calculateErrorVectors(
    data, model, forecastHorizon, windowMode, windowSize=None, silent=True, startInd=25
):
    # Calculate the forecasting errors

    if model.modelType == "LSTM":
        data.createLSTMDataSet(model.lookBack)

    finalInd = data.scaledTimeSeries.shape[0] - forecastHorizon - data.maxLookBack

    def modelForecast(model, index):

        forecast, actual = model.multi_step_ahead_forecast(
            data, forecastHorizon, index, windowMode, windowSize
        )

        forecast = data.scaler.inverse_transform(forecast)
        actual = data.scaler.inverse_transform(actual)

        if data.nameDataSet == "OxfordMan":
            actual = np.exp(actual)
            forecast = np.exp(forecast)

        if data.nameDataSet == "Multivariate":
            forecast = np.array(
                [
                    matrix_exponential(covMatFromVector(vector, data.noAssets))
                    for vector in forecast
                ]
            )
            actual = np.array(
                [
                    matrix_exponential(covMatFromVector(vector, data.noAssets))
                    for vector in actual
                ]
            )

        assert actual.shape == forecast.shape

        return actual, forecast

    errorTypesList = ["RMSE", "QLIK", "L1Norm"]
    errorMatrices = {"RMSE": [], "QLIK": [], "L1Norm": []}
    errorOneDay = {"RMSE": [], "QLIK": [], "L1Norm": []}
    avgErrorVectors = dict.fromkeys(errorTypesList)

    def calculateForecastingError(errorType, actual, forecast):
        # Calculate the forecasting errors

        def RMSE(i):
            return np.matmul(
                (actual[i : i + 1].flatten() - forecast[i : i + 1].flatten()),
                (actual[i : i + 1].flatten() - forecast[i : i + 1].flatten()).T,
            )

        def QLIK(i):
            (sign, logdet) = np.linalg.slogdet(forecast[i] * 10000)

            result = logdet + np.trace(
                np.matmul(np.linalg.inv(forecast[i] * 10000), actual[i] * 10000)
            )
            return result

        def L1Norm(i):
            return np.linalg.norm((actual[i] - forecast[i]), ord=1)

        errorVector = [0]
        for i in range(0, forecast.shape[0]):
            try:
                errorVector.append(eval(errorType + "(i)"))
            except:
                print("Error when calculating" + errorType)

        return (
            np.clip(errorVector[1:], a_min=None, a_max=1).reshape(-1, 1),
            errorVector[1],
        )

    for index in range(startInd, finalInd):

        if silent is False and (index % 100 == 0 or index == finalInd - 1):
            print(
                model.modelName
                + " Evaluation is at index: "
                + str(index)
                + "/ "
                + str(finalInd - 1)
            )

        actual, forecast = modelForecast(model, index)

        for errorType in errorTypesList:
            oneDayError, errorVector = calculateForecastingError(
                errorType, actual, forecast
            )
            errorMatrices[errorType].append(oneDayError)
            errorOneDay[errorType].append(errorVector)

    avgErrorVectors["RMSE"] = np.sqrt(errorMatrices["RMSE"])
    for errorType in errorTypesList:
        avgErrorVectors[errorType] = np.mean(
            np.concatenate(errorMatrices[errorType], axis=1), axis=1, keepdims=True
        )

        def showForecast(errorType):
            # Debug Function

            avgErrorVector = avgErrorVectors[errorType]
            errorMatrix = errorMatrices[errorType]

            ax2 = plt.subplot(2, 1, 2)
            for i in range(0, len(errorMatrix)):
                shade = str(i / (len(errorMatrix) + 0.1))
                ax2.plot(np.sqrt(errorMatrix[i]), color=shade, linestyle="dotted")
            ax2.plot(avgErrorVector, color="blue", marker="x")
            ax2.set_title("Error Vectors.")

            ax3 = plt.subplot(2, 1, 3)
            ax3.set_title("Error Vector Avg. Index: " + str(index))
            ax3.plot(avgErrorVector, color="blue", marker="x")

            plt.tight_layout()
            plt.show()

    return ErrorMetrics(
        avgErrorVectors,
        errorMatrices,
        model.modelName,
        model.modelType,
        (finalInd - startInd + forecastHorizon),
        errorOneDay,
    )


def limitCPU(cpuLimit):
    # Limit CPU usage to protect hardware

    try:
        limitCommand = "cpulimit --pid " + str(getpid()) + " --limit " + str(cpuLimit)
        Popen(limitCommand, shell=True)
        print("CPU Limit at " + str(cpuLimit))
    except:
        print("Limiting CPU Usage failed")
