<div align='center'>
  <img src='Pictures/ESN.png'>
</div>

## Echo State Networks

This repository implements [Echo State Networks](http://www.scholarpedia.org/article/Echo_state_network) in Python 3.6

### Getting Started

```py
import EchoStateNetworks
import dataUtils
from numpy import ones

# Initialize the ESN
my_ESN = EchoStateNetworks.EchoStateNetwork(no_input_nodes=4,
                                            no_output_nodes=4, 
                                            no_internal_nodes=30)

# Create data set and split it into test & training
data = Data(model_input=3*ones((100,4)), 
            model_output=4*ones((100,4)))
data.split_data_by_index(80)

# Fit the ESN
my_ESN.fit(data, no_forget_points=10)

# Do a 30 steps ahead multivariate forecast
multi_step_forecast, _ = my_ESN.forecast(data=data, forecast_horizon=30)

# Print the the 3 steps ahead of the forcecast
print(multi_step_forecast[0:3])
>> [[3.99883257 3.99883257 3.99883257 3.99883257]
>> [3.99886162 3.99886162 3.99886162 3.99886162]
>> [3.99886162 3.99886162 3.99886162 3.99886162]]
```

### Documentation
***
#### _class_ EchoStateNetwork(nInputNodes, nOutputNodes, internalNodes)
###### Input Arguments

* **no_input_nodes**, required, integer >=1, Sets the number of input time series for the ESN
* **no_output_nodes**, required, integer >=1, Sets the number of output time series for the ESN
* **no_internal_nodes**, required, integer >=1, Defines the size of the quadratic reservoir matrix
* _model_name_, optional, str, Name of the ESN
* _input_scaling_, optional, float >=0.0, default: 1.0, Scales the input
* _input_shift_,optional, float, default: 0.0, Shifts the input
* _regression_lambda_, optional, float >0.0, default: 1e-12, Regularization parameter for the regression
* _spectral_radius_, optional, float >0.0, default: 0.7, SpectralRadius for the reservoir (<1.2 recommended)
* _leaking_rate_, optional, float 1.0>x>0.0, default: 0.0, Inertia parameter for the reservoir state
* _connectivity_, optional, float 1.0>x>0.0, default: 10/internalNodes, Sparsity of the reservoir matrix
* _input_mask_, optional, float(nInputNodes,1), default: 1*(nInputNodes,1), Filter for the input
* _seed_, optional, int, seed for the random number generator that creates the reservoir
***
    
#### _class method_ EchoStateNetwork.fit(data)
###### Input Arguments

* **data**, required, _class_ Data, contains the data
* _no_forget_points_, optional, integer >0, default: 100, Initial reservoir states that should be forgotten
***
  
#### _class method_ EchoStateNetwork.forecast(data, forecast_horizon)
###### Input Arguments
* **data**, required, Data object
* **forecast_horizon**, required, integer >= 1, Set the amount of total time steps for iterative forecasting
* _no_samples_ , optional, integer >0, default: 10, Determines the number of samples drawn 
from the collected state matrizes

###### Output
* **multi_step_forecast**, the forecast by the ESN
* **actual**, the actual value of the test set (None if those values are not available)

#### _class method_ EchoStateNetwork.moving_window_forecasts(data, index_base)
This function allows to iterate over the data set and make a multi step ahead forecasts on each point in time.
The Network may be refitted every time moving forward on a a window of past data.  
* **data**, required, Data object
* **index_base**, required, int, index in the dataset on where to start
* **index_end**, optional, int, index in the dataset on where to end
* _forecast_horizon_, optional, integer >= 1, Set the amount of total time steps for iterative forecasting
* _window_mode_, optional, "Expanding","Fixed", "Rolling", default: "Expanding" Determines on how the backlog of data is being used.
* _window_size_, optional, integer >0, default: 20, Determines the size of the window

###### Output
* **forecast_collection**, dict, a dictionary with the base indices as keys and the respective 
multi step ahead forecast
***
#### hyper_parameter_grid_search(data, split_index, hyper_parameter_space)
Iterates over the hyper paramter space and yields the combination that 
has has the lowest error on the training set
* **data**, required, Data object
* **split_index**, required, int, index that determines the portion of the training set
* **hyper_parameter_space**, required, dict, Determines the hyper paramater space that is being searched on
Format: {"parameter_name": (min, max, steps), ...} e.g. {"internal_nodes": (30, 300, 10), ...}
* _show_status_, optional, bool, Determine whether you want to print the results of each run in the console

###### Output
* **min_error**, the minimum error on the test set
* **optimal_hyper_parameter**, the optimal set of hyper paramters that achieved the respective error

***
#### hyper_parameter_bayesian_optimization(data, split_index, hyper_parameter_space)
Searches the hyper paramter space using bayesian optimisation
* **data**, required, Data object
* **split_index**, required, int, index that determines the portion of the training set
* **hyper_parameter_space**, required, dict, Determines the hyper paramater space that is being searched on
Format: {"parameter_name": (min, max, steps), ...} e.g. {"internal_nodes": (30, 300, 10), ...}

###### Output
* **min_error**, the minimum error on the test set
The optimal hyper parameter are printed to the console

***
***
### Current ToDos for version 0.1.1
* Make the ESN independent of the Data class, it should be usable just with with numpy arrays
* Implement teacher forcing
* Improve the use case example

For bugs, feedback write to schreiner.lukas1 (at) gmail.com
***

### References
Developed in Python 3.6.9

The code in this file is inspired by an ESN implementation in MATLAB by H. Jaeger`

Jaeger, Herbert, "The echo state approach to analysing and training recurrent neural networks-with an erratum note",
Bonn, Germany: German National Research Center for Information Technology GMD Technical Report 148, 34 (2001), pp. 13.

The code was formatted using [Black](https://github.com/psf/black)