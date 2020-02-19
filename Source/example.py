import os, sys
import numpy as np
import dataUtils
import EchoStateNetworks
sys.path.append(os.path.join(os.getcwd()))

if __name__ == "__main__":
    my_esn = EchoStateNetworks.EchoStateNetwork(no_input_nodes=3,
                              no_output_nodes=2,
                              no_internal_nodes=100,
                              spectral_radius=0.7)

    data = dataUtils.Data(time_series_data_input=np.random.rand(100, 3))

    data.split_data_by_index(50)
    my_esn.fit(data, 10)
    my_esn.evaluate(data, True)

