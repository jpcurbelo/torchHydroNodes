
import xarray
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp as sp_solve_ivp

from src.modelzoo_hybrid.basemodel import BaseHybridModel
from src.modelzoo_nn.basemodel import BaseNNModel
from src.utils.load_process_data import (
    Config,
    ExpHydroCommon,
)

# Ref: exphydro -> https://hess.copernicus.org/articles/26/5085/2022/
class ExpHydroM100(BaseHybridModel, ExpHydroCommon):

    def __init__(self,
                 cfg: Config,
                 nnmodel: BaseNNModel,
                 ds: xarray.Dataset,
    ):
        super().__init__(cfg, nnmodel, ds)

        # Interpolators
        self.interpolators = self.create_interpolator_dict()

        # Parameters per basin
        self.params_dict = self.get_parameters()

    

    
    # def run(self, inputs, basin):

    #     # Get the interpolator functions for the basin
    #     self.precp_interp = self.interpolators[basin]['prcp']
    #     self.temp_interp = self.interpolators[basin]['tmean']
    #     self.lday_interp = self.interpolators[basin]['dayl']

    #     # Get the parameters for the basin
    #     basin_params = self.get_parameters(basin)

    #     # Set the initial conditions
    #     y0 = np.array([basin_params[0], basin_params[1]])

    #     # Set the parameters
    #     params = tuple(basin_params[2:])

    #     print(self.time_series)

    #     # # Run the model
    #     # y = sp_solve_ivp(self.hydro_odes_M100, t_span=(0, self.precp.shape[1] - 1), y0=y0, t_eval=self.time_series, 
    #     #                  args=params, 
    #     #                  method=self.odesmethod,
    #     #                 #  method='DOP853',
    #     #                 # rtol=1e-9, atol=1e-12,
    #     #             )