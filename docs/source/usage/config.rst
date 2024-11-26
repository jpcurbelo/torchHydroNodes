Configuration Arguments
=======================

This page provides a list of possible configuration arguments.
For examples of how a config file could look like, check out the files:

-  `docs/source/tutorials/run_concept_model/config_run_concept.yml <https://github.com/jpcurbelo/torchHydroNodes/blob/master/docs/source/tutorials/run_concept_model/config_run_concept.yml>`__ 
-  `docs/source/tutorials/run_hybrid_model/config_run_hybrid_mlp.yml <https://github.com/jpcurbelo/torchHydroNodes/blob/master/docs/source/tutorials/run_hybrid_model/config_run_hybrid_mlp.yml>`__ 

Data entries
------------

- ``dataset``: Defines which dataset will be used. Currently supported is ``camelsus`` (CAMELS-US dataset by `Newman et al., 2015 <https://hess.copernicus.org/articles/19/209/2015/>`__. The code is intended to support other datasets but might require specific adaptations, such as extending the parent class to handle differences in dataset structure or features.

- ``concept_data_dir``: Specifies the path to the data source required for the conceptual model. This path should be defined in the configuration file ``src/utils/data_dir.yml``.

- ``forcings``: This entry can be ignored if the dataset is not ``camelsus`` or unless it is strictly required by a newly defined dataset. It can be either a string or a list of strings corresponding to forcing products in the CAMELS dataset. 

  *Examples*:  ``[daymet, maurer, maurer_extended, nldas]``.


General experiment entries
--------------------------

- ``basin_file``: Specifies the full or relative path to a text file containing the basin IDs used for training, validation, and testing. Each line in the file should contain a single basin ID, as defined in the dataset.

- ``ode_solver_lib``: Specifies the library used for solving ODEs. Supported options include ``scipy`` and ``torchdiffeq``. 

  - ``scipy``: Solves ODEs using ``solve_ivp`` from ``scipy.integrate``.  
    Reference: `SciPy Documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`__.  
    Supported methods: ``RK45``, ``RK23``, ``DOP853``, ``Radau``, ``BDF``, ``LSODA``.  

    Example:

    - For adaptive-step solvers:

      - ``odesmethod``: ``RK23``  
      - ``rtol``: ``1e-4``  
      - ``atol``: ``1e-6``  

    **Note**: Methods such as ``euler`` and ``rk4`` are not part of the `scipy` module and have been separately implemented in the model class.  

    Example:

    - For fixed-step solvers:

      - ``odesmethod``: ``euler``  
      - ``time_step``: ``0.5``  

  - ``torchdiffeq``: Solves ODEs using the ``torchdiffeq`` library.  
    Reference: `torchdiffeq documentation <https://github.com/rtqichen/torchdiffeq/blob/master/README.md>`__.  
    Supported methods: ``euler``, ``rk4``, ``midpoint``, ``adaptive_heun``, ``bosh3``,``dopri5``.  

- ``train_start_date``: Start date of the training period (first day of discharge) in the format `DD/MM/YYYY`.  
  Corresponding pairs of start and end dates denote the different periods.

- ``train_end_date``: End date of the training period (last day of discharge) in the format `DD/MM/YYYY`.  

- ``valid_start_date``: Start date of the validation period (first day of discharge) in the format `DD/MM/YYYY`.  

- ``valid_end_date``: End date of the validation period (last day of discharge) in the format `DD/MM/YYYY`.  


- ``metrics``: Specifies the list of metrics to calculate during validation (testing).  
  Available metrics include: `NSE`, `Alpha-NSE`, `Beta-NSE`, `FHV`, `FMS`, `FLV`, `KGE`, `Beta-KGE`, `Peak-Timing`, `Peak-MAPE`, `Pearson-r`.

  **Reference**: For a full list of available metrics, see `src/utils/metrics`.

- ``experiment_name``: Defines the name of your experiment that will be used as a folder name (+ date-time string suffix) to save the model and results.,

- ``device``: Which device to use in format of ``cuda:0``, ``cuda:1``, etc, for GPUs or ``cpu``.

- ``seed``: Fixed random seed. If empty, a random seed is generated for this run.

- ``precision``: Sets the precision for the model.  
  Supported options: `float32`, `float64`.  

- ``verbose``: Specifies the verbosity level of the model's logging and progress display.  
  - ``0``: Only log informational messages; progress bars are not shown.  
  - ``1``: Show progress bars along with informational messages.  



Conceptual model entries
------------------------

- ``concept_model``: Specifies the conceptual model to use. Supported models include ``exphydro``. The code is intended to support other conceptual models but might require specific adaptations, such as extending the parent class to accommodate the specifics of a newly defined model.

  ``exphydro``: A two-bucket model (water and snow) with 5 processes and 6 parameters. (`Höge et al., 2022. <https://hess.copernicus.org/articles/26/5085/2022/>`__)

Neural network entries
----------------------

Hybrid model entries
--------------------