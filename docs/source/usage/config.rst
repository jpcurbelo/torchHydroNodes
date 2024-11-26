Configuration Arguments
=======================

This page provides a list of possible configuration arguments.
For examples of how a config file could look like, check out the files:

-  `tutorials/01-RunConceptModel/config_run_concept.yml <https://github.com/jpcurbelo/torchHydroNodes/blob/master/tutorials/01-RunConceptModel/config_run_concept.yml>`_

-  `tutorials/02-PretrainNNmodel/config_pretrain_nn_model.yml <https://github.com/jpcurbelo/torchHydroNodes/blob/master/tutorials/02-PretrainNNmodel/config_run_pretrain.yml>`_

Data entries
------------

- ``dataset``: Defines which dataset will be used. Currently supported is ``camelsus`` (CAMELS-US dataset by `Newman et al., 2015 <https://hess.copernicus.org/articles/19/209/2015/>`_. The code is intended to support other datasets but might require specific adaptations, such as extending the parent class to handle differences in dataset structure or features.

- ``concept_data_dir``: Specifies the path to the data source required for the conceptual model. This path should be defined in the configuration file ``src/utils/data_dir.yml``.

- ``forcings``: This entry can be ignored if the dataset is not ``camelsus`` or unless it is strictly required by a newly defined dataset. It can be either a string or a list of strings corresponding to forcing products in the CAMELS dataset. 

  Examples:  ``[daymet, maurer, maurer_extended, nldas]``.


General experiment entries
--------------------------

- ``basin_file``: Specifies the full or relative path to a text file containing the basin IDs used for training, validation, and testing. Each line in the file should contain a single basin ID, as defined in the dataset. 

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

  ``exphydro``: A two-bucket model (water and snow) with 5 processes and 6 parameters. (`Höge et al., 2022. <https://hess.copernicus.org/articles/26/5085/2022/>`_)

- ``ode_solver_lib``: Specifies the library used for solving ODEs. Supported options include ``scipy`` and ``torchdiffeq``. 

  - ``scipy``: Solves ODEs using ``solve_ivp`` from ``scipy.integrate``.  
    Reference: `SciPy Documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_.

    Supported methods: ``RK45``, ``RK23``, ``DOP853``, ``Radau``, ``BDF``, ``LSODA``.  

    Example:

    For adaptive-step solvers:

    - ``ode_solver_lib``: ``scipy``
    - ``odesmethod``: ``RK23``  
    - ``rtol``: ``1e-4``  
    - ``atol``: ``1e-6``  

    **Note**: Methods such as ``euler`` and ``rk4`` are not part of the `scipy` module and have been separately implemented in the model class.  

    For fixed-step solvers:

    - ``ode_solver_lib``: ``scipy``
    - ``odesmethod``: ``euler``  
    - ``time_step``: ``0.5``  

  - ``torchdiffeq``: Solves ODEs using the ``torchdiffeq`` library.  
    Reference: `torchdiffeq documentation <https://github.com/rtqichen/torchdiffeq/blob/master/README.md>`_.  
    Supported methods: ``euler``, ``rk4``, ``midpoint``, ``adaptive_heun``, ``bosh3``, ``dopri5``. 

    Example:

    - ``ode_solver_lib``: ``torchdiffeq``
    - ``odesmethod``: ``dopri5``  
    - ``rtol``: ``1e-4``  
    - ``atol``: ``1e-6``

Neural network entries
----------------------

- ``data_dir``: Specifies the folder that contains the data obtained by running the conceptual model. The path should be: ``src/data/data_dir`` - beware of locating the data in the correct folder.

- ``nn_model``: Specifies the neural network model to use. Supported models include ``mlp`` and ``lstm``. The code is intended to support other neural network models but might require specific adaptations.

  ``mlp``: A multi-layer perceptron model with fully connected layers.  

  ``lstm``: A Long Short-Term Memory model.

- ``hidden_size``: Specifies the number of hidden units in each layer of the neural network.  

  Example: ``[32, 32, 32, 32, 32]``

- ``seq_length``: Length of the input sequence. Only required for LSTM models.

- ``nn_dynamic_inputs``: Specifies the dynamic inputs to the neural network.  

  Example: ``[s_snow, s_water, prcp, tmean]``

- ``nn_mech_targets``: Specifies the mechanistic targets to the neural network (neural network outputs).  

  Example: ``[ps_bucket, pr_bucket, m_bucket, et_bucket, q_bucket]``

- ``target_variables``: Specifies the main target variables for the neural network - the one that will be used to train the model.  

  Example: ``[obs_runoff]``

  **Note**: The ``nn_dynamic_inputs``, ``nn_mech_targets``, adn ``target_variables`` entries should be consistent with the variables in the ``dataset`` and be inluded as ``model_inputs``, ``nn_mech_targets``, and ``target_variables``, respectively, in the ``concept_model`` entry definded in the file ``src/utils/concept_model_vars.yml``.

- ``loss_pretrain``: Specifies the loss function to use during the pre-training phase. Supported options include `nse` and `mae`, but the code is intended to support other loss functions.

- ``lr_pretrain``: Specifies the learning rate for the pre-training phase.

- ``epochs_pretrain``: Specifies the number of epochs for the pre-training phase.



Hybrid model entries
--------------------

``data_dir``: Same as in the **Neural network entries**.

- ``hybrid_model``: Specifies the hybrid model to use. Supported models include ``exphydroM100``. The code is intended to support other hybrid models but might require specific adaptations.

  ``exphydroM100``: A hybrid model that combines a conceptual model with a neural network model. (`Höge et al., 2022. <https://hess.copernicus.org/articles/26/5085/2022/>`_). See ``class ExpHydroM100`` in `src/modelzoo_hybrid/exphydroM100.py` for more details.

- ``concept_model``: Same as in the **Conceptual model entries**.

- ``ode_solver_lib``: Same as in the **Conceptual model entries** but only ``torchdiffeq`` is supported for hybrid models.

- ``basin_file``: Same as in the **General experiment entries**.

- ``nn_model_dir``: Specifies the path to the pre-trained neural network model. 

  **Note**: If ``nn_model_dir`` is not specified, the model will be trained from scratch and all the **Neural network entries** should be defined in the configuration file.

- ``scale_target_vars``: Specifies whether to scale the target variables. If set to `True`, the target variables will be scaled using the `mea` and `standard deviation` of the training period.

- ``loss``: Specifies the loss function to use. Supported options include `mse`, `nse`, and `nse-nh`.

- ``epochs``: Specifies the number of epochs to train the model.

- ``patience``: Specifies the patience for early stopping.

- ``clip_gradient_norm``: If a value, clips the gradients during training to that norm.

- ``batch_size``: Specifies the batch size for training. If set to `-1`, the whole dataset will be used in a single batch.

- ``optimizer``: Specifies the optimizer to use. Supported options include `adam` and `sgd`.

- ``learning_rate``: Learning rate. Can be either a single number (for a constant learning rate) or a dictionary. See `How to adjust learning rate <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_ in the Pytorch documentation for more information.

  Example:  

  ``learning_rate``:

  - ``initial``: ``0.001``
  - ``decay``: ``0.5`` 
  - ``decay_step_fraction``: ``2`` 

  **Note**: The learning rate will be decayed by a factor of ``decay`` every ``decay_step_fraction`` epochs.

- ``log_n_basins``: Specifies the number of basins to log during training. If set to `0`, no basins will be logged.

- ``log_every_n_epochs``: If a value and greater than `0`, logs figures and metrics, and saves the model after each `n` epochs.