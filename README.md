<table>
  <tr>
    <td>
      <img src="docs/source/_static/img/torchhydronodes-logo.webp" alt="torchHydroNodes Logo" title="Logo designed with the help of AI via OpenAI's tools" width="200">
    </td>
    <td>
      <strong>torchHydroNodes</strong>: Python package to train hybrid hydrological models (conceptual/physics-based models + neural networks)
    </td>
  </tr>
</table>

## Key Features

# Getting Started

## Linux (Recommended Configuration)

Tested on 22.04

### Installation

#### 1. Clone GO-RXR from the main branch in this repository:
```bash
$ git clone https://github.com/jpcurbelo/torchHydroNodes.git
```

#### 2. Prerequisites and Setup (Tested with Python 3.12.13)

2.1. **Set Up the Virtual Environment**


  - **Navigate to the `torchHydroNodes` Directory**:
    ```bash
    $ cd torchHydroNodes
    ```

   - **Create the virtual environment**:  

      ```bash
      $ python3.12 -m venv venv-thn
      ```

   - **Activate the virtual environment**:  

     ```bash
     (venv-go-rxr) $ source venv-go-rxr/bin/activate
     ```

2.2. **Install Package Dependencies**

  <!-- Install the required Python packages using `pip`:

  ```bash
  (venv-go-rxr) $ pip install .
  ``` -->


## Data (download, location, and file configuration)

## Configuration Arguments

### Data entries

- ```dataset```: Defines which dataset will be used. Currently supported is ```camelsus``` (CAMELS-US dataset by [Newman et al., 2015](https://hess.copernicus.org/articles/19/209/2015/)). The code is intended to support other datasets but might require specific adaptations, such as extending the parent class to handle differences in dataset structure or features.

- ```concept_data_dir```: Specifies the path to the data source required for the conceptual model. This path should be defined in the configuration file ```src/utils/data_dir.yml```.

- ```forcings```: This entry can be ignored if the dataset is not ```camelsus``` or unless it is strictly required by a newly defined dataset. It can be either a string or a list of strings corresponding to forcing products in the CAMELS dataset. 

    *Examples*:  ```[daymet, maurer, maurer_extended, nldas]```.

### Conceptual model entries

- ```concept_model```: Specifies the conceptual model to use. Supported models include ```exphydro```. The code is intended to support other conceptual models but might require specific adaptations, such as extending the parent class to accommodate the specifics of a newly defined model.

  ```exphydro```: A two-bucket model (water and snow) with 5 processes and 6 parameters. Reference: [Höge et al., 2022.](https://hess.copernicus.org/articles/26/5085/2022/)

### Neural network entries

### Hybrid model entries

### General experiment entries

- **```basin_file```**: Specifies the full or relative path to a text file containing the basin IDs used for training, validation, and testing. Each line in the file should contain a single basin ID, as defined in the dataset.

- **```ode_solver_lib```**: Specifies the library used for solving ODEs. Supported options include ```scipy``` and ```torchdiffeq```. 

  - ```scipy```: Solves ODEs using ```solve_ivp``` from ```scipy.integrate```.  
    Reference: [SciPy Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).  
    Supported methods: ```RK45```, ```RK23```, ```DOP853```, ```Radau```, ```BDF```, ```LSODA```.  
    Example:
    - **```odesmethod```**: ```RK23```  
    - For adaptive solvers:
      - **```rtol```**: ```1e-4```  
      - **```atol```**: ```1e-6```  

    **Note**: Methods such as ```euler``` and ```rk4``` are not part of the `scipy` module and have been separately implemented in the model class.  
    Example:
    - **```odesmethod```**: ```euler```  
    - **```time_step```**: ```0.5```  

  - **```torchdiffeq```**: Solves ODEs using the ```torchdiffeq``` library.  
    Reference: [TorchDiffEq Documentation](https://github.com/rtqichen/torchdiffeq/blob/master/README.md).  
    Supported methods: ```bosh3```, ```rk4```, ```dopri5```, ```euler```, ```adaptive_heun```, ```midpoint```.  
    Example:
    - **```odesmethod```**: ```bosh3```  
    - **```rtol```**: ```1e-4```  
    - **```atol```**: ```1e-6```


- **`train_start_date`**: Start date of the training period (first day of discharge) in the format `DD/MM/YYYY`.  
  Corresponding pairs of start and end dates denote the different periods.

- **`train_end_date`**: End date of the training period (last day of discharge) in the format `DD/MM/YYYY`.  

- **`valid_start_date`**: Start date of the validation period (first day of discharge) in the format `DD/MM/YYYY`.  

- **`valid_end_date`**: End date of the validation period (last day of discharge) in the format `DD/MM/YYYY`.  


- **`metrics`**: Specifies the list of metrics to calculate during validation (testing).  
  Available metrics include: `NSE`, `Alpha-NSE`, `Beta-NSE`, `FHV`, `FMS`, `FLV`, `KGE`, `Beta-KGE`, `Peak-Timing`, `Peak-MAPE`, `Pearson-r`.

  **Reference**: For a full list of available metrics, see `src/utils/metrics`.

- **```experiment_name```**: Defines the name of your experiment that will be used as a folder name (+ date-time string),

- **```device```**: Which device to use in format of ```cuda:0```, ```cuda:1```, etc, for GPUs or ```cpu```.

- **```seed```**: Fixed random seed. If empty, a random seed is generated for this run.

- **`precision`**: Sets the precision for the model.  
  Supported options: `float32`, `float64`.  

- **`verbose`**: Specifies the verbosity level of the model's logging and progress display.  
  - `0`: Only log informational messages; progress bars are not shown.  
  - `1`: Show progress bars along with informational messages.  
