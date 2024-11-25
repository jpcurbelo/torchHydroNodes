Getting started
===============

Installation
------------
Recommended Configuration: Tested on Ubuntu 22.04 with Python 3.12.

1. Clone the repository from the main branch:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   .. code-block:: bash

      $ git clone https://github.com/jpcurbelo/torchHydroNodes.git

2. Prerequisites and Setup (Tested with Python 3.12.13)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.1. Set Up the Virtual Environment

- Navigate to the ``torchHydroNodes`` directory:

   .. code-block:: bash

      $ cd torchHydroNodes

- Create the virtual environment:

   .. code-block:: bash

      $ python3.12 -m venv venv-thn

- Activate the virtual environment:

    .. code-block:: bash

      $ source venv-thn/bin/activate

  2.2. Install Package Dependencies

  - Install the required packages using `pip`:

      .. code-block:: bash

        (venv-thn) $ pip install .

    This command will handle the installation of all dependencies specified in the `pyproject.toml` file. 

  - If you encounter any issues, you can manually install the dependencies listed in the `requirements.txt` file using:

      .. code-block:: bash

        (venv-thn) $ pip install -r requirements.txt

3. Data Preparation
^^^^^^^^^^^^^^^^^^^

- Download and place your dataset in a convenient directory.

  **Note**: For the CAMELS US dataset, we recommend referring to the entries on meteorological time series, streamflow data, and catchment attributes in the `Tutorial on Data Prerequisites <https://neuralhydrology.readthedocs.io/en/latest/tutorials/data-prerequisites.html>`_ within the `NeuralHydrology documentation <https://neuralhydrology.readthedocs.io>`_.

- Define or update the corresponding ``data_dir`` entry in the ``src/utils/data_dir.yml`` file.

  **Note**: In the default configurations of our tutorials, the key ``data_dir`` is defined as:

  .. code-block:: yaml

      data_dir_camelsus: /gladwell/hydrology/SUMMA/summa-ml-models/CAMELS_US

  This path may vary depending on your setup.

- Ensure that the shapefiles (`.shp`) required for plotting are specified in the ``data_dir.yml`` file. These files are dataset-specific and are used for geographic visualization of catchment boundaries or other spatial attributes. 

  **For example:**

  .. code-block:: yaml

      # Relative path after the data_dir
      map_shape_file: basin_set_full_res/usa-states-census-2014.shp
      hm_catchment_file: basin_set_full_res/HCDN_nhru_final_671.shp
