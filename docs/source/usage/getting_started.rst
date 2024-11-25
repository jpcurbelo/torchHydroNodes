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
