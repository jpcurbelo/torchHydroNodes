import numpy as np
import spotpy
from spotpy.examples.spot_setup_hymod_python import spot_setup
import matplotlib.pyplot as plt

def initialize_sampler(objective_function=spotpy.objectivefunctions.rmse, maximize=False):
    """
    Initialize the SPOTPY sampler with SCEUA algorithm and set up configuration
    with a selectable objective function.
    
    Parameters:
    - objective_function: The objective function to be used (e.g., rmse, nashsutcliffe).
    - maximize: Boolean indicating if the objective function should be maximized.
    
    Returns:
    - setup: The spotpy setup instance.
    - sampler: The spotpy SCEUA sampler initialized with the setup.
    """
    setup = spot_setup(objective_function)
    sampler = spotpy.algorithms.sceua(setup, dbname='SCEUA_hymod', dbformat='csv')
    # sampler = spotpy.algorithms.lhs(setup, dbname='SCEUA_hymod', dbformat='csv')
    return setup, sampler, maximize

def run_sampler(sampler, rep=5000, ngs=7, kstop=3, peps=0.1, pcento=0.1):
    """Run the sampler with given parameters."""
    sampler.sample(rep, ngs=ngs, kstop=kstop, peps=peps, pcento=pcento)
    # sampler.sample(rep)
    results = spotpy.analyser.load_csv_results('SCEUA_hymod')
    return results

def plot_objective_function_trace(results):
    """Plot the Objective Function value across iterations for the objective function trace."""
    fig = plt.figure(1, figsize=(9, 5))
    plt.plot(results['like1'])
    plt.ylabel('Objective Function Value')
    plt.xlabel('Iteration')
    plt.show()
    fig.savefig('SCEUA_objectivefunctiontrace.png', dpi=150)

def get_best_model_run(results, maximize=False):
    """Retrieve the best model run and extract simulation data based on minimization or maximization."""
    if maximize:
        bestindex = np.argmax(results['like1'])  # Find the index of the maximum objective function value
        bestobjf = results['like1'][bestindex]
    else:
        bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)  # Minimize by default

    best_model_run = results[bestindex]

    # Retrieve fields starting with 'sim'
    fields = [word for word in best_model_run.dtype.names if word.startswith('sim')]
    best_simulation = list(best_model_run[fields])

    return best_simulation, bestobjf

def plot_best_model_run(best_simulation, bestobjf, setup):
    """Plot the best model simulation and observed data."""
    fig = plt.figure(figsize=(16, 9))
    ax = plt.subplot(1, 1, 1)
    ax.plot(best_simulation, color='black', linestyle='solid', label=f'Best objf.={bestobjf}')
    ax.plot(setup.evaluation(), 'r.', markersize=3, label='Observation data')
    plt.xlabel('Number of Observation Points')
    plt.ylabel('Discharge [l s-1]')
    plt.legend(loc='upper right')
    fig.savefig('SCEUA_best_modelrun.png', dpi=150)

def example_sceua(objective_function=spotpy.objectivefunctions.rmse, maximize=False):
    """
    Main example function to run the SCEUA algorithm and plot results.
    
    Parameters:
    - objective_function: The objective function to be used (e.g., rmse, nashsutcliffe).
    - maximize: Boolean indicating if the objective function should be maximized.
    """
    setup, sampler, maximize_flag = initialize_sampler(objective_function, maximize)
    results = run_sampler(sampler)
    plot_objective_function_trace(results)
    
    best_simulation, bestobjf = get_best_model_run(results, maximize=maximize_flag)
    plot_best_model_run(best_simulation, bestobjf, setup)

def nse_loss_minimized(observed, simulated):
    """
    Custom NSE function modified for minimization.
    
    Parameters:
    - observed: Array or list of observed values.
    - simulated: Array or list of simulated values.
    
    Returns:
    - Modified NSE value, with 0 indicating a perfect match.
    """
    # Convert observed and simulated to numpy arrays if they are not already
    observed = np.array(observed)
    simulated = np.array(simulated)
    
    observed_mean = np.mean(observed)
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - observed_mean) ** 2)

    return - (1 - numerator / denominator)

def main():
    # Use Nash-Sutcliffe Efficiency as the objective function and set maximize=True for testing
    # # example_sceua(objective_function=spotpy.objectivefunctions.nashsutcliffe, maximize=True)
    example_sceua(objective_function=nse_loss_minimized, maximize=False)


if __name__ == "__main__":
    main()
