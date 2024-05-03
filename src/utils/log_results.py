import os
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

from src.utils.metrics import (
    NSE_eval,
)



def save_plot_simulation(ds, q_bucket, basin, period='train', 
                         model_name='exphydro',
                         plots_dir=None,
                         plot_prcp=False):
    '''
    Plot the model simulations and observed values for a basin and save the plot.
    
    - Args:
        ds: xarray.Dataset, dataset with the input data.
        q_bucket: array_like, model predictions.
        basin: str, basin name.
        period: str, period of the run ('train', 'test', 'valid').
        plot_prcp: bool, whether to plot the precipitation rate.
        
    '''
    
    dates = ds['date']
    q_obs = ds['obs_runoff(mm/day)']            
        
    # Plot the simulated and actual values
    _, ax1 = plt.subplots(figsize=(16, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Discharge (mm/day)', color=color)
    ax1.plot(dates, q_obs, label='Observed', linewidth=3, color=color, zorder=2)
    ax1.plot(dates, q_bucket, ':', linewidth=3, label='Simulated', color='tab:red', zorder=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Set the major and minor locators and formatters for the x-axis
    years = YearLocator()   # every year
    months = MonthLocator()  # every month
    yearsFmt = DateFormatter('%Y')

    # Set the x-axis locators and formatters
    ax1.xaxis.set_major_locator(years)
    ax1.xaxis.set_major_formatter(yearsFmt)
    ax1.xaxis.set_minor_locator(months)
    
    # Set the x-axis limits
    start_date = dates.min()
    end_date = dates.max()
    ax1.set_xlim(start_date, end_date)
    # Enable autoscaling for the view
    ax1.autoscale_view()
        
    # Legend
    ax1.legend(loc='upper right')
    
    if plot_prcp:
        prcp = ds['prcp(mm/day)']
        # Create a twin Axes sharing the x-axis
        ax2 = ax1.twinx()
        color = 'lightslategray'
        ax2.set_ylabel('Precipitation Rate', color=color)
        ax2.plot(dates, prcp, label='Second Y-axis Data', color=color, zorder=1)
        ax2.tick_params(axis='y', labelcolor='darkslategray')
        ax2.invert_yaxis()  # Invert the y-axis of the second axis
            
    nse_val = NSE_eval(q_obs, q_bucket)
    plt.title(f'Model Predictions ({model_name}) | $NSE = {nse_val:.3f}$ | {period} period')
    
    plot_file_name = f'{basin}_{period.lower()}.png'
    
    plt.savefig(os.path.join(plots_dir, plot_file_name), bbox_inches='tight')
    plt.close()





if __name__ == "__main__":
    pass