
import pandas as pd
from pathlib import Path
import sys

# Make sure code directory is in path,
# Add the parent directory of your project to the Python path
src_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(src_dir)

def load_and_get_boundaries(csv_file, fraction=0.15):
    # Load DataFrame from CSV file
    df = pd.read_csv(csv_file)

    # Get min and max for each column
    min_values = df.min()
    max_values = df.max()

    # Extend limits by a given fraction
    low_boundaries = min_values - fraction * (max_values - min_values)
    high_boundaries = max_values + fraction * (max_values - min_values)

    # Return the boundaries as a DataFrame
    boundaries = pd.DataFrame({
        'min_values': min_values,
        'max_values': max_values,
        'low_boundaries': low_boundaries,
        'high_boundaries': high_boundaries
    })

    return boundaries


if __name__ == '__main__':

    # Example usage:
    csv_file = Path(src_dir) / 'modelzoo_concept' / 'bucket_parameter_files' / 'bucket_exphydro.csv'
    fraction = 0.15  # Extend by X% of the range

    boundaries = load_and_get_boundaries(csv_file, fraction)
    print(boundaries)


