import h5py
import numpy as np
import matplotlib.pyplot as plt

# Function to load data from gprMax output file
def load_gprmax_output(file_path, snapshot_index):
    with h5py.File(file_path, 'r') as f:
        # Example for electric field component 'Ex'
        data = f['Ex'][snapshot_index, :, :]
    return data

# Function to plot wavefield snapshot
def plot_wavefield_snapshot(data, title='Wavefield Snapshot', cmap='viridis'):
    plt.imshow(data, cmap=cmap, aspect='auto')
    plt.colorbar(label='Electric Field (V/m)')
    plt.title(title)
    plt.xlabel('X (grid points)')
    plt.ylabel('Y (grid points)')
    plt.show()

# Example usage
file_path = 'your_gprmax_output.out'  # Replace with your actual file path
snapshot_index = 0  # Index of the snapshot you want to plot

# Load data
data = load_gprmax_output(file_path, snapshot_index)

# Plot snapshot
plot_wavefield_snapshot(data, title=f'Wavefield Snapshot at Index {snapshot_index}')
