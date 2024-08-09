import numpy as np
from scipy.sparse import lil_matrix
from scipy.ndimage import binary_dilation, binary_fill_holes
import pandas as pd
import csv

scale = 0.5

# Domain
x = 150*scale
z = 74*scale
y_air = 5*scale
y_moon = 50*scale
y = y_moon + y_air

y_cave = 10
x_co = 0.25
x_step = 0.1 # 0.09 m 
y_gpr = 0.1

# Reglith Layer
y_regolith = 4.5

# Crust Layer
y_crust = y_moon - y_regolith

# Frequency and Wavelength
f_max = 2*100e6 # Maximum Relevant Frequency for the Ricker Wavelet
c = 3e8

# Permittivity and Density
density = 2 # g/cm^3
eps_bg = 1.919**density # Background Permittivity for Moon
vel_bg = c / eps_bg**0.5

lambda_0 = c / (f_max * eps_bg**0.5)

# # Fresnel Zone - Radius
# R = 0.5 * (lambda_0 * d)**0.5
# y = 4 * R

# Grid Descretization
dx = np.round(lambda_0 / 17.3, 3) * 2
dy = np.round(lambda_0 / 17.3, 3) * 2
dz = np.round(lambda_0 / 17.3, 3) * 2

nx = int(x / dx)
ny = int(y / dy)
ny_m = int(y_moon / dy)
nz = int(z / dz)
print(f"nx: {nx}, ny: {ny}, nz: {nz}")

# x_grid, y_grid, z_grid = np.meshgrid(x_vals, y_vals, z_vals, indexing='xy')

# Time Window
time_window = 2 * y_moon / vel_bg

# Print
print(f"Domain: {x} x {y} x {z}")
print(f"Base Permittivity of Moon: {eps_bg}")
print(f"Frequency: {f_max/2}")
print(f"Wavelength: {lambda_0}")
# print(f"Fresnel Zone Radius: {R}")
print(f"Grid: {dx} x {dy} x {dz}")
print(f"Time Window: {time_window}")

# Editing the Lava Tube Points to Comply with the Domain

data_all = pd.read_csv('LavaTubeData/LiDAR_InclineCave_TUBE_TLS_points.txt', sep=' ', header=None)
# Convert to numpy arrays X, Y, Z
data_points = data_all.to_numpy()
X = (data_points[:, 0])*scale
Y = (data_points[:, 2])*scale
Z = (data_points[:, 1])*scale

# Find the minimum and maximum values of X, Y, Z, and find their index

X_min_index = X.argmin() 
X_min = X[X_min_index]
X_max_index = X.argmax()
X_max = X[X_max_index]
Y_min_index = Y.argmin()
Y_min = Y[Y_min_index]
Y_max_index = Y.argmax()
Y_max = Y[Y_max_index]
Z_min_index = Z.argmin()
Z_min = Z[Z_min_index]
Z_max_index = Z.argmax()
Z_max = Z[Z_max_index]

# Shift all points in X, Y, Z to the center of the domain
X = X - X_min
Y = Y - Y_min
Z = Z - Z_min

X_min_index = X.argmin() 
X_min = X[X_min_index]
X_max_index = X.argmax()
X_max = X[X_max_index]
Y_min_index = Y.argmin()
Y_min = Y[Y_min_index]
Y_max_index = Y.argmax()
Y_max = Y[Y_max_index]
Z_min_index = Z.argmin()
Z_min = Z[Z_min_index]
Z_max_index = Z.argmax()
Z_max = Z[Z_max_index]

X = X - (X_min + X_max)/2 + x/2
Y = Y - Y_max + y_moon - y_cave
Z = Z - (Z_min + Z_max)/2 + z/2

X_min_index = X.argmin() 
X_min = X[X_min_index]
X_max_index = X.argmax()
X_max = X[X_max_index]
Y_min_index = Y.argmin()
Y_min = Y[Y_min_index]
Y_max_index = Y.argmax()
Y_max = Y[Y_max_index]
Z_min_index = Z.argmin()
Z_min = Z[Z_min_index]
Z_max_index = Z.argmax()
Z_max = Z[Z_max_index]

print(f"X_min: {X_min}")
print(f"X_max: {X_max}")
print(f"Y_min: {Y_min}")
print(f"Y_max: {Y_max}")
print(f"Z_min: {Z_min}")
print(f"Z_max: {Z_max}")

data_all_shifted = pd.DataFrame(np.column_stack((X, Y, Z)))
data_all_shifted = data_all_shifted.to_numpy()

point_cloud = data_all_shifted

# Define the bounds of your grid
x_min, y_min, z_min = np.min(point_cloud, axis=0)
x_max, y_max, z_max = np.max(point_cloud, axis=0)

# Calculate the grid dimensions
nx = int(np.ceil((x_max - x_min) / dx))
ny = int(np.ceil((y_max - y_min) / dy))
nz = int(np.ceil((z_max - z_min) / dz))

# Initialize a dense voxel grid with zeros
voxel_grid = np.zeros((nx, ny, nz), dtype=int)

# Map each point to the nearest voxel
for point in point_cloud:
    ix = int((point[0] - x_min) // dx)
    iy = int((point[1] - y_min) // dy)
    iz = int((point[2] - z_min) // dz)
    voxel_grid[ix, iy, iz] = 1

# Fill the interior of the cave (assuming it's a closed surface)
filled_grid = binary_fill_holes(voxel_grid).astype(int)
dilated_grid = binary_dilation(filled_grid, iterations=1).astype(int)

# Tolerance value to ensure start < end
tolerance = 2 * min(dx, dy, dz)

# Function to find the end of the box
def find_box_end(x_start, y_start, z_start, voxel_grid, covered, min_size):
    x_end, y_end, z_end = x_start, y_start, z_start

    # Expand in the x direction
    while (x_end + 1 < voxel_grid.shape[0] and
           voxel_grid[x_end + 1, y_start:y_end + 1, z_start:z_end + 1].all() and
           not covered[x_end + 1, y_start:y_end + 1, z_start:z_end + 1].any()):
        x_end += 1
        if (x_end - x_start + 1) * dx >= min_size:
            break

    # Expand in the y direction
    while (y_end + 1 < voxel_grid.shape[1] and
           voxel_grid[x_start:x_end + 1, y_end + 1, z_start:z_end + 1].all() and
           not covered[x_start:x_end + 1, y_end + 1, z_start:z_end + 1].any()):
        y_end += 1
        if (y_end - y_start + 1) * dy >= min_size:
            break

    # Expand in the z direction
    while (z_end + 1 < voxel_grid.shape[2] and
           voxel_grid[x_start:x_end + 1, y_start:y_end + 1, z_end + 1].all() and
           not covered[x_start:x_end + 1, y_start:y_end + 1, z_end + 1].any()):
        z_end += 1
        if (z_end - z_start + 1) * dz >= min_size:
            break

    return x_end, y_end, z_end

# Function to fit boxes to the voxel grid (using grid indices)
def fit_boxes(voxel_grid, min_size):
    covered = np.zeros_like(voxel_grid, dtype=int)
    boxes = []

    for x in range(voxel_grid.shape[0]):
        for y in range(voxel_grid.shape[1]):
            for z in range(voxel_grid.shape[2]):
                if voxel_grid[x, y, z] and not covered[x, y, z]:
                    x_end, y_end, z_end = find_box_end(x, y, z, voxel_grid, covered, min_size)
                    
                    # Ensure box meets minimum size requirements
                    if (x_end - x + 1) * dx >= min_size and \
                       (y_end - y + 1) * dy >= min_size and \
                       (z_end - z + 1) * dz >= min_size:
                        boxes.append(((x, y, z), (x_end, y_end, z_end)))
                        covered[x:x_end+1, y:y_end+1, z:z_end+1] = 1
    
    return boxes

# Run the fit_boxes function to get the boxes in grid coordinates
min_box_size = 2 * min(dx, dy, dz)
boxes_grid = fit_boxes(dilated_grid, min_box_size)

# Convert grid indices to real-world coordinates and ensure dimensions are valid
boxes_real = []
for start, end in boxes_grid:
    start_x = min(start[0], end[0])
    start_y = min(start[1], end[1])
    start_z = min(start[2], end[2])
    end_x = max(start[0], end[0])
    end_y = max(start[1], end[1])
    end_z = max(start[2], end[2])

    # Ensure box dimensions are valid and non-zero
    if (end_x - start_x) < 1:
        end_x = start_x + 1
    if (end_y - start_y) < 1:
        end_y = start_y + 1
    if (end_z - start_z) < 1:
        end_z = start_z + 1

    box_start = (start_x * dx + x_min, start_y * dy + y_min, start_z * dz + z_min)
    box_end = (end_x * dx + x_min, end_y * dy + y_min, end_z * dz + z_min)
    
    # Ensure real-world end coordinates are strictly greater than start coordinates
    if abs(box_end[0] - box_start[0]) < tolerance:
        box_end = (box_start[0] + tolerance, box_end[1], box_end[2])
    if abs(box_end[1] - box_start[1]) < tolerance:
        box_end = (box_end[0], box_start[1] + tolerance, box_end[2])
    if abs(box_end[2] - box_start[2]) < tolerance:
        box_end = (box_end[0], box_end[1], box_start[2] + tolerance)
    
    boxes_real.append((box_start, box_end))

print(f"Found {len(boxes_real)} boxes")
# Export real-world coordinates to a CSV file
with open(f'boxes_{int(1/scale)}.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Start X', 'Start Y', 'Start Z', 'End X', 'End Y', 'End Z'])
    for start, end in boxes_real:
        writer.writerow([start[0], start[1], start[2], end[0], end[1], end[2]])

print(f"Boxes have been exported to 'boxes_{int(1/scale)}.csv'")
