import pandas as pd
import numpy as np

scale = 1

# Domain
x = 200*scale
z = 100*scale
y_air = 1*scale
y_moon = 50*scale
y = y_moon + y_air


y_cave = 10*scale
x_co = 0.25*scale
x_step = 0.09/scale
y_gpr = 0.1*scale

# Reglith Layer
y_regolith = 4.5*scale

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
dx = np.round(lambda_0 / 17.3, 3)
dy = np.round(lambda_0 / 17.3, 3)
dz = np.round(lambda_0 / 17.3, 3)

nx = int(x / dx)
ny = int(y / dy)
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

print(f"y_regolith: {y_regolith}")
print(f"y_crust: {y_crust}")
print(f"y_air: {y_air}")
print(f"y_moon: {y_moon}")
print(f"y_cave: {y_cave}")
print(f"x_co: {x_co}")
print(f"x_step: {x_step}")
print(f"y_gpr: {y_gpr}")

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

# Save the new points to vtk
from pyevtk.hl import pointsToVTK
pointsToVTK("LiDAR_InclineCave_TUBE_adjusted", X, Y, Z)

from scipy.spatial import Delaunay

permittivity_values = 7.6* np.ones_like(X)


#Grid
# x_val = np.linspace(0, x, 5)
# z_val = np.linspace(0, z, 5)
# y_val = np.linspace(0, y_moon, 5)
x_val = np.linspace(0, x, int(nx*scale))
z_val = np.linspace(0, z, int(ny*scale))
y_val = np.linspace(0, y_moon, int(nz*scale))


meshgrid_x, meshgrid_y, meshgrid_z = np.meshgrid(x_val, y_val, z_val, indexing='ij')
points = np.vstack((meshgrid_x.flatten(), meshgrid_y.flatten(), meshgrid_z.flatten())).T

# Create Delaunay triangulation
tri = Delaunay(np.column_stack((X, Y, Z)))


# Check which points are within the convex hull
# Check for interpolation
simplex_indices = tri.find_simplex(points)
mask = simplex_indices >= 0


# x, y, z for just the mask
x_mask = meshgrid_x.flatten()[mask]
y_mask = meshgrid_y.flatten()[mask]
z_mask = meshgrid_z.flatten()[mask]
points_cave_edge = np.vstack((x_mask, y_mask, z_mask)).T
print(f"Number of Points in Convex Hull: {len(z_mask)}")
print(f"Number of Points in Total: {len(points_cave_edge)}")


from scipy.spatial import ConvexHull

# Calculate the centroid of the original points
centroid_cave = np.mean(points_cave_edge, axis=0)

# Function to shrink each point towards the centroid
def shrink_point(point, centroid, dx=2*dx, dy=2*dy, dz=2*dz):
    direction = point - centroid
    factor = np.array([dx, dy, dz]) / np.linalg.norm(direction)
    shrink_vector = direction * factor
    return point - shrink_vector

# Apply shrinking to all points
points_cave_vacuum_all = np.array([shrink_point(p, centroid_cave, dx, dy, dz) for p in points_cave_edge])

# Compute the new convex hull from shrunken points
convex_cave_vacuum = ConvexHull(points_cave_vacuum_all)

# Extract the new set of coordinates defining the new convex hull
points_cave_vacuum = points_cave_vacuum_all[convex_cave_vacuum.vertices]
print(f"Number of Points in Vacuum Convex Hull: {len(points_cave_vacuum)}")


with open('mare_tube_ring.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/2 - x_co} {y_moon} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 13 17.8 2.2 3.2 13.5 15.5 7.5 12.5 40.5 47.5 8.5 12.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 19 22 1 6 6 10.5 6 11.5 42 47 9.5 10.5 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    eps_r_value = 7.6
    sigma_value = 1e-10
    mu_r_value = 1.0
    sigma_m_value = 0.0
    f.write(f"#material: {eps_r_value} {sigma_value} {mu_r_value} {sigma_m_value} moon_lava_tube \n")
    # Dielectric smoothing needs to be turned off, add higher discretization for numerical stability

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} moon_lava_tube n\n")

    for i in range(len(points_cave_vacuum_all)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {points_cave_vacuum_all[i][0]} {points_cave_vacuum_all[i][1]} {points_cave_vacuum_all[i][2]} {points_cave_vacuum_all[i][0]+dx} {points_cave_vacuum_all[i][1]+dy} {points_cave_vacuum_all[i][2]+dz} free_space n\n")

    f.write(f"#rx: {x/2 + x_co} {y_moon} {z/2}\n")
    f.write(f"#geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} mare_tube_ring n \n")

# Output success message
print("gprMax input file 'mare_tube_ring.in' generated.")


with open('mare_tube_vacuum.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/2 - x_co} {y_moon} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 13 17.8 2.2 3.2 13.5 15.5 7.5 12.5 40.5 47.5 8.5 12.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 19 22 1 6 6 10.5 6 11.5 42 47 9.5 10.5 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} free_space n\n")

    f.write(f"#rx: {x/2 + x_co} {y_moon} {z/2}\n")
    f.write(f"#geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} mare_tube_vacuum n \n")

# Output success message
print("gprMax input file 'mare_tube_vacuum.in' generated.")


with open('mare_tube_ring_B_0_0.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 13 17.8 2.2 3.2 13.5 15.5 7.5 12.5 40.5 47.5 8.5 12.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 19 22 1 6 6 10.5 6 11.5 42 47 9.5 10.5 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    eps_r_value = 7.6
    sigma_value = 1e-10
    mu_r_value = 1.0
    sigma_m_value = 0.0
    f.write(f"#material: {eps_r_value} {sigma_value} {mu_r_value} {sigma_m_value} moon_lava_tube \n")
    # Dielectric smoothing needs to be turned off, add higher discretization for numerical stability

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} moon_lava_tube n\n")

    for i in range(len(points_cave_vacuum_all)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {points_cave_vacuum_all[i][0]} {points_cave_vacuum_all[i][1]} {points_cave_vacuum_all[i][2]} {points_cave_vacuum_all[i][0]+dx} {points_cave_vacuum_all[i][1]+dy} {points_cave_vacuum_all[i][2]+dz} free_space n\n")

    f.write(f"#rx: {x/10} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"#geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} mare_tube_ring_B_0_0 n \n")

# Output success message
print("gprMax input file 'mare_tube_ring_B_0_0.in' generated.")


with open('mare_tube_ring_B_0_01.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon + y_gpr} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 13 17.8 2.2 3.2 13.5 15.5 7.5 12.5 40.5 47.5 8.5 12.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 19 22 1 6 6 10.5 6 11.5 42 47 9.5 10.5 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    f.write(f"#material: {eps_r_value} {sigma_value} {mu_r_value} {sigma_m_value} moon_lava_tube \n")
    # Dielectric smoothing needs to be turned off, add higher discretization for numerical stability

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} moon_lava_tube n\n")

    for i in range(len(points_cave_vacuum_all)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {points_cave_vacuum_all[i][0]} {points_cave_vacuum_all[i][1]} {points_cave_vacuum_all[i][2]} {points_cave_vacuum_all[i][0]+dx} {points_cave_vacuum_all[i][1]+dy} {points_cave_vacuum_all[i][2]+dz} free_space n\n")

    f.write(f"#rx: {x/10} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} mare_tube_ring_B_0_01 n \n")

# Output success message
print("gprMax input file 'mare_tube_ring_B_0_01.in' generated.")

with open('mare_tube_ring_B_025_01.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon + y_gpr} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 13 17.8 2.2 3.2 13.5 15.5 7.5 12.5 40.5 47.5 8.5 12.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 19 22 1 6 6 10.5 6 11.5 42 47 9.5 10.5 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    eps_r_value = 7.6
    sigma_value = 1e-10
    mu_r_value = 1.0
    sigma_m_value = 0.0
    f.write(f"#material: {eps_r_value} {sigma_value} {mu_r_value} {sigma_m_value} moon_lava_tube \n")
    # Dielectric smoothing needs to be turned off, add higher discretization for numerical stability

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} moon_lava_tube n\n")

    for i in range(len(points_cave_vacuum_all)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {points_cave_vacuum_all[i][0]} {points_cave_vacuum_all[i][1]} {points_cave_vacuum_all[i][2]} {points_cave_vacuum_all[i][0]+dx} {points_cave_vacuum_all[i][1]+dy} {points_cave_vacuum_all[i][2]+dz} free_space n\n")

    f.write(f"#rx: {x/2 + x_co} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} mare_tube_ring_B_025_01 n \n")

# Output success message
print("gprMax input file 'mare_tube_ring_B_025_01.in' generated.")

with open('mare_tube_ring_B_025_0.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 13 17.8 2.2 3.2 13.5 15.5 7.5 12.5 40.5 47.5 8.5 12.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 19 22 1 6 6 10.5 6 11.5 42 47 9.5 10.5 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    eps_r_value = 7.6
    sigma_value = 1e-10
    mu_r_value = 1.0
    sigma_m_value = 0.0
    f.write(f"#material: {eps_r_value} {sigma_value} {mu_r_value} {sigma_m_value} moon_lava_tube \n")
    # Dielectric smoothing needs to be turned off, add higher discretization for numerical stability

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} moon_lava_tube n\n")

    for i in range(len(points_cave_vacuum_all)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {points_cave_vacuum_all[i][0]} {points_cave_vacuum_all[i][1]} {points_cave_vacuum_all[i][2]} {points_cave_vacuum_all[i][0]+dx} {points_cave_vacuum_all[i][1]+dy} {points_cave_vacuum_all[i][2]+dz} free_space n\n")

    f.write(f"#rx: {x/10 + x_co} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} mare_tube_ring_B_025_0 n \n")

# Output success message
print("gprMax input file 'mare_tube_ring_B_025_0.in' generated.")

with open('mare_tube_vacuum_B_0_0.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 13 17.8 2.2 3.2 13.5 15.5 7.5 12.5 40.5 47.5 8.5 12.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 19 22 1 6 6 10.5 6 11.5 42 47 9.5 10.5 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} free_space n\n")

    f.write(f"#rx: {x/10} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} mare_tube_vacuum_B_0_0 n \n")

# Output success message
print("gprMax input file 'mare_tube_vacuum_B_0_0.in' generated.")

with open('mare_tube_vacuum_B_0_01.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon + y_gpr} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 13 17.8 2.2 3.2 13.5 15.5 7.5 12.5 40.5 47.5 8.5 12.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 19 22 1 6 6 10.5 6 11.5 42 47 9.5 10.5 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} free_space n\n")

    f.write(f"#rx: {x/10} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} mare_tube_vacuum_B_0_01 n \n")

# Output success message
print("gprMax input file 'mare_tube_vacuum_B_0_01.in' generated.")

with open('mare_tube_vacuum_B_025_01.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon + y_gpr} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 13 17.8 2.2 3.2 13.5 15.5 7.5 12.5 40.5 47.5 8.5 12.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 19 22 1 6 6 10.5 6 11.5 42 47 9.5 10.5 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} free_space n\n")

    f.write(f"#rx: {x/2 + x_co} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} mare_tube_vacuum_B_025_01 n \n")

# Output success message
print("gprMax input file 'mare_tube_vacuum_B_025_01.in' generated.")

with open('mare_tube_vacuum_B_025_0.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 13 17.8 2.2 3.2 13.5 15.5 7.5 12.5 40.5 47.5 8.5 12.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 19 22 1 6 6 10.5 6 11.5 42 47 9.5 10.5 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} free_space n\n")

    f.write(f"#rx: {x/10 + x_co} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} mare_tube_vacuum_B_025_0 n \n")

# Output success message
print("gprMax input file 'mare_tube_vacuum_B_025_0.in' generated.")

with open('hland_tube_ring.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/2 - x_co} {y_moon} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 1 10 0.1 3 24 34 1 9.5 41.5 48.5 11 21.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 5 10.01 0.4 1.59 17.9 28 6.9 12.5 44.4 48.4 11.1 15 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    eps_r_value = 7.6
    sigma_value = 1e-10
    mu_r_value = 1.0
    sigma_m_value = 0.0
    f.write(f"#material: {eps_r_value} {sigma_value} {mu_r_value} {sigma_m_value} moon_lava_tube \n")
    # Dielectric smoothing needs to be turned off, add higher discretization for numerical stability

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} moon_lava_tube n\n")

    for i in range(len(points_cave_vacuum_all)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {points_cave_vacuum_all[i][0]} {points_cave_vacuum_all[i][1]} {points_cave_vacuum_all[i][2]} {points_cave_vacuum_all[i][0]+dx} {points_cave_vacuum_all[i][1]+dy} {points_cave_vacuum_all[i][2]+dz} free_space n\n")

    f.write(f"#rx: {x/2 + x_co} {y_moon} {z/2}\n")
    f.write(f"#geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} hland_tube_ring n \n")

# Output success message
print("gprMax input file 'hland_tube_ring.in' generated.")

with open('hland_tube_vacuum.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/2 - x_co} {y_moon} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 1 10 0.1 3 24 34 1 9.5 41.5 48.5 11 21.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 5 10.01 0.4 1.59 17.9 28 6.9 12.5 44.4 48.4 11.1 15 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} free_space n\n")

    f.write(f"#rx: {x/2 + x_co} {y_moon} {z/2}\n")
    f.write(f"#geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} hland_tube_vacuum n \n")

# Output success message
print("gprMax input file 'hland_tube_vacuum.in' generated.")

with open('hland_tube_ring_B_0_0.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 1 10 0.1 3 24 34 1 9.5 41.5 48.5 11 21.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 5 10.01 0.4 1.59 17.9 28 6.9 12.5 44.4 48.4 11.1 15 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    eps_r_value = 7.6
    sigma_value = 1e-10
    mu_r_value = 1.0
    sigma_m_value = 0.0
    f.write(f"#material: {eps_r_value} {sigma_value} {mu_r_value} {sigma_m_value} moon_lava_tube \n")
    # Dielectric smoothing needs to be turned off, add higher discretization for numerical stability

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} moon_lava_tube n\n")

    for i in range(len(points_cave_vacuum_all)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {points_cave_vacuum_all[i][0]} {points_cave_vacuum_all[i][1]} {points_cave_vacuum_all[i][2]} {points_cave_vacuum_all[i][0]+dx} {points_cave_vacuum_all[i][1]+dy} {points_cave_vacuum_all[i][2]+dz} free_space n\n")

    f.write(f"#rx: {x/10} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} hland_tube_ring_B_0_0 n \n")

# Output success message
print("gprMax input file 'hland_tube_ring_B_0_0.in' generated.")


with open('hland_tube_ring_B_0_01.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon + y_gpr} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 1 10 0.1 3 24 34 1 9.5 41.5 48.5 11 21.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 5 10.01 0.4 1.59 17.9 28 6.9 12.5 44.4 48.4 11.1 15 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    f.write(f"#material: {eps_r_value} {sigma_value} {mu_r_value} {sigma_m_value} moon_lava_tube \n")
    # Dielectric smoothing needs to be turned off, add higher discretization for numerical stability

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} moon_lava_tube n\n")

    for i in range(len(points_cave_vacuum_all)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {points_cave_vacuum_all[i][0]} {points_cave_vacuum_all[i][1]} {points_cave_vacuum_all[i][2]} {points_cave_vacuum_all[i][0]+dx} {points_cave_vacuum_all[i][1]+dy} {points_cave_vacuum_all[i][2]+dz} free_space n\n")

    f.write(f"#rx: {x/10} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} hland_tube_ring_B_0_01 n \n")

# Output success message
print("gprMax input file 'hland_tube_ring_B_0_01.in' generated.")

with open('hland_tube_ring_B_025_01.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon + y_gpr} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 1 10 0.1 3 24 34 1 9.5 41.5 48.5 11 21.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 5 10.01 0.4 1.59 17.9 28 6.9 12.5 44.4 48.4 11.1 15 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    eps_r_value = 7.6
    sigma_value = 1e-10
    mu_r_value = 1.0
    sigma_m_value = 0.0
    f.write(f"#material: {eps_r_value} {sigma_value} {mu_r_value} {sigma_m_value} moon_lava_tube \n")
    # Dielectric smoothing needs to be turned off, add higher discretization for numerical stability

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} moon_lava_tube n\n")

    for i in range(len(points_cave_vacuum_all)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {points_cave_vacuum_all[i][0]} {points_cave_vacuum_all[i][1]} {points_cave_vacuum_all[i][2]} {points_cave_vacuum_all[i][0]+dx} {points_cave_vacuum_all[i][1]+dy} {points_cave_vacuum_all[i][2]+dz} free_space n\n")

    f.write(f"#rx: {x/2 + x_co} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} hland_tube_ring_B_025_01 n \n")

# Output success message
print("gprMax input file 'hland_tube_ring_B_025_01.in' generated.")

with open('hland_tube_ring_B_025_0.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 1 10 0.1 3 24 34 1 9.5 41.5 48.5 11 21.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 5 10.01 0.4 1.59 17.9 28 6.9 12.5 44.4 48.4 11.1 15 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    eps_r_value = 7.6
    sigma_value = 1e-10
    mu_r_value = 1.0
    sigma_m_value = 0.0
    f.write(f"#material: {eps_r_value} {sigma_value} {mu_r_value} {sigma_m_value} moon_lava_tube \n")
    # Dielectric smoothing needs to be turned off, add higher discretization for numerical stability

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} moon_lava_tube n\n")

    for i in range(len(points_cave_vacuum_all)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {points_cave_vacuum_all[i][0]} {points_cave_vacuum_all[i][1]} {points_cave_vacuum_all[i][2]} {points_cave_vacuum_all[i][0]+dx} {points_cave_vacuum_all[i][1]+dy} {points_cave_vacuum_all[i][2]+dz} free_space n\n")

    f.write(f"#rx: {x/10 + x_co} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} hland_tube_ring_B_025_0 n \n")

# Output success message
print("gprMax input file 'hland_tube_ring_B_025_0.in' generated.")

with open('hland_tube_vacuum_B_0_0.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 1 10 0.1 3 24 34 1 9.5 41.5 48.5 11 21.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 5 10.01 0.4 1.59 17.9 28 6.9 12.5 44.4 48.4 11.1 15 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} free_space n\n")

    f.write(f"#rx: {x/10} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} hland_tube_vacuum_B_0_0 n \n")

# Output success message
print("gprMax input file 'hland_tube_vacuum_B_0_0.in' generated.")

with open('hland_tube_vacuum_B_0_01.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon + y_gpr} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 1 10 0.1 3 24 34 1 9.5 41.5 48.5 11 21.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 5 10.01 0.4 1.59 17.9 28 6.9 12.5 44.4 48.4 11.1 15 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} free_space n\n")

    f.write(f"#rx: {x/10} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} hland_tube_vacuum_B_0_01 n \n")

# Output success message
print("gprMax input file 'hland_tube_vacuum_B_0_01.in' generated.")

with open('hland_tube_vacuum_B_025_01.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon + y_gpr} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 1 10 0.1 3 24 34 1 9.5 41.5 48.5 11 21.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 5 10.01 0.4 1.59 17.9 28 6.9 12.5 44.4 48.4 11.1 15 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} free_space n\n")

    f.write(f"#rx: {x/2 + x_co} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} hland_tube_vacuum_B_025_01 n \n")

# Output success message
print("gprMax input file 'hland_tube_vacuum_B_025_01.in' generated.")

with open('hland_tube_vacuum_B_025_0.in', 'w') as f:

    # Write the domain and discretization
    f.write(f"#domain: {x} {y} {z}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")
    f.write(f"#time_window: {9e-7}\n")

    # Define the GPR source
    f.write(f"#waveform: ricker 1 {f_max/2} ricker_wavelet\n")
    f.write(f"#hertzian_dipole: z {x/10} {y_moon} {z/2} ricker_wavelet\n")
    
    f.write(f"#soil_gen_bruggeman_moon: 1 10 0.1 3 24 34 1 9.5 41.5 48.5 11 21.5 soil_reg\n")
    f.write(f"#fractal_box: 0 {y_crust} 0 {x} {y_moon} {z} 2.5 0.75 0.75 1.5 25 soil_reg moon_regolith\n")

    f.write(f"#soil_gen_bruggeman_moon: 5 10.01 0.4 1.59 17.9 28 6.9 12.5 44.4 48.4 11.1 15 rocks_crust\n")
    f.write(f"#fractal_box: 0 0 0 {x} {y_crust} {z} 2.5 0.75 0.75 1.5 25 rocks_crust moon_crust\n")

    # Loop over each cell to define the materials
    for i in range(len(x_mask)):
        # Get the dielectric properties for the current cell
        f.write(f"#box: {x_mask[i]} {y_mask[i]} {z_mask[i]} {x_mask[i]+dx} {y_mask[i]+dy} {z_mask[i]+dz} free_space n\n")

    f.write(f"#rx: {x/10 + x_co} {y_moon} {z/2}\n")
    f.write(f"#src_steps: {x_step} 0 0\n#rx_steps: {x_step} 0 0\n")
    f.write(f"geometry_view: 0 0 0 {x} {y} {z} {dx} {dy} {dz} hland_tube_vacuum_B_025_0 n \n")

# Output success message
print("gprMax input file 'hland_tube_vacuum_B_025_0.in' generated.")