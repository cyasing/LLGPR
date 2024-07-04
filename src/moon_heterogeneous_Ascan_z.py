import numpy as np

# Example dimensions of the model (in cells)
nx, ny, nz = 100, 100, 50

# Generate a sample porosity model (for demonstration purposes, using random values)
porosity_model = np.random.rand(nx, ny, nz)

# Define empirical relationships (constants)
epsilon_solid = 5.0
epsilon_water = 80.0
sigma_solid = 0.01
sigma_water = 0.5

# Translate porosity to dielectric properties
epsilon_r = (1 - porosity_model) * epsilon_solid + porosity_model * epsilon_water
sigma = (1 - porosity_model) * sigma_solid + porosity_model * sigma_water

# Define the gprMax domain and discretization
dx, dy, dz = 0.01, 0.01, 0.01
x_length, y_length, z_length = nx * dx, ny * dy, nz * dz

# Open the gprMax input file for writing
with open('moon_heterogeneous_Ascan_z.in', 'w') as f:
    # Write the domain and discretization
    f.write(f"#domain: {x_length} {y_length} {z_length}\n")
    f.write(f"#dx_dy_dz: {dx} {dy} {dz}\n")

    # Loop over each cell to define the materials
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Get the dielectric properties for the current cell
                eps_r_value = epsilon_r[i, j, k]
                sigma_value = sigma[i, j, k]

                # Define the material and the cell coordinates
                f.write(f"#material: {i*ny*nz + j*nz + k + 1} 0 {eps_r_value} {sigma_value}\n")
                f.write(f"#box: {i*dx} {(i+1)*dx} {j*dy} {(j+1)*dy} {k*dz} {(k+1)*dz}\n")

    # Define the GPR source
    f.write("#waveform: ricker 1.0e9\n")
    f.write("#hertzian_dipole: z 0.1 0.1 0.0\n")
    
    # Define the receivers
    receiver_positions = [[0.2, 0.1, 0.0], [0.3, 0.1, 0.0], [0.4, 0.1, 0.0]]
    for rx in receiver_positions:
        f.write(f"#rx: {rx[0]} {rx[1]} {rx[2]}\n")
    
    # Set the simulation time window
    f.write("#time_window: 2.0e-7\n")

# Output success message
print("GprMax input file 'moon_heterogeneous_Ascan_z.in' generated successfully.")
