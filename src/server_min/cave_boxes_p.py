import open3d as o3d
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
import csv
import pandas as pd

def poisson_surface_reconstruction(point_cloud, depth=9):
    """Perform Poisson Surface Reconstruction."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    
    # Perform Poisson Surface Reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth)
    
    return mesh

def voxel_grid_from_mesh(mesh, voxel_size):
    """Create a voxel grid from a mesh."""
    vertices = np.asarray(mesh.vertices)
    
    # Compute the bounding box of the mesh
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    
    # Compute grid dimensions
    nx = int(np.ceil((max_bound[0] - min_bound[0]) / voxel_size))
    ny = int(np.ceil((max_bound[1] - min_bound[1]) / voxel_size))
    nz = int(np.ceil((max_bound[2] - min_bound[2]) / voxel_size))
    
    # Calculate voxel indices
    ix = np.floor((vertices[:, 0] - min_bound[0]) / voxel_size).astype(int)
    iy = np.floor((vertices[:, 1] - min_bound[1]) / voxel_size).astype(int)
    iz = np.floor((vertices[:, 2] - min_bound[2]) / voxel_size).astype(int)
    
    # Ensure indices are within bounds
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    iz = np.clip(iz, 0, nz - 1)
    
    # Debugging output to verify indices
    print(f"Indices: ix={ix}, iy={iy}, iz={iz}")
    
        # Initialize the voxel grid
    voxel_grid = np.zeros((nx, ny, nz), dtype=bool)
    
    # Calculate voxel indices
    ix = np.floor((vertices[:, 0] - min_bound[0]) / voxel_size).astype(int)
    iy = np.floor((vertices[:, 1] - min_bound[1]) / voxel_size).astype(int)
    iz = np.floor((vertices[:, 2] - min_bound[2]) / voxel_size).astype(int)
    
    # Ensure indices are within bounds
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    iz = np.clip(iz, 0, nz - 1)
    
    # Update the voxel grid
    voxel_grid[ix, iy, iz] = True
    
    return voxel_grid, min_bound, voxel_size, nx, ny, nz

def find_box_end(x_start, y_start, z_start, voxel_grid, covered, min_size, dx, dy, dz):
    """Find the end of the box given a starting point."""
    x_end, y_end, z_end = x_start, y_start, z_start
    
    while (x_end + 1 < voxel_grid.shape[0] and
           voxel_grid[x_end + 1, y_start:y_end + 1, z_start:z_end + 1].all() and
           not covered[x_end + 1, y_start:y_end + 1, z_start:z_end + 1].any()):
        x_end += 1
        if (x_end - x_start + 1) * dx >= min_size:
            break
    
    while (y_end + 1 < voxel_grid.shape[1] and
           voxel_grid[x_start:x_end + 1, y_end + 1, z_start:z_end + 1].all() and
           not covered[x_start:x_end + 1, y_end + 1, z_start:z_end + 1].any()):
        y_end += 1
        if (y_end - y_start + 1) * dy >= min_size:
            break
    
    while (z_end + 1 < voxel_grid.shape[2] and
           voxel_grid[x_start:x_end + 1, y_start:y_end + 1, z_end + 1].all() and
           not covered[x_start:x_end + 1, y_start:y_end + 1, z_end + 1].any()):
        z_end += 1
        if (z_end - z_start + 1) * dz >= min_size:
            break
    
    return x_end, y_end, z_end

def fit_boxes(voxel_grid, min_size, dx, dy, dz):
    """Fit boxes to the voxel grid using grid indices."""
    covered = np.zeros_like(voxel_grid, dtype=int)
    boxes = []
    
    for x in range(voxel_grid.shape[0]):
        for y in range(voxel_grid.shape[1]):
            for z in range(voxel_grid.shape[2]):
                if voxel_grid[x, y, z] and not covered[x, y, z]:
                    x_end, y_end, z_end = find_box_end(x, y, z, voxel_grid, covered, min_size, dx, dy, dz)
                    
                    if (x_end - x + 1) * dx >= min_size and \
                       (y_end - y + 1) * dy >= min_size and \
                       (z_end - z + 1) * dz >= min_size:
                        boxes.append(((x, y, z), (x_end, y_end, z_end)))
                    
                    covered[x:x_end+1, y:y_end+1, z:z_end+1] = 1
    
    return boxes

def export_boxes_to_csv(boxes, x_min, y_min, z_min, dx, dy, dz, output_file):
    """Export the bounding boxes to a CSV file."""
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max'])
        for (start, end) in boxes:
            start_x = min(start[0], end[0]) * dx + x_min
            start_y = min(start[1], end[1]) * dy + y_min
            start_z = min(start[2], end[2]) * dz + z_min
            end_x = max(start[0], end[0]) * dx + x_min
            end_y = max(start[1], end[1]) * dy + y_min
            end_z = max(start[2], end[2]) * dz + z_min
            writer.writerow([start_x, start_y, start_z, end_x, end_y, end_z])

def write_vtk_file(boxes_real, scale):
    with open(f'boxes_{int(1/scale)}.vtk', 'w') as vtk_file:
        # VTK header
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("Box Data\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET UNSTRUCTURED_GRID\n")

        # Calculate the total number of points (8 points per box)
        num_points = len(boxes_real) * 8
        vtk_file.write(f"POINTS {num_points} float\n")

        # Write points (vertices of each box)
        points = []
        for box in boxes_real:
            start, end = box
            x0, y0, z0 = start
            x1, y1, z1 = end
            
            # Define the 8 vertices of the box
            vertices = [
                (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)
            ]
            points.extend(vertices)
        
        for point in points:
            vtk_file.write(f"{point[0]} {point[1]} {point[2]}\n")

        # Define cells (hexahedrons) and their connectivity
        num_boxes = len(boxes_real)
        vtk_file.write(f"CELLS {num_boxes} {num_boxes * 9}\n")
        for i in range(num_boxes):
            vtk_file.write(f"8 {i*8} {i*8+1} {i*8+2} {i*8+3} {i*8+4} {i*8+5} {i*8+6} {i*8+7}\n")

        # Define cell types (12 = VTK_HEXAHEDRON)
        vtk_file.write(f"CELL_TYPES {num_boxes}\n")
        for _ in range(num_boxes):
            vtk_file.write("12\n")

def read_boxes_from_csv(csv_file):
    """Read bounding boxes from a CSV file."""
    boxes = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            x_min, y_min, z_min, x_max, y_max, z_max = map(float, row)
            boxes.append(((x_min, y_min, z_min), (x_max, y_max, z_max)))
    return boxes

# Main execution
if __name__ == "__main__":
    scale = 0.125

    # Domain
    x = 150*scale
    z = 74*scale
    y_air = 5*scale
    y_moon = 50*scale
    y = y_moon + y_air

    if scale >= 0.5:
        y_cave = 10
    else:
        y_cave = 10*scale
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
    
    print(1)
    # Perform Poisson Surface Reconstruction
    mesh = poisson_surface_reconstruction(point_cloud, depth=9)
    print(2)
    # Generate voxel grid and bounding boxes
    voxel_grid, min_bound, voxel_size, nx, ny, nz = voxel_grid_from_mesh(mesh, dx)
    min_box_size = 2 * min(dx, dy, dz)
    print(3)
    boxes = fit_boxes(voxel_grid, min_box_size, dx, dy, dz)
    print(f"Found {len(boxes)} boxes")

    # Export bounding boxes to CSV file
    output_file = f"boxes_{int(1/scale)}.csv"
    export_boxes_to_csv(boxes, min_bound[0], min_bound[1], min_bound[2], dx, dy, dz, output_file)
    
    print(f"Bounding boxes have been exported to {output_file}")
    boxes = read_boxes_from_csv(output_file)
    
    # Write bounding boxes to VTK file
    write_vtk_file(boxes, scale)

    print(f"Boxes have been exported to 'boxes_{int(1/scale)}.vtk'")