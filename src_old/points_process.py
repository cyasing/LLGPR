import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from Data_LiDAR folder
data_all = pd.read_csv('2020_Bell/Data_LiDAR/LiDAR_SkullCave_TUBE_CutDspc_1_skull_v1_UTM_Tube_1cm_clean_Cloud.txt', sep=' ', header=None)
data_points = data_all.iloc[:, 0:3]
# Convert to numpy arrays X, Y, Z
data_points = data_points.to_numpy()
X = data_points[:, 0]
Y = data_points[:, 1]
Z = data_points[:, 2]

# Save x, y, z data to a new file
np.savetxt('LiDAR_SkullCave_TUBE_points.txt', data_points, fmt='%f')

# Export to vtk file
from pyevtk.hl import pointsToVTK
pointsToVTK("LiDAR_SkullCave_TUBE", X, Y, Z)