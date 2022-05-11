# !pip3 install pyvista

import numpy as np
import pyvista as pv
import sys

path = sys.argv[1]
points = (np.load( path))
point_cloud = pv.PolyData(points)

# np.allclose(points, point_cloud.points)
point_cloud.plot()
