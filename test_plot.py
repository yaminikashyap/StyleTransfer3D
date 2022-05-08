# !pip3 install pyvista

import numpy as np
import pyvista as pv
points = (np.load("jui.npy")[0])
point_cloud = pv.PolyData(points)

# np.allclose(points, point_cloud.points)
point_cloud.plot()
