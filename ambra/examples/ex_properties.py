from ambra.property import DataBufferProperty

import numpy as np


vertices = np.zeros((32, 3), np.float32)

p0 = DataBufferProperty(vertices)
p1 = DataBufferProperty(vertices, np.float32)
p2 = DataBufferProperty(vertices, np.float32)
