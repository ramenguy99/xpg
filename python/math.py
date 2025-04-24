from pyxpg import math

if False:
    # y = math.ivec3(10)
    z = math.ivec2(5, 6)
    # w = math.ivec3(7, z)
    # v = math.ivec2(w)

    a = math.vec3(1, 3, 10)
    b = math.vec3(2, 5, 7)

    v = math.dot(a, b)

    print(f"{a} o {b} = {v}")


import numpy as np

# arr = np.array(math.vec2(3, 2), dtype=np.int32)
# print(arr)

typ = np.dtype({
    'v3': ((np.int32, (3,)),  0),
    'v2': ((np.float32, (2,)), 16),
}, align=False)

# Can assign to all fields at once (basically interleave data into array)
N = 1

# Backed allocation
mem = bytearray(typ.itemsize * N)
a = np.frombuffer(mem, typ)
a["v3"] = np.array([1, 2, 3])
a["v2"] = math.vec2(3.5, 2.5)
print(a)