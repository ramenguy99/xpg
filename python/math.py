from pyxpg import math

# y = math.ivec3(10)
z = math.ivec2(5, 6)
# w = math.ivec3(7, z)
# v = math.ivec2(w)

a = math.vec3(1, 3, 10)
b = math.vec3(2, 5, 7)

v = math.dot(a, b)

print(f"{a} o {b} = {v}")
