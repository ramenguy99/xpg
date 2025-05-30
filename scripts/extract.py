from lz4 import block
import struct
import sys

# Run on bistro_compressed.lz4 generated by gray
data = open(sys.argv[1], "rb").read()
with open("res/bistro.bin", "wb") as out:
    it = memoryview(data)

    total = 0
    while len(it):
        c_size, u_size = struct.unpack("<II", it[:8])
        print(f"{u_size:x} {c_size:x}")
        decompressed = block.decompress(it[4:4+c_size])
        it = it[4+c_size:]
        total += len(decompressed)
        print(len(decompressed))

        out.write(decompressed)



