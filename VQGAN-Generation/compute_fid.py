import numpy as np
from cleanfid import fid
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir1', type=str, default=r"/l/datasets/afhq512/alaki2")
parser.add_argument('--dir2', type=str, default="./generations")
args = parser.parse_args()

fid_array = np.zeros((1,))

score = fid.compute_fid(args.dir1, args.dir2)
fid_array[0] = score

print(f"FID={score}")
np.save('FID', fid_array)