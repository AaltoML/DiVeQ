import numpy as np
from cleanfid import fid
import argparse
import os

os.makedirs("./fid/", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('dir1', type=str)
parser.add_argument('dir2', type=str)
parser.add_argument('fid_path', type=str)
args = parser.parse_args()

fid_array = np.zeros((1,))

score = fid.compute_fid(args.dir1, args.dir2)
fid_array[0] = score

print(score)
np.save(f'./fid/fid_{args.fid_path}', fid_array)