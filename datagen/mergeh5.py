import h5py
import numpy as np
import os
import sys

def merge_files(Dc, iternum, inf):
  ls = []
  for i in range(11):
    fname = os.path.join(Dc,'plt%05d.h5'%(i*100))
    with h5py.File(fname, 'r') as f:
      dset = f['level_0']['data:datatype=0']
      d = np.reshape(dset, (128, 128, 1), order='F')
      ls.append(d)
  phi = np.concatenate(ls, axis=-1)

  # Get params
  params = {}
  with open(os.path.join(Dc, inf)) as f:
    lines = f.readlines()
    for line in lines:
      if 'my_constants' in line or 'diffusionCoefficient' in line:
        k,v = line.split(" = ")
        params[k.replace('my_constants.', '')] = np.array(v).reshape(1,)

  outf = './aggdat_'+str(iternum)+'_'+Dc+'.h5'
  with h5py.File(outf, 'a') as f:
    f.create_dataset('phi', data=phi.astype(np.float32))
    for k,v in params.items():
      f.create_dataset(k, data=v.astype(np.float32))


idx = sys.argv[1]
inpfname = sys.argv[2]

for Dc in ['Dc1', 'Dc2']:
  merge_files(Dc, idx, inpfname)

