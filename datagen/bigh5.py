import h5py
import numpy as np
import os


def checkfname(sDir, ji, ri):
  f1 = sDir+str(ji)+'/aggdat_'+str(ri)+'_Dc1.h5'
  f2 = sDir+str(ji)+'/aggdat_'+str(ri)+'_Dc2.h5'
  return os.path.isfile(f1) and os.path.isfile(f2)


Nperjob=150
Njobs=17

scratch = '/global/cscratch1/sd/pharring/diffUNet/'
cfs = '/global/cfs/cdirs/m3363/pharring/diffUNet/data/' 
tag='baseOOD_asymm'
scratchDir=scratch+tag+'-'

#keys = ['phi', 'amp', 'xc', 'yc', 'width', 'diffusionCoefficient'] #base
keys = ['phi', 'amp', 'xc', 'yc', 'a', 'b', 'c', 'diffusionCoefficient'] #base_asymm

out1 = {k:[] for k in keys}
out2 = {k:[] for k in keys}

outs = [out1, out2]

bad=0
for jobIdx in range(Njobs):
  
  for runIdx in range(Nperjob):
    if not checkfname(scratchDir,jobIdx,runIdx):
      bad+=1
      print('%d: Missing %s'%(bad,scratchDir+str(jobIdx)+'/aggdat_'+str(runIdx)))
      continue

    for i, Dc in enumerate(['Dc1','Dc2']):
      fname = scratchDir+str(jobIdx)+'/aggdat_'+str(runIdx)+'_'+Dc+'.h5'
      with h5py.File(fname, 'r') as f:
        for k in keys:
          outs[i][k].append(f[k][...])


for i, Dc in enumerate(['Dc1','Dc2']):
  outf = cfs+tag+'_aggdata_'+Dc+'.h5'

  with h5py.File(outf, 'a') as f:
    dat = outs[i]
    for k, v in dat.items():
      f.create_dataset(k, data=np.stack(v, axis=0))
      print(k, f[k].shape, f[k].dtype)

print('DONE')
