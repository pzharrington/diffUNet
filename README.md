# diffUNet
U-Nets for heat equation mappings

#### Installation
`pip install requirements.txt` for necessary packages

#### Data
Amrex heat eq runs w/ gaussian initial conditions hosted at https://portal.nersc.gov/project/m3363/diffUNet/data/

 File descriptions:
 ```
 *_Dc1.h5 is with diffusion coeff=1.0, *_Dc2.h5 is with diffusion coeff=0.5
 base_Dc_aggdata* is train+test dataset for base config
 baseOOD_Dc* is OOD "scale" test dataset
 baseOOD_asymm* is OOD "asymm" test dataset
```
Hdf5 file keys + shapes:

`N` == number of runs. For files `*_Dc1.h5` and `*_Dc2.h5`, index i along first dimension corresponds to unique initial conditions used for paired runs with different diffusion coeffs
```
amp: (N, 1) amplitude of gaussian
diffusionCoefficient: (N, 1)
phi: (N, 128, 128, 11) diffusing field (128x128, 11 time steps)
width: (N, 1) width of gaussian
xc: (N, 1) x-coord gaussian
yc: (N, 1) y-coord gaussian
```

#### Running
To train: `python train.py --yaml_config=./config/UNet.yaml --config def_sched_relu --run_num=run_number`

To evaluate on OOD datasets: `python eval.py --yaml_config=./config/UNet.yaml --config def_sched_relu --weights=/path/to/trained/weights`

#### Notes
Distributed training not tested

Single GPU traintime is ~2.5 hrs on Nvidia 2080Ti
