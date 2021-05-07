# diffUNet
U-Nets for heat equation mappings

### Installation
`pip install requirements.txt` for necessary packages

### Data


#### Pre-generated data

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

#### Generating new data:

Amrex data generation is done with [Reva's fork of amrex](https://github.com/RevathiJambunathan/amrex/tree/add_Parser_HeatEq1_C_HDF5), which allows parsing numbers from input file for initiliazing random Gaussians. For compiling on Cori, clone the repo and navigate to `amrex/Tutorials/Basic/HeatEquation_EX1_C_HDF5/Exec`. For 2D runs (as everything done so far), make sure to set `DIM = 2` in the `GNUMakefile`. Then, do
```
1. module switch PrgEnv-intel/6.0.5 PrgEnv-gnu
2. module load cray-hdf5-parallel
make -j 16
```

Once you have the executable, copy into this repo's `datagen/Exec/` directory and give it a useful name there. All code for generating and managing amrex data lives in theis repo's `datagen` subdir. The actual runs to generate data are done via slurm job arrays on Cori Haswell nodes -- currently configured for each job to take a node and run 150 unique runs, handled by `batchRun_haswell.slr` slurm script. Generating, e.g., 10k runs is as simple as 
```
sbatch --array=0-70 batchrun_haswell.slr
```
This will launch an array of 71 jobs (71*150 = 10650 runs, some will fail/time out and you will end up with ~10k). Make sure to adjust `batchrun_haswell.slr` to use the proper executable and initial condition script.

To generate random initial conditions in the input file, see for example the `run_rand_init.sh` script (which is called from `batchrun_haswell.slr`). This generates some random numbers for the amplitude, location, and width of the initial gaussian, and writes them to the input files before running. The key idea is to run the same initial conditions twice -- one run normal, and one with a 2x lower diffusion coefficient. Thus, this script will then copy out the executable to two subdirs `Dc1` and `Dc2` to do these runs with different diffusion coeffs, before calling `mergeh5.py` to take the hdf5 outputs for each timestep and aggregate them in one file. 

After running a batch of jobs, you can use `bigh5.py` to aggregate all data into one big hdf5 file for use in ML training. Simply edit that script to adjust file paths, give the runs an appropriate tag name, and set `Nperjob` (number of runs per job in your array, e.g. 150 above) and `Njobs` (number of jobs in the job array).

### Running
To train: `python train.py --yaml_config=./config/UNet.yaml --config def_sched_relu --run_num=run_number`

To evaluate on OOD datasets: `python eval.py --yaml_config=./config/UNet.yaml --config def_sched_relu --weights=/path/to/trained/weights`


### Notes
Distributed training not tested

Single GPU traintime is ~2.5 hrs on Nvidia 2080Ti

