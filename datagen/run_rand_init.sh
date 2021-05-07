#!/bin/bash 
 
set -u ;  # exit  if you try to use an uninitialized variable
set -e ;    #  bash exits if any statement returns a non-true return value
set -o errexit ;  # exit if any statement returns a non-true return value



myRandomF () {
    rndI=`od -A n -t d -N 3 /dev/random | tr -d ' '`
    # This is 3-byte truly random generator, max value: 256^3=16777216-1
    # rndF is [0,1]
    rndF=` echo $rndI | awk  '{printf "%.7f", $1/ 16777216 }'`
}


configFile=$1
iternum=$2

# set up directory for this run
outPath=out_${iternum}
mkdir $outPath
cp ./main2d.gnu.haswell.MPI.ex $configFile $outPath
cd $outPath

# set random amp, xc, yz, width
myRandomF
uamp=$rndF
myRandomF
uxc=$rndF
myRandomF
uyc=$rndF
myRandomF
uwd=$rndF

amp=` echo $uamp | awk '{printf "%f", (0.1 + 0.9*$1)}'` # 0.1 - 1. 
xc=` echo $uxc | awk '{printf "%f", (2.*$1 - 1.)}'`
yc=` echo $uyc | awk '{printf "%f", (2.*$1 - 1.)}'`
wd=` echo $uwd | awk '{printf "%f", ((2./128)*(2. + 8.*$1))}'` # 2-10 *dx

sed -i "s/my_constants.amp .*/my_constants.amp = $amp/" $configFile
sed -i "s/my_constants.xc .*/my_constants.xc = $xc/" $configFile
sed -i "s/my_constants.yc .*/my_constants.yc = $yc/" $configFile
sed -i "s/my_constants.width .*/my_constants.width = $wd/" $configFile
sed -i "s/diffusionCoefficient .*/diffusionCoefficient = 1.0/" $configFile

# 2nd run with lower diffusion coeff
configFile2=${configFile}_Dc2
cp $configFile $configFile2
sed -i "s/diffusionCoefficient .*/diffusionCoefficient = 0.5/" $configFile2

mkdir Dc1
mkdir Dc2
cp $configFile Dc1
cp $configFile2 Dc2/inputs

# and run
cd Dc1
srun -n 2 ../main2d.gnu.haswell.MPI.ex ${configFile} amrex.v=0
cd ../Dc2
srun -n 2 ../main2d.gnu.haswell.MPI.ex ${configFile} amrex.v=0
cd ..

# Merge files and clean up
module load python
python ../mergeh5.py $iternum
cp ./aggdat*.h5 ../
#rm -r ./Dc*

