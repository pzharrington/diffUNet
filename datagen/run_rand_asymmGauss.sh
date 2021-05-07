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
uwx=$rndF
myRandomF
uwy=$rndF
myRandomF
utht=$rndF

amp=` echo $uamp | awk '{printf "%f", (0.1 + 0.9*$1)}'` # 0.1 - 1. 
xc=` echo $uxc | awk '{printf "%f", (2.*$1 - 1.)}'`
yc=` echo $uyc | awk '{printf "%f", (2.*$1 - 1.)}'`
wx=` echo $uwx | awk '{printf "%f", ((2./128)*(20. + 1.*$1))}'` 
wy=` echo $uwy | awk '{printf "%f", ((2./128)*(9. + 1.*$1))}'` 
tht=` echo $utht | awk '{printf "%f", (6.28318*$1)}'` # 0-2pi

a=` echo $tht $wx $wy | awk '{printf "%f", (cos($1)*cos($1))/(2.*$2*$2) + (sin($1)*sin($1))/(2.*$3*$3) }'`
b=` echo $tht $wx $wy | awk '{printf "%f", -(sin(2.*$1))/(4.*$2*$2) + (sin(2.*$1))/(4.*$3*$3) }'`
c=` echo $tht $wx $wy | awk '{printf "%f", (sin($1)*sin($1))/(2.*$2*$2) + (cos($1)*cos($1))/(2.*$3*$3) }'`


sed -i "s/my_constants.amp .*/my_constants.amp = $amp/" $configFile
sed -i "s/my_constants.xc .*/my_constants.xc = $xc/" $configFile
sed -i "s/my_constants.yc .*/my_constants.yc = $yc/" $configFile
sed -i "s/my_constants.a .*/my_constants.a = $a/" $configFile
sed -i "s/my_constants.b .*/my_constants.b = $b/" $configFile
sed -i "s/my_constants.c .*/my_constants.c = $c/" $configFile
sed -i "s/diffusionCoefficient .*/diffusionCoefficient = 1.0/" $configFile

# 2nd run with lower diffusion coeff
configFile2=${configFile}_Dc2
cp $configFile $configFile2
sed -i "s/diffusionCoefficient .*/diffusionCoefficient = 0.5/" $configFile2

mkdir Dc1
mkdir Dc2
cp $configFile Dc1
cp $configFile2 Dc2/$configFile

# and run
cd Dc1
srun -n 2 ../main2d.gnu.haswell.MPI.ex ${configFile} amrex.v=0
cd ../Dc2
srun -n 2 ../main2d.gnu.haswell.MPI.ex ${configFile} amrex.v=0
cd ..

# Merge files and clean up
module load python
python ../mergeh5.py $iternum $configFile
cp ./aggdat*.h5 ../
#rm -r ./Dc*

