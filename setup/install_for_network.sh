# create the anaconda environment
conda env create -f setup/rigidity.yml

# activate the environment
conda activate rigidity

# install the flow modules
cd external_packages/correlation-pytorch-master
sh make_cuda.sh
cd ..

