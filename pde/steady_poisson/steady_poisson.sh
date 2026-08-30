# Steady Poisson PINN benchmark on the local RTX 3070 (device 0).

python steady_poisson.py --mode train --network mlp --datatype poisson --dim 20 --alpha 10 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network mlp --datatype poisson --dim 20 --alpha 10 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network mlp --datatype poisson --dim 20 --alpha 10 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 20 --alpha 10 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 20 --alpha 10 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 20 --alpha 10 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

python steady_poisson.py --mode train --network kan --datatype poisson --dim 20 --alpha 10 --kanshape 8,8,8 --degree 8 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network kan --datatype poisson --dim 20 --alpha 10 --kanshape 8,8,8 --degree 8 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network kan --datatype poisson --dim 20 --alpha 10 --kanshape 8,8,8 --degree 8 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

python steady_poisson.py --mode train --network sinckan --datatype poisson --dim 20 --alpha 10 --kanshape 8 --degree 8 --len_h 2 --init_h 2 --skip 1 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network sinckan --datatype poisson --dim 20 --alpha 10 --kanshape 8 --degree 8 --len_h 2 --init_h 2 --skip 1 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network sinckan --datatype poisson --dim 20 --alpha 10 --kanshape 8 --degree 8 --len_h 2 --init_h 2 --skip 1 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

python steady_poisson.py --mode train --network modifiedsinckan --datatype poisson --dim 20 --alpha 10 --features 128 --layers 6 --degree 32 --len_h 3 --init_h 2 --skip 1 --gate_mode sigmoid --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network modifiedsinckan --datatype poisson --dim 20 --alpha 10 --features 128 --layers 6 --degree 32 --len_h 3 --init_h 2 --skip 1 --gate_mode sigmoid --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network modifiedsinckan --datatype poisson --dim 20 --alpha 10 --features 128 --layers 6 --degree 32 --len_h 3 --init_h 2 --skip 1 --gate_mode sigmoid --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0
