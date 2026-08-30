# Low-dimensional steady Poisson PINN benchmark on the local RTX 3070 (device 0).
# Use args.alpha / dim = 0.1, i.e. alpha = 0.1 * dim.

# dim = 2, alpha = 0.2
python steady_poisson.py --mode train --network mlp --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network mlp --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network mlp --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

python steady_poisson.py --mode train --network kan --datatype poisson --dim 2 --alpha 0.2 --kanshape 16,16,16 --degree 16 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network kan --datatype poisson --dim 2 --alpha 0.2 --kanshape 16,16,16 --degree 16 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network kan --datatype poisson --dim 2 --alpha 0.2 --kanshape 16,16,16 --degree 16 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

python steady_poisson.py --mode train --network sinckan --datatype poisson --dim 2 --alpha 0.2 --kanshape 16 --degree 16 --len_h 2 --init_h 2 --skip 1 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network sinckan --datatype poisson --dim 2 --alpha 0.2 --kanshape 16 --degree 16 --len_h 2 --init_h 2 --skip 1 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network sinckan --datatype poisson --dim 2 --alpha 0.2 --kanshape 16 --degree 16 --len_h 2 --init_h 2 --skip 1 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

python steady_poisson.py --mode train --network modifiedsinckan --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --degree 32 --len_h 3 --init_h 2 --skip 1 --gate_mode separate_residual --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network modifiedsinckan --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --degree 32 --len_h 3 --init_h 2 --skip 1 --gate_mode separate_residual --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network modifiedsinckan --datatype poisson --dim 2 --alpha 0.2 --features 128 --layers 6 --degree 32 --len_h 3 --init_h 2 --skip 1 --gate_mode separate_residual --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

# dim = 10, alpha = 1.0
python steady_poisson.py --mode train --network mlp --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network mlp --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network mlp --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network modifiedmlp --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

python steady_poisson.py --mode train --network kan --datatype poisson --dim 10 --alpha 1.0 --kanshape 16,16,16 --degree 16 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network kan --datatype poisson --dim 10 --alpha 1.0 --kanshape 16,16,16 --degree 16 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network kan --datatype poisson --dim 10 --alpha 1.0 --kanshape 16,16,16 --degree 16 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

python steady_poisson.py --mode train --network sinckan --datatype poisson --dim 10 --alpha 1.0 --kanshape 16 --degree 16 --len_h 2 --init_h 2 --skip 1 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network sinckan --datatype poisson --dim 10 --alpha 1.0 --kanshape 16 --degree 16 --len_h 2 --init_h 2 --skip 1 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network sinckan --datatype poisson --dim 10 --alpha 1.0 --kanshape 16 --degree 16 --len_h 2 --init_h 2 --skip 1 --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0

python steady_poisson.py --mode train --network modifiedsinckan --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --degree 32 --len_h 3 --init_h 2 --skip 1 --gate_mode separate_residual --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 0 --device 0
python steady_poisson.py --mode train --network modifiedsinckan --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --degree 32 --len_h 3 --init_h 2 --skip 1 --gate_mode separate_residual --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 1 --device 0
python steady_poisson.py --mode train --network modifiedsinckan --datatype poisson --dim 10 --alpha 1.0 --features 128 --layers 6 --degree 32 --len_h 3 --init_h 2 --skip 1 --gate_mode separate_residual --epochs 2000 --ite 20 --n_interior 2000 --n_boundary 100 --seed 2 --device 0
