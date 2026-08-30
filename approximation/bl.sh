# Boundary-layer benchmark on the local RTX 3070 (device 0).

python approximation_1d.py --epochs 5000 --network mlp --datatype bl --npoints 5000 --ntrain 3000 --ntest 10000 --seed 0 --device 0
python approximation_1d.py --epochs 5000 --network mlp --datatype bl --npoints 5000 --ntrain 3000 --ntest 10000 --seed 1 --device 0
python approximation_1d.py --epochs 5000 --network mlp --datatype bl --npoints 5000 --ntrain 3000 --ntest 10000 --seed 2 --device 0

python approximation_1d.py --epochs 5000 --network modifiedmlp --datatype bl --features 100 --layers 10 --npoints 5000 --ntrain 3000 --ntest 10000 --seed 0 --device 0
python approximation_1d.py --epochs 5000 --network modifiedmlp --datatype bl --features 100 --layers 10 --npoints 5000 --ntrain 3000 --ntest 10000 --seed 1 --device 0
python approximation_1d.py --epochs 5000 --network modifiedmlp --datatype bl --features 100 --layers 10 --npoints 5000 --ntrain 3000 --ntest 10000 --seed 2 --device 0

python approximation_1d.py --epochs 5000 --network kan --kanshape 63 --degree 63 --datatype bl --npoints 5000 --ntrain 3000 --ntest 10000 --seed 0 --device 0
python approximation_1d.py --epochs 5000 --network kan --kanshape 120 --degree 120 --datatype bl --npoints 5000 --ntrain 3000 --ntest 10000 --seed 1 --device 0
python approximation_1d.py --epochs 5000 --network kan --kanshape 42,42 --degree 42 --datatype bl --npoints 5000 --ntrain 3000 --ntest 10000 --seed 2 --device 0

python approximation_1d.py --epochs 5000 --network sinckan --kanshape 16 --degree 100 --normalization 1 --datatype bl --len_h 6 --npoints 5000 --ntrain 3000 --ntest 10000 --seed 0 --device 0
python approximation_1d.py --epochs 5000 --network sinckan --kanshape 16 --degree 100 --normalization 1 --datatype bl --len_h 6 --npoints 5000 --ntrain 3000 --ntest 10000 --seed 1 --device 0
python approximation_1d.py --epochs 5000 --network sinckan --kanshape 16 --degree 100 --normalization 1 --datatype bl --len_h 6 --npoints 5000 --ntrain 3000 --ntest 10000 --seed 2 --device 0

python approximation_1d.py --epochs 5000 --network modifiedsinckan --datatype bl --features 64 --layers 5 --degree 64 --len_h 4 --init_h 2 --decay inverse --normalization 1 --activation tanh --uv_activation none --gate_mode separate_residual --skip True --npoints 5000 --ntrain 3000 --ntest 10000 --seed 0 --device 0
python approximation_1d.py --epochs 5000 --network modifiedsinckan --datatype bl --features 64 --layers 5 --degree 64 --len_h 4 --init_h 2 --decay inverse --normalization 1 --activation tanh --uv_activation none --gate_mode separate_residual --skip True --npoints 5000 --ntrain 3000 --ntest 10000 --seed 1 --device 0
python approximation_1d.py --epochs 5000 --network modifiedsinckan --datatype bl --features 64 --layers 5 --degree 64 --len_h 4 --init_h 2 --decay inverse --normalization 1 --activation tanh --uv_activation none --gate_mode separate_residual --skip True --npoints 5000 --ntrain 3000 --ntest 10000 --seed 2 --device 0
