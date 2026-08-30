# Fractal 2D benchmark on the local RTX 3070 (device 0).

python interpolation_hd.py --epochs 4000 --network mlp --datatype fractal --dim 2 --features 100 --layers 10 --ntrain 2000 --ntest 4000 --seed 0 --device 0
python interpolation_hd.py --epochs 4000 --network mlp --datatype fractal --dim 2 --features 100 --layers 10 --ntrain 2000 --ntest 4000 --seed 1 --device 0
python interpolation_hd.py --epochs 4000 --network mlp --datatype fractal --dim 2 --features 100 --layers 10 --ntrain 2000 --ntest 4000 --seed 2 --device 0

python interpolation_hd.py --epochs 4000 --network modifiedmlp --datatype fractal --dim 2 --features 100 --layers 10 --ntrain 2000 --ntest 4000 --seed 0 --device 0
python interpolation_hd.py --epochs 4000 --network modifiedmlp --datatype fractal --dim 2 --features 100 --layers 10 --ntrain 2000 --ntest 4000 --seed 1 --device 0
python interpolation_hd.py --epochs 4000 --network modifiedmlp --datatype fractal --dim 2 --features 100 --layers 10 --ntrain 2000 --ntest 4000 --seed 2 --device 0

python interpolation_hd.py --epochs 4000 --network kan --datatype fractal --dim 2 --kanshape 32,32 --degree 32 --ntrain 2000 --ntest 4000 --seed 0 --device 0
python interpolation_hd.py --epochs 4000 --network kan --datatype fractal --dim 2 --kanshape 24,24,24 --degree 24 --ntrain 2000 --ntest 4000 --seed 1 --device 0
python interpolation_hd.py --epochs 4000 --network kan --datatype fractal --dim 2 --kanshape 48 --degree 48 --ntrain 2000 --ntest 4000 --seed 2 --device 0

python interpolation_hd.py --epochs 4000 --network sinckan --kanshape 32 --degree 100 --normalization 0 --datatype fractal --dim 2 --len_h 6 --noise 0 --activation none --interval 0,1 --skip 0 --ntrain 2000 --ntest 4000 --seed 0 --device 0
python interpolation_hd.py --epochs 4000 --network sinckan --kanshape 32 --degree 100 --normalization 0 --datatype fractal --dim 2 --len_h 6 --noise 0 --activation none --interval 0,1 --skip 0 --ntrain 2000 --ntest 4000 --seed 1 --device 0
python interpolation_hd.py --epochs 4000 --network sinckan --kanshape 32 --degree 100 --normalization 0 --datatype fractal --dim 2 --len_h 6 --noise 0 --activation none --interval 0,1 --skip 0 --ntrain 2000 --ntest 4000 --seed 2 --device 0

python interpolation_hd.py --epochs 4000 --network modifiedsinckan --datatype fractal --dim 2 --features 96 --layers 6 --degree 96 --len_h 4 --init_h 2 --decay inverse --activation tanh --uv_activation none --gate_mode sigmoid --skip 1 --ntrain 2000 --ntest 4000 --seed 0 --device 0
python interpolation_hd.py --epochs 4000 --network modifiedsinckan --datatype fractal --dim 2 --features 96 --layers 6 --degree 96 --len_h 4 --init_h 2 --decay inverse --activation tanh --uv_activation none --gate_mode sigmoid --skip 1 --ntrain 2000 --ntest 4000 --seed 1 --device 0
python interpolation_hd.py --epochs 4000 --network modifiedsinckan --datatype fractal --dim 2 --features 96 --layers 6 --degree 96 --len_h 4 --init_h 2 --decay inverse --activation tanh --uv_activation none --gate_mode sigmoid --skip 1 --ntrain 2000 --ntest 4000 --seed 2 --device 0
