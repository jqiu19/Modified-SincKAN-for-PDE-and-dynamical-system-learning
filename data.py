import numpy as np
import scipy
from scipy.special import ellipj, ellipkinc, ellipeinc, jn, yn, lpmv, sph_harm, gamma
from numpy import arange, exp, cos, sin, e, pi, absolute, meshgrid


def get_data(datatype):
    if datatype=='100D':
        generate_data= function_100D
    elif datatype=='4D':
        generate_data= function_4D
    elif datatype=='allen_cahn':
        generate_data= allen_cahn
    elif datatype=='bl':
        generate_data= boundary_layer
    elif datatype=='bl2d':
        generate_data= boundary_layer2d
    elif datatype=='burgers_1d':
        generate_data= burgers_1d
    elif datatype=='cdiff':
        generate_data= cdiff
    elif datatype=='control_100d':
        generate_data= control_100d
    elif datatype=='damped_oscillator':
        generate_data= damped_oscillator
    elif datatype=='double_exponential':
        generate_data= double_exponential
    elif datatype=='ellipeinc':
        generate_data= incomplete_elliptic_integral_of_the_second_kind
    elif datatype=='ellipj':
        generate_data= jacobian_elliptic_function
    elif datatype=='ellipkinc':
        generate_data= incomplete_elliptic_integral_of_the_first_kind
    elif datatype=='endpoint':
        generate_data= endpoint_singularity_function
    elif datatype=='fractal':
        generate_data= fractal_function
    elif datatype=='fraction':
        generate_data= fraction
    elif datatype=='heat_1d':
        generate_data= heat_1d_solution
    elif datatype=='image_50d':
        generate_data= image_50d
    elif datatype=='jn':
        generate_data= bessel_function_of_the_first_kind
    elif datatype=='lpmv':
        generate_data= associated_legendre_function_of_the_first_kind
    elif datatype=='multimodal1':
        generate_data= multimodal_function1
    elif datatype=='multimodal2':
        generate_data= multimodal_function2
    elif datatype=='multitone_signal':
        generate_data= multitone_signal
    elif datatype=='multi_sqrt':
        generate_data= multi_sqrt_function
    elif datatype=='nonlinear':
        generate_data= nonlinear
    elif datatype=='ns_tg':
        generate_data= ns_tg
    elif datatype=='ode_100d':
        generate_data= ode_100d
    elif datatype=='ode_4d':
        generate_data= ode_4d
    elif datatype=='pbl':
        generate_data= pbl
    elif datatype=='pde_100d':
        generate_data= pde_100d
    elif datatype=='pde_param_4d':
        generate_data= pde_param_4d
    elif datatype=='piecewise':
        generate_data= piece_wise_function
    elif datatype=='poisson':
        generate_data= poisson
    elif datatype=='poisson_1d':
        generate_data= poisson_1d_solution
    elif datatype=='poisson_2d_solution':
        generate_data= poisson_2d_solution
    elif datatype=='poisson_sin':
        generate_data= poisson_sin
    elif datatype=='rlc_response':
        generate_data= rlc_response
    elif datatype=='schrodinger':
        generate_data= schrodinger
    elif datatype=='signal_50d':
        generate_data= signal_50d
    elif datatype=='sin_high':
        generate_data= sin_high
    elif datatype=='sin_low':
        generate_data= sin_low
    elif datatype=='sine_gordon':
        generate_data= sine_gordon
    elif datatype=='spectral_bias':
        generate_data= spectral_bias
    elif datatype=='spectral_bias2D':
        generate_data= spectral_bias2D
    elif datatype=='spectral_bias_2D':
        generate_data= spectral_bias_2d
    elif datatype=='stochastic_pde_100d':
        generate_data= stochastic_pde_100d
    elif datatype=='sph_harm01':
        generate_data= spherical_harmonics01
    elif datatype=='sph_harm02':
        generate_data= spherical_harmonics02
    elif datatype=='sph_harm11':
        generate_data= spherical_harmonics11
    elif datatype=='sph_harm12':
        generate_data= spherical_harmonics12
    elif datatype=='sph_harm22':
        generate_data= spherical_harmonics22
    elif datatype=='sqrt':
        generate_data= sqrt
    elif datatype=='surrogate_100d':
        generate_data= surrogate_100d
    elif datatype=='sweep_chirp':
        generate_data= sweep_chirp
    elif datatype=='singular_frac':
        generate_data= singular_frac
    elif datatype=='t_nonlinear':
        generate_data= t_nonlinear
    elif datatype=='yn':
        generate_data= bessel_function_of_the_second_kind
    else:
        assert False, f'{datatype} does not exist'
    return generate_data


def piece_wise_function(x):
    y= np.zeros_like(x)
    mask1= x< 0.5
    y[mask1]= x[mask1]**2
    mask2=(x>= 0.5)&(x<= 1)
    y[mask2]= np.cos(2* np.pi* x[mask2])
    mask3= x> 1
    y[mask3]= np.log(x[mask3]- 1)/ np.log(2)- np.cos(2* np.pi* x[mask3])
    return y


def sqrt(x):
    y= np.zeros_like(x)
    mask1= x< 0
    y[mask1]= 0
    mask2= x>= 0
    y[mask2]= x[mask2]** 0.5
    return y


def boundary_layer(x, alpha=100):
    y= np.exp(-x* alpha)
    return y


def boundary_layer2d(x, y, alpha=100):
    y= np.exp(-x* alpha)+ np.exp(-y* alpha)
    return y


def endpoint_singularity_function(x):
    y= np.zeros_like(x)
    mask1= x< 0
    y[mask1]= 0
    mask2=(0<= x)&(x<= 1)
    y[mask2]= x[mask2]** 0.5*(1- x[mask2])**(3/ 4)
    mask3= x> 1
    y[mask3]= 0
    return y


def sin_low(x):
    y= np.zeros_like(x)
    mask1=(-1<= x)&(x<= 1)
    y[mask1]= np.sin(4* np.pi* x[mask1])
    return y


def sin_high(x):
    y= np.zeros_like(x)
    mask=(-1<= x)&(x<= 1)
    y[mask]= np.sin(400* np.pi* x[mask])
    return y


def double_exponential(x):
    y= np.zeros_like(x)
    mask1= x< 0
    y[mask1]= 0
    mask2=(0<= x)&(x<= 1)
    y[mask2]=(x[mask2]*(1- x[mask2])* e**(-x[mask2]))/(0.5** 2+(x[mask2]- 0.5)** 2)
    mask3= x> 1
    y[mask3]= 0
    return y


def damped_oscillator(x):
    x = np.asarray(x)
    return np.exp(-4.0 * x) * np.cos(24.0 * np.pi * x)


def spectral_bias(x):
    y= np.zeros_like(x)
    mask1= x<-1
    y[mask1]= 0
    mask2=(-1<= x)&(x<= 0)
    y[mask2]= 5+(np.sin(x[mask2])+ np.sin(2*x[mask2])+ np.sin(3*x[mask2])+ np.sin(4*x[mask2]))
    mask3=(0< x)&(x<= 1)
    y[mask3]= np.cos(10*x[mask3])
    mask4= x> 1
    y[mask4]= 0
    return y


def multitone_signal(x):
    x = np.asarray(x)
    return (np.sin(6.0 * np.pi * x)
            + 0.5 * np.sin(18.0 * np.pi * x + 0.3)
            + 0.25 * np.sin(42.0 * np.pi * x + 0.1))


def sweep_chirp(x):
    x = np.asarray(x)
    phase = 8.0 * np.pi * x + 36.0 * np.pi * x ** 2
    return np.sin(phase)


def rlc_response(x):
    x = np.asarray(x)
    return np.exp(-3.5 * x) * (np.cos(20.0 * np.pi * x) + 0.15 * np.sin(20.0 * np.pi * x))


def heat_1d_solution(x, t=0.03):
    x = np.asarray(x)
    return np.exp(-np.pi ** 2 * t) * np.sin(np.pi * x) + 0.35 * np.exp(-9.0 * np.pi ** 2 * t) * np.sin(3.0 * np.pi * x)


def poisson_1d_solution(x):
    x = np.asarray(x)
    return x * (1.0 - x) + 0.1 * np.sin(5.0 * np.pi * x)


def spectral_bias_2d(X: np.ndarray) -> np.ndarray:
    return spectral_bias2D(X)

def spectral_bias2D(X: np.ndarray) -> np.ndarray:
    """
    2-D discontinuous test function from §4.1.1.
    X : array-like, shape (..., 2)   # last axis = (x, y)
    returns array of shape X.shape[:-1]
    """
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 2:
        raise ValueError('Last dimension of X must be 2 (x and y coordinates).')

    x = X[..., 0]          # every x-coordinate
    y = X[..., 1]          # every y-coordinate

    def h(t):
        """1-D piecewise definition h(t) from the paper."""
        out = np.empty_like(t)
        mask = t < 0
        out[mask] = 5 + (np.sin(t[mask]) +
                         np.sin(2*t[mask]) +
                         np.sin(3*t[mask]) +
                         np.sin(4*t[mask]))
        out[~mask] = np.cos(10*t[~mask])
        return out

    return h(x) * h(y)     # tensor-product f(x,y)=h(x)·h(y)


def jacobian_elliptic_function(x, k=0.5):
    sn, cn, dn, ph= ellipj(x, k)
    y= sn
    return y


def incomplete_elliptic_integral_of_the_first_kind(x, k=0.5):
    y= ellipkinc(x, k)
    return y


def incomplete_elliptic_integral_of_the_second_kind(x, k=0.5):
    y= ellipeinc(x, k)
    return y


def bessel_function_of_the_first_kind(x, n=3):
    y= jn(n, x)
    return y


def bessel_function_of_the_second_kind(x, n=3):
    y= yn(n, x)
    return y


def associated_legendre_function_of_the_first_kind(x, n=3):
    y= lpmv(1, n, x)
    return y


def spherical_harmonics01(theta):
    l= 1
    m= 0
    phi= 0
    y= sph_harm(m, l, phi, theta).real
    return y


def spherical_harmonics11(theta):
    l= 1
    m= 1
    phi= 0
    y= sph_harm(m, l, phi, theta).real
    return y


def spherical_harmonics02(theta):
    l= 2
    m= 0
    phi= 0
    y= sph_harm(m, l, phi, theta).real
    return y


def spherical_harmonics12(theta):
    l= 2
    m= 1
    phi= 0
    y= sph_harm(m, l, phi, theta).real
    return y


def spherical_harmonics22(theta):
    l= 2
    m= 2
    phi= 0
    y= sph_harm(m, l, phi, theta).real
    return y


def fractal_function(x, y=None):
    if y is None:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != 2:
            raise ValueError("fractal_function expects points with last dimension 2.")
        x0 = x[..., 0]
        y0 = x[..., 1]
    else:
        x0 = np.asarray(x, dtype=float)
        y0 = np.asarray(y, dtype=float)
    z= np.sin(10* np.pi* x0)* np.cos(10* np.pi* y0)+ np.sin(np.pi*(x0** 2+ y0** 2))
    z+= np.abs(x0- y0)+(np.sin(5* x0* y0)/(0.1+ np.abs(x0+ y0)))
    z*= np.exp(-0.1*(x0** 2+ y0** 2))
    return z


def poisson_2d_solution(X):
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 2:
        raise ValueError("poisson_2d_solution expects points with last dimension 2.")
    x = X[..., 0]
    y = X[..., 1]
    return np.sin(np.pi * x) * np.sin(np.pi * y) + 0.15 * np.sin(3.0 * np.pi * x) * np.sin(2.0 * np.pi * y)


def ode_4d(X):
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 4:
        raise ValueError("ode_4d expects points with last dimension 4.")
    # Exact solution of a linear 4D ODE system y' = A y via matrix exponential.
    # This follows the standard matrix-exponential treatment of linear systems.
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-4.0, -0.4, 1.2, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.8, 0.0, -9.0, -0.3],
    ])
    T = 1.0
    Phi = scipy.linalg.expm(T * A)
    yT = np.einsum('ij,...j->...i', Phi, X)
    return yT[..., 0]


def pde_param_4d(X):
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 4:
        raise ValueError("pde_param_4d expects points with last dimension 4.")
    x, y, alpha, t = [X[..., i] for i in range(4)]
    alpha = np.maximum(alpha, 0.0)
    t = np.maximum(t, 0.0)
    return np.exp(-2.0 * np.pi ** 2 * alpha * t) * np.sin(np.pi * x) * np.sin(np.pi * y)


def signal_50d(X):
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 50:
        raise ValueError("signal_50d expects points with last dimension 50.")
    idx = np.arange(1, 51, dtype=float)
    harmonic = np.sin(np.pi * idx * X)
    modulation = np.cos(0.5 * np.pi * X)
    return np.mean(harmonic * modulation, axis=-1) + 0.05 * np.sum(X[..., :10] ** 2, axis=-1)


def image_50d(X):
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 50:
        raise ValueError("image_50d expects points with last dimension 50.")
    # Patch-style implicit image task:
    # first 49 dims encode a 7x7 grayscale patch, last dim is a normalized
    # position/context feature. The output is a center-pixel surrogate.
    patch = X[..., :49].reshape(X.shape[:-1] + (7, 7))
    pos = X[..., 49]

    center = patch[..., 3, 3]
    cross = (
        patch[..., 2, 3] + patch[..., 4, 3] +
        patch[..., 3, 2] + patch[..., 3, 4]
    ) / 4.0
    diag = (
        patch[..., 2, 2] + patch[..., 2, 4] +
        patch[..., 4, 2] + patch[..., 4, 4]
    ) / 4.0
    patch_mean = np.mean(patch, axis=(-2, -1))
    laplace = 4.0 * center - (
        patch[..., 2, 3] + patch[..., 4, 3] +
        patch[..., 3, 2] + patch[..., 3, 4]
    )

    return (
        0.55 * center
        + 0.20 * cross
        + 0.10 * diag
        + 0.08 * patch_mean
        - 0.06 * laplace
        + 0.05 * np.sin(2.0 * np.pi * pos) * (center - cross)
    )


def pde_100d(X):
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 100:
        raise ValueError("pde_100d expects points with last dimension 100.")
    idx = np.arange(1, 101, dtype=float)
    weights = np.exp(-0.03 * idx)
    return np.exp(-0.02 * np.sum(X ** 2, axis=-1)) * (1.0 + 0.25 * np.sum(weights * np.sin(np.pi * X), axis=-1))


def stochastic_pde_100d(X):
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 100:
        raise ValueError("stochastic_pde_100d expects points with last dimension 100.")

    # 100-term KL-coefficient surrogate for a stochastic elliptic PDE:
    # -div(a(x, xi) grad u(x, xi)) = f(x), xi in R^100.
    # We report a fixed-point response u(x*, xi) at x*=(0.37, 0.63).
    idx = np.arange(1, 101, dtype=float)
    lam = np.exp(-0.08 * idx)
    x_star = np.array([0.37, 0.63], dtype=float)
    phi = np.sin(np.pi * idx * x_star[0]) * np.sin(np.pi * ((idx % 19) + 1.0) * x_star[1])

    xi = X
    kl_field = np.sum(np.sqrt(lam) * phi * xi, axis=-1)
    a_star = np.exp(0.15 + 0.35 * kl_field)

    rhs_star = np.sin(np.pi * x_star[0]) * np.sin(np.pi * x_star[1])
    modal_response = np.sum(lam * np.sin(0.5 * np.pi * xi), axis=-1) / np.sum(lam)
    correction = 0.05 * np.sum(np.sqrt(lam) * xi, axis=-1) / np.sum(np.sqrt(lam))

    return (rhs_star / (2.0 * np.pi ** 2 * a_star)) * (1.0 + 0.25 * modal_response) + correction


def surrogate_100d(X):
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 100:
        raise ValueError("surrogate_100d expects points with last dimension 100.")
    low_rank = np.mean(np.sin(0.5 * np.pi * X[..., :50]), axis=-1)
    high_rank = np.mean(np.cos(1.5 * np.pi * X[..., 50:]), axis=-1)
    interaction = 0.1 * np.sum(X[..., :20] * X[..., 20:40], axis=-1)
    return low_rank + 0.8 * high_rank + interaction


def ode_100d(X):
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 100:
        raise ValueError("ode_100d expects points with last dimension 100.")
    idx = np.arange(1, 101, dtype=float)
    decay = np.exp(-0.01 * idx)
    return np.sum(decay * np.sin(idx * np.pi * X / 20.0), axis=-1) / np.sum(decay)


def control_100d(X):
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 100:
        raise ValueError("control_100d expects points with last dimension 100.")
    state = np.tanh(X[..., :50])
    control = X[..., 50:]
    return 0.6 * np.mean(state, axis=-1) + 0.4 * np.mean(np.sin(np.pi * control), axis=-1) + 0.05 * np.sum(state * control, axis=-1) / 50.0


def multimodal_function1(x, y):
    z=-absolute(sin(x)* cos(y)* exp(absolute(1-(np.sqrt(x** 2+ y** 2)/ pi))))
    return z


def multimodal_function2(x, y):
    z=-20.0* exp(-0.2* np.sqrt(0.5*(x** 2+ y** 2)))- exp(0.5*(cos(2* pi* x)+ cos(2 * pi* y)))+ e+ 20
    return z


def function_4D(x1, x2, x3, x4):
    z= np.exp(0.5* np.sin(np.pi*(x1** 2+ x2** 2))+ 0.5* np.sin(np.pi*(x3** 2+ x4**2)))
    return z


def function_100D(x):
    x= np.asarray(x)
    if len(x)!= 100:
        raise ValueError("Input should be a 100-dimensional vector.")
    z= exp(0.01* np.sum(sin(pi* x/ 2)** 2))
    return z


def pbl(x, alpha=100):
    eps= 1/ alpha
    z= 1+ x+(np.exp(x/ eps)- 1)/(np.exp(1/ eps)- 1)
    return z


def nonlinear(x):
    z= x**(5/ 2)*(1- x)** 2+ x** 3+ 1
    return z


def burgers_1d(x, t, a=0.1, nu=0.01):
    z= a/ 2- a/ 2* np.tanh(a*(x- a* t/ 2)/ 4/ nu)
    return z


def ns_tg(x, y, t, nu, k=1):
    u=-np.cos(k* x)* np.sin(k* y)* np.exp(-2* t* nu)
    v= np.sin(k* x)* np.cos(k* y)* np.exp(-2* t* nu)
    p=-(np.cos(2* k* x)+ np.sin(2* k* y))* np.exp(-4* t* nu)/ 4
    return u, v, p


def t_nonlinear(x, t):
    z= np.cos((x+ 2)*(t+ 1))
    return z


def cdiff(x, t, a, eps, N=6):
    Z= 0
    for k in range(N):
        Z= Z+ np.sin(k*(x- a* t))* np.exp(-eps* k** 2* t)
    return Z


def poisson(x, alpha):
    y= np.exp(-alpha* np.sum(x** 2, axis=1))[:, None]
    return y


def allen_cahn(x, alpha, c):
    B=-alpha* np.sum(x** 2, axis=1)
    return np.exp(B)[:, None]


def sine_gordon(x, alpha, c):
    A= np.mean(np.exp(-c* x[:,:-2]* x[:, 1:-1]* x[:, 2:]), axis=1)
    B=-alpha* np.sum(x** 2, axis=1)
    return(A* np.exp(B))[:, None]


def poisson_sin(x, dim):
    temp= np.sum(x, axis=1)/ dim
    y=(temp)** 2+ np.sin(temp)
    return y[:, None]


def schrodinger(x, coeffs):
    hbar= coeffs['hbar']
    m= coeffs['m']
    omega= coeffs['omega']
    vec_s= coeffs['vec_s']
    vec_mu= coeffs['vec_mu']
    x0= coeffs['x0']
    x= x0*x
    alpha= vec_mu- vec_s/ 2
    # Associated Laguerre Polynomials for n=1
    # L= lambda x: x**(-alpha)* np.exp(x)*(-x**(1+ alpha)* np.exp(-x)+(1+ alpha)* x**(alpha)* np.exp(-x))
    L= lambda x: 1+ alpha- x
    func_plus= lambda x:(np.exp(-m* omega/ 2/ hbar* x** 2)* x**((1- 1)/ 2)*(
        1+ vec_mu[0]- 1/ 2- m* omega/ hbar* x** 2))** 2
    int_plus,_= scipy.integrate.quad(func_plus,-np.inf, np.inf)

    func_minus= lambda x:(np.exp(-m* omega/ 2/ hbar* x** 2)* x**((1+ 1)/ 2)*(
        1+ vec_mu[0]+ 1/ 2- m* omega/ hbar* x** 2))** 2
    int_minus,_= scipy.integrate.quad(func_minus,-np.inf, np.inf)

    c_plus= np.sqrt(1/ int_plus)
    c_minus= np.sqrt(1/ int_minus)

    # c_plus= 1
    # c_minus= 1

    vec_psi=(c_plus*(vec_s== 1)+ c_minus*(vec_s==-1))* np.exp(-m* omega/ 2/ hbar* x** 2)* x**(
        (1- vec_s)/ 2)* L(m* omega/ hbar* x** 2)

    psi= np.prod(vec_psi, axis=1)

    return psi[:, None]


def fraction(x):
    # return x*(np.abs(1- x**2))**(alpha/ 2)
    return x** 3*(1- x)** 3


def singular_frac(x, alpha, d=1):
    s= alpha/ 2
    u= 2**(-2* s)* gamma(d/ 2)/ gamma(d/ 2+ s)/ gamma(1+ s)\
        *(1- np.sum(x** 2, axis=1))** s
    return u[:, None]
