#
#======================================================#
# Modified SincKAN layers for Neural ODE vector fields
#======================================================#

@inline function sincoverx(x; eps=1.0f-3)
    x2 = x^2
    if abs(x) < eps
        1 - x2 / 6 + x2^2 / 120 - x2^3 / 5040 + x2^4 / 362880
    else
        sin(x) / x
    end
end

_zeros_init(::AbstractRNG, dims...) = zeros(Float32, dims...)

@concrete struct SincDense <: LuxCore.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    degree::Int
    len_h::Int
    init_h
    normalizer
    apply_activation::Bool
    use_skip::Bool
    init_C
    init_W
    init_b
    ks
    hs
end

function SincDense(
    in_dims::Int,
    out_dims::Int,
    degree::Int;
    len_h::Int = 1,
    init_h = 2.0f0,
    normalizer = tanh_fast,
    apply_activation::Bool = true,
    use_skip::Bool = true,
    init_C = glorot_uniform,
    init_W = glorot_uniform,
    init_b = _zeros_init,
)
    d = Int(degree)
    iseven(d) || error("SincDense degree must be even in this implementation; got degree=$(d).")
    ks = Float32.(collect((-d ÷ 2):(d ÷ 2)))
    hs = Float32.(1.0 ./ (Float32(init_h) .* (1:len_h)))
    SincDense(
        in_dims, out_dims, d, len_h, Float32(init_h),
        NNlib.fast_act(normalizer), apply_activation, use_skip,
        init_C, init_W, init_b, ks, hs,
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, l::SincDense)
    basis_len = l.in_dims * l.len_h * (l.degree + 1)
    p = (;
        C = l.init_C(rng, l.out_dims, basis_len),
    )
    if l.use_skip
        p = (;
            p...,
            W = l.init_W(rng, l.out_dims, l.in_dims),
            b = l.init_b(rng, l.out_dims, 1),
        )
    end
    p
end

LuxCore.initialstates(::AbstractRNG, ::SincDense) = (;)
LuxCore.statelength(::SincDense) = 0

function LuxCore.parameterlength(l::SincDense)
    len = l.in_dims * l.len_h * (l.degree + 1) * l.out_dims
    if l.use_skip
        len += l.in_dims * l.out_dims + l.out_dims
    end
    len
end

function (l::SincDense)(x::AbstractArray, p, st)
    size_in = size(x)
    size_out = (l.out_dims, size_in[2:end]...,)
    x = reshape(x, l.in_dims, :)
    K = size(x, 2)

    x_basis = l.apply_activation ? _broadcast(l.normalizer, x) : x
    # Keep the basis construction as a single broadcasted array operation.
    # A comprehension creates a very long tuple/vector here, and Zygote tries to
    # differentiate through the internal push!, which is unsupported.
    xx = reshape(x_basis, l.in_dims, 1, 1, K) ./
         reshape(l.hs, 1, l.len_h, 1, 1) .-
         reshape(l.ks, 1, 1, l.degree + 1, 1)
    basis = reshape(sincoverx.(xx), l.in_dims * l.len_h * (l.degree + 1), K)
    y = p.C * basis

    if l.use_skip
        y = y .+ (p.W * x .+ p.b)
    end

    reshape(y, size_out), st
end

@concrete struct ModifiedSincKANBlock <: LuxCore.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    alpha
    u_layer
    v_layer
    init_Wg
    init_bg
end

function ModifiedSincKANBlock(
    in_dims::Int,
    out_dims::Int;
    u_degree::Int = 16,
    u_len_h::Int = 3,
    v_degree::Int = 28,
    v_len_h::Int = 5,
    init_h = 2.0f0,
    alpha = 0.09f0,
    normalizer = tanh_fast,
    apply_activation::Bool = true,
    init_C = glorot_uniform,
    init_W = glorot_uniform,
    init_b = _zeros_init,
)
    u_layer = SincDense(
        in_dims, out_dims, u_degree;
        len_h = u_len_h, init_h, normalizer, apply_activation,
        use_skip = true, init_C, init_W, init_b,
    )
    v_layer = SincDense(
        in_dims, out_dims, v_degree;
        len_h = v_len_h, init_h, normalizer, apply_activation,
        use_skip = true, init_C, init_W, init_b,
    )
    ModifiedSincKANBlock(
        in_dims, out_dims, Float32(alpha), u_layer, v_layer,
        init_W, init_b,
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, l::ModifiedSincKANBlock)
    (;
        u = LuxCore.initialparameters(rng, l.u_layer),
        v = LuxCore.initialparameters(rng, l.v_layer),
        Wg = l.init_Wg(rng, l.out_dims, l.in_dims),
        bg = l.init_bg(rng, l.out_dims, 1),
    )
end

function LuxCore.initialstates(rng::AbstractRNG, l::ModifiedSincKANBlock)
    (;
        u = LuxCore.initialstates(rng, l.u_layer),
        v = LuxCore.initialstates(rng, l.v_layer),
    )
end

function LuxCore.statelength(l::ModifiedSincKANBlock)
    LuxCore.statelength(l.u_layer) + LuxCore.statelength(l.v_layer)
end

function LuxCore.parameterlength(l::ModifiedSincKANBlock)
    LuxCore.parameterlength(l.u_layer) +
    LuxCore.parameterlength(l.v_layer) +
    l.out_dims * l.in_dims +
    l.out_dims
end

function (l::ModifiedSincKANBlock)(x::AbstractArray, p, st)
    size_in = size(x)
    size_out = (l.out_dims, size_in[2:end]...,)
    x2d = reshape(x, l.in_dims, :)

    u, stu = l.u_layer(x, p.u, st.u)
    v, stv = l.v_layer(x, p.v, st.v)
    u2d = reshape(u, l.out_dims, :)
    v2d = reshape(v, l.out_dims, :)

    gate = sigmoid.(p.Wg * x2d .+ p.bg)
    mixed = gate .* u2d .+ (1 .- gate) .* v2d

    y = if l.in_dims == l.out_dims
        x2d .+ l.alpha .* mixed
    else
        mixed
    end

    reshape(y, size_out), (; u = stu, v = stv)
end
