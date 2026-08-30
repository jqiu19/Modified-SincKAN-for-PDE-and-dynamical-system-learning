using Random, Lux, LuxCore, LinearAlgebra, Statistics
using NNlib, ConcreteStructs, WeightInitializers, ChainRulesCore
using ComponentArrays
using OrdinaryDiffEq, DiffEqFlux
using Flux
using Flux: update!
using MAT
using Plots
using ProgressBars
using Zygote: gradient as Zgrad

ENV["GKSwstype"] = "100"

env_int(name, default) = haskey(ENV, name) ? parse(Int, ENV[name]) : default
env_float32(name, default) = haskey(ENV, name) ? Float32(parse(Float64, ENV[name])) : Float32(default)
env_float64(name, default) = haskey(ENV, name) ? parse(Float64, ENV[name]) : default

dir = @__DIR__
dir = dir * "/"
cd(dir)

include("src/KolmogorovArnold.jl")
using .KolmogorovArnold

model_name = lowercase(get(ENV, "MODEL", "modifiedsinckan"))
seed = env_int("SEED", 0)
run_tag = get(ENV, "RUN_TAG", "table_" * model_name * "_seed" * string(seed))
fname = "LV_" * run_tag
add_path = "results_" * run_tag * "/"
figpath = dir * add_path * "figs"
ckptpath = dir * add_path * "checkpoints"
mkpath(figpath)
mkpath(figpath * "/training")
mkpath(ckptpath)

function lotka!(du, u, p, t)
    alpha, beta, gamma, delta = p
    du[1] = alpha * u[1] - beta * u[2] * u[1]
    du[2] = gamma * u[1] * u[2] - delta * u[2]
end

timestep = 0.1
n_plot_save = env_int("N_PLOT_SAVE", 1000)
rng = Random.default_rng()
Random.seed!(rng, seed)
tspan = (0.0, 14)
tspan_train = (0.0, 3.5)
u0 = [1, 1]
p_true = Float32[1.5, 1, 1, 3]
prob = ODEProblem(lotka!, u0, tspan, p_true)
solution = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = timestep)
end_index = Int64(floor(length(solution.t) * tspan_train[2] / tspan[2]))
t = solution.t
t_train = t[1:end_index]
X = Array(solution)
Xn = deepcopy(X)

@concrete struct ModifiedMLPBlock <: LuxCore.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    activation
    init_W
    init_b
end

function ModifiedMLPBlock(
    in_dims::Int,
    out_dims::Int;
    activation = tanh_fast,
    init_W = glorot_uniform,
    init_b = (rng, dims...) -> zeros(Float32, dims...),
)
    ModifiedMLPBlock(in_dims, out_dims, NNlib.fast_act(activation), init_W, init_b)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::ModifiedMLPBlock)
    (;
        Wu = l.init_W(rng, l.out_dims, l.in_dims),
        bu = l.init_b(rng, l.out_dims, 1),
        Wv = l.init_W(rng, l.out_dims, l.in_dims),
        bv = l.init_b(rng, l.out_dims, 1),
        Wg = l.init_W(rng, l.out_dims, l.in_dims),
        bg = l.init_b(rng, l.out_dims, 1),
    )
end

LuxCore.initialstates(::AbstractRNG, ::ModifiedMLPBlock) = (;)
LuxCore.statelength(::ModifiedMLPBlock) = 0
LuxCore.parameterlength(l::ModifiedMLPBlock) = 3 * (l.out_dims * l.in_dims + l.out_dims)

function (l::ModifiedMLPBlock)(x::AbstractArray, p, st)
    size_in = size(x)
    size_out = (l.out_dims, size_in[2:end]...,)
    x2d = reshape(x, l.in_dims, :)
    u = l.activation.(p.Wu * x2d .+ p.bu)
    v = l.activation.(p.Wv * x2d .+ p.bv)
    gate = sigmoid.(p.Wg * x2d .+ p.bg)
    y = gate .* u .+ (1 .- gate) .* v
    reshape(y, size_out), st
end

function build_model(model_name)
    width = env_int("LAYER_WIDTH", 40)
    depth = env_int("NUM_LAYERS", 2)
    grid_size = env_int("GRID_SIZE", 10)
    u_degree = env_int("U_DEGREE", 16)
    u_len_h = env_int("U_LEN_H", 1)
    v_degree = env_int("V_DEGREE", 34)
    v_len_h = env_int("V_LEN_H", 1)
    degree = env_int("DEGREE", 16)
    len_h = env_int("LEN_H", 1)
    alpha = env_float32("ALPHA", 0.09)
    normalizer = tanh_fast

    if model_name == "kanode" || model_name == "kan"
        model = Lux.Chain(
            KDense(2, width, grid_size; use_base_act = true, basis_func = rbf, normalizer),
            KDense(width, 2, grid_size; use_base_act = true, basis_func = rbf, normalizer),
        )
        size_info = [2, width, grid_size]
    elseif model_name == "mlp"
        model = Lux.Chain(Lux.Dense(2 => width, tanh), Lux.Dense(width => 2))
        size_info = [2, width]
    elseif model_name == "modifiedmlp"
        hidden_layers = [ModifiedMLPBlock(width, width; activation = tanh_fast) for _ in 1:max(depth - 1, 0)]
        model = Lux.Chain(
            ModifiedMLPBlock(2, width; activation = tanh_fast),
            hidden_layers...,
            ModifiedMLPBlock(width, 2; activation = tanh_fast),
        )
        size_info = [2, width, depth, "separate_sigmoid"]
    elseif model_name == "sinckan"
        model = Lux.Chain(
            SincDense(2, width, degree; len_h, normalizer, use_skip = true),
            SincDense(width, 2, degree; len_h, normalizer, use_skip = true),
        )
        size_info = [2, width, degree, len_h]
    elseif model_name == "modifiedsinckan"
        model = Lux.Chain(
            ModifiedSincKANBlock(2, width; u_degree, u_len_h, v_degree, v_len_h, alpha, normalizer),
            ModifiedSincKANBlock(width, 2; u_degree, u_len_h, v_degree, v_len_h, alpha, normalizer),
        )
        size_info = [2, width, u_degree, u_len_h, v_degree, v_len_h]
    else
        error("Unknown MODEL=$(model_name)")
    end

    return model, size_info
end

model, size_info = build_model(model_name)
pM, stM = Lux.setup(rng, model)
pM_data = getdata(ComponentArray(pM))
pM_axis = getaxes(ComponentArray(pM))
p = deepcopy(pM_data) ./ 1e5

println("run_tag: ", run_tag)
println("model: ", model_name)
println("seed: ", seed)
println("parameter size: ", length(p))
println("size_info: ", size_info)

train_node = NeuralODE(model, tspan_train, Tsit5(), saveat = t_train)
train_node_test = NeuralODE(model, tspan, Tsit5(), saveat = t)

function predict(p)
    Array(train_node(u0, ComponentArray(p, pM_axis), stM)[1])
end

function predict_test(p)
    Array(train_node_test(u0, ComponentArray(p, pM_axis), stM)[1])
end

function loss(p)
    mean(abs2, Xn[:, 1:end_index] .- predict(p))
end

function loss_train(p)
    mean(abs2, Xn[:, 1:end_index] .- predict(p))
end

function loss_test(p)
    mean(abs2, Xn .- predict_test(p))
end

lr = env_float64("LR", model_name == "mlp" ? 1e-2 : 5e-4)
N_iter = env_float64("N_ITER", 1e5)
opt = Flux.Adam(lr)
i_current = 1
l = []
l_test = []
p_list = []
println("training config: lr=", lr, " N_iter=", N_iter, " n_plot_save=", n_plot_save)

function plot_save(l, l_test, p_list, epoch)
    plt = Plots.plot(l, yaxis = :log, label = "train", dpi = 600)
    plot!(l_test, yaxis = :log, label = "test")
    xlabel!("Epoch")
    ylabel!("Loss")
    png(plt, string(figpath, "/loss.png"))
    println("minimum train loss: ", minimum(l), " minimum test loss: ", minimum(l_test))

    p_curr = p_list[end]
    train_node_full = NeuralODE(model, tspan, Tsit5(), saveat = timestep)
    pred_sol = train_node_full(u0, ComponentArray(p_curr, pM_axis), stM)[1]
    plt = scatter(solution, alpha = 0.75)
    plot!(pred_sol)
    vline!([3.5], color = :black, label = "train/test split")
    xlabel!("Time [s]")
    ylabel!("x, y")
    png(plt, string(figpath, "/training/results.png"))

    p_list_ = zeros(size(p_list, 1), size(p_list[1], 1), size(p_list[1], 2))
    for j = 1:size(p_list, 1)
        p_list_[j, 1:length(p_list[j]), :] = p_list[j]
    end
    l_ = zeros(size(p_list, 1))
    l_test_ = zeros(size(p_list, 1))
    for j = 1:size(l, 1)
        l_[j] = l[j]
        l_test_[j] = l_test[j]
    end

    file = matopen(dir * add_path * "checkpoints/" * fname * "_results.mat", "w")
    write(file, "p_list", p_list_)
    write(file, "loss", l_)
    write(file, "loss_test", l_test_)
    write(file, "size_info", size_info)
    write(file, "seed", seed)
    write(file, "model_name", model_name)
    close(file)
end

total_train_iters = Int(N_iter - i_current)
train_start_time = time()
iters = tqdm(1:total_train_iters)
for i in iters
    global i_current
    grad = Zgrad(loss, p)[1]
    update!(opt, p, grad)

    loss_curr = deepcopy(loss_train(p))
    loss_curr_test = deepcopy(loss_test(p))
    append!(l, [loss_curr])
    append!(l_test, [loss_curr_test])
    append!(p_list, [deepcopy(p)])

    set_description(iters, string("Loss:", loss_curr))
    i_current = i_current + 1

    if i % n_plot_save == 0
        plot_save(l, l_test, p_list, i)
    end
end

plot_save(l, l_test, p_list, total_train_iters)
train_elapsed = time() - train_start_time
println("training elapsed seconds: ", train_elapsed)
println("average iterations per second: ", total_train_iters / train_elapsed)
