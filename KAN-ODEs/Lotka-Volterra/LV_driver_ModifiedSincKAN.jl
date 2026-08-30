using Random, Lux, LinearAlgebra
using NNlib, ConcreteStructs, WeightInitializers, ChainRulesCore
using ComponentArrays
using BenchmarkTools
using OrdinaryDiffEq, Plots, DiffEqFlux, ForwardDiff
using Flux: Adam, mae, update!
using Flux
using MAT
using Plots
using ProgressBars
using Zygote: gradient as Zgrad

ENV["GKSwstype"] = "100"

env_int(name, default) = haskey(ENV, name) ? parse(Int, ENV[name]) : default
env_float32(name, default) = haskey(ENV, name) ? Float32(parse(Float64, ENV[name])) : Float32(default)
env_float64(name, default) = haskey(ENV, name) ? parse(Float64, ENV[name]) : default

# Directories
dir = @__DIR__
dir = dir * "/"
cd(dir)
run_tag = get(ENV, "RUN_TAG", "modifiedsinckan")
fname = "LV_" * run_tag
add_path = "results_" * run_tag * "/"
figpath = dir * add_path * "figs"
ckptpath = dir * add_path * "checkpoints"
mkpath(figpath)
mkpath(figpath * "/training")
mkpath(ckptpath)

# Load local KAN/SincKAN layers.
include("src/KolmogorovArnold.jl")
using .KolmogorovArnold

# Lotka-Volterra ground-truth dynamics.
function lotka!(du, u, p, t)
    alpha, beta, gamma, delta = p
    du[1] = alpha * u[1] - beta * u[2] * u[1]
    du[2] = gamma * u[1] * u[2] - delta * u[2]
end

# Data generation parameters.
timestep = 0.1
n_plot_save = 1000
rng = Random.default_rng()
Random.seed!(rng, 0)
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

# ModifiedSincKAN-ODE architecture.
# This keeps the NeuralODE interface identical to the original KAN-ODE driver:
# vector field f_theta: R^2 -> R^2.
num_layers = env_int("NUM_LAYERS", 2)
layer_width = env_int("LAYER_WIDTH", 10)
u_degree = env_int("U_DEGREE", 12)
u_len_h = env_int("U_LEN_H", 1)
v_degree = env_int("V_DEGREE", 24)
v_len_h = env_int("V_LEN_H", 1)
alpha = env_float32("ALPHA", 0.09)
normalizer = tanh_fast

println(
    "ModifiedSincKAN config: run_tag=", run_tag,
    " layer_width=", layer_width,
    " u_degree=", u_degree,
    " u_len_h=", u_len_h,
    " v_degree=", v_degree,
    " v_len_h=", v_len_h,
    " alpha=", alpha,
)

model = Lux.Chain(
    ModifiedSincKANBlock(
        2, layer_width;
        u_degree, u_len_h, v_degree, v_len_h,
        alpha, normalizer,
    ),
    ModifiedSincKANBlock(
        layer_width, 2;
        u_degree, u_len_h, v_degree, v_len_h,
        alpha, normalizer,
    ),
)

pM, stM = Lux.setup(rng, model)
pM_data = getdata(ComponentArray(pM))
pM_axis = getaxes(ComponentArray(pM))
p = (deepcopy(pM_data)) ./ 1e5
println("parameter size: ", length(p))

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

lr = env_float64("LR", 5e-4)
opt = Flux.Adam(lr)
N_iter = env_float64("N_ITER", 1e5)
println("training config: lr=", lr, " N_iter=", N_iter)
i_current = 1
l = []
l_test = []
p_list = []

function plot_save(l, l_test, p_list, epoch)
    plt = Plots.plot(l, yaxis = :log, label = "train", dpi = 600)
    plot!(l_test, yaxis = :log, label = "test")
    xlabel!("Epoch")
    ylabel!("Loss")
    png(plt, string(figpath, "/loss.png"))
    print("minimum train loss: ")
    print(minimum(l))
    print("          minimum test loss: ")
    print(minimum(l_test))

    p_curr = p_list[end]
    train_node_full = NeuralODE(model, tspan, Tsit5(), saveat = timestep)
    pred_sol = train_node_full(u0, ComponentArray(p_curr, pM_axis), stM)[1]
    pred_sol_true = solve(
        ODEProblem(lotka!, u0, tspan, p_true),
        Tsit5(),
        abstol = 1e-12,
        reltol = 1e-12,
        saveat = timestep,
    )
    plt = scatter(pred_sol_true, alpha = 0.75)
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
    write(file, "kan_pred_t", pred_sol.t)
    write(file, "kan_pred_u1", reduce(hcat, pred_sol.u)'[:, 1])
    write(file, "kan_pred_u2", reduce(hcat, pred_sol.u)'[:, 2])
    write(file, "size_KAN", [num_layers, layer_width, u_degree, u_len_h, v_degree, v_len_h])
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

train_elapsed = time() - train_start_time
println("training elapsed seconds: ", train_elapsed)
println("average iterations per second: ", total_train_iters / train_elapsed)
