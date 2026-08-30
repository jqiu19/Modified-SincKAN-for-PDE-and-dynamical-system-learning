module KolmogorovArnold

using Random
using LinearAlgebra

using NNlib
using LuxCore
using WeightInitializers
using ConcreteStructs

using ChainRulesCore
const CRC = ChainRulesCore

include("utils.jl")
export rbf, rswaf, iqf

include("kdense.jl")
export KDense

include("modified_sinc_kan.jl")
export SincDense, ModifiedSincKANBlock

# include("explicit")
# export GDense

end # module
