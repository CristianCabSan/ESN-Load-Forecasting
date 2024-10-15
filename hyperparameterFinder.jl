#=
A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data 
in "plain" Julia.
from https://mantas.info/code/simple_esn/
(c) 2020 Mantas Lukoševičius
Distributed under MIT license https://opensource.org/licenses/MIT
=#

using DelimitedFiles
import Random
using LinearAlgebra
import SparseArrays
using Distributions
using Metaheuristics
using Plots
using Wandb
using Dates
using Logging

# PSO parameters
Population = 20
selfTrust = 2.0
neighbourTrust = 2.0
inertia = 0.8

#alpha,		beta,		rho,				in_s
#leaking, 	reg coef, 	spectral radius, 	input scaling
lower_parameters = 0.01, 1*10^(-8), 0.01, 0.1
upper_parameters = 0.3, 1*10^(-4), 1, 5

PSO(;
    N  = Population,
    C1 = selfTrust,	
    C2 = neighbourTrust,
    ω  = inertia,	
    information = Information(),
    options = Options()
)

# load the data
trainLen = 2000
testLen = 600
initLen = 200
data = readdlm("data.txt")

# generate the ESN reservoir
inSize = outSize = 1
resSize = 1000
density = 0.1
errores = Dict()
randomSeed = 42
Random.seed!(randomSeed)


# Start a new run, tracking hyperparameters in config

lg = WandbLogger(project = "PSO-ESN",
				name = "Ajustes ESN-$(now())",
				config = Dict(
					"Population" => Population,
					"selfTrust" => selfTrust,
					"neighbourTrust" => neighbourTrust,
					"inertia" => inertia,
					"lower_parameters" => lower_parameters,
					"upper_parameters" => upper_parameters,

					"trainLen" => trainLen,
					"testLen" => testLen,
					"initLen" => initLen,

					"resSize" => resSize,
					"density" => density,
					"randomSeed" => randomSeed
					)
)
global_logger(lg)

#global i = 0
#global j = 0

function fitness(hyperparameters)
	#leaking, reg coef, spectral radius, input scaling
	alpha, beta, rho, in_s = hyperparameters
	
	Win = (rand(resSize, 1+inSize) .- 0.5) .* 1
	W = SparseArrays.sprand(resSize, resSize, density, x-> rand(Uniform(-in_s,in_s), x ))
	W = Array(W)

	# normalizing and setting spectral radius
	#print("Computing spectral radius...")
	rhoW = maximum(abs.(eigvals(W)))
	#println("done.")
	W .*= (rho / rhoW)

	# allocated memory for the design (collected states) matrix
	X = zeros(1+inSize+resSize, trainLen-initLen)
	# set the corresponding target matrix directly
	Yt = transpose(data[initLen+2:trainLen+1]) 

	# run the reservoir with the data and collect X
	x = zeros(resSize, 1)
	for t = 1:trainLen
		u = data[t]
		x = (1-alpha).*x .+ alpha.*tanh.( Win*[1;u] .+ W*x ) 
		if t > initLen
			X[:,t-initLen] = [1;u;x]
		end
	end

	# train the output by ridge regression
	# using Julia backslash solver:
	Wout = transpose((X*transpose(X) + beta*I) \ (X*transpose(Yt)))

	# run the trained ESN in a generative mode. no need to initialize here, 
	# because x is initialized with training data and we continue from there.
	Y = zeros(outSize, testLen)
	u = data[trainLen+1]
	for t = 1:testLen 
		x = (1-alpha).*x .+ alpha.*tanh.( Win*[1;u] .+ W*x )
		y = Wout*[1;u;x]
		Y[:,t] = y
		# generative mode:
		u = y
		# this would be a predictive mode:
		#global u = data[trainLen+t+1]
	end

	# compute MSE for the first errorLen time steps
	errorLen = testLen
	mse = sum( abs2.( data[trainLen+2:trainLen+errorLen+1] .- 
		Y[1,1:errorLen] ) ) / errorLen
	
	errores[mse] = hyperparameters
	
	#=
	println("Poblador $i de iteracion $j")
	if(i == (Population-1))	
		# Get the element with the smallest key
		best_hyperparameters, min_error,  = findmin(errores)

		# Display the result
		println("Minimum error from iteration $j: $min_error")
		println("Best hyperparameters from iteration $j: $best_hyperparameters")
		empty!(errores)
		global i = 0
		global j += 1
	else 
		global i += 1
	end
	=#

	return mse
end

function custom_logger(information)
    # Get the current best solution
    println("$information")	
	println("minimum: $(information.best_sol.f)")
	println("hyperparameters: $(information.best_sol.x[1])")
	hyperparams_dict = Dict(
    "alpha" => information.best_sol.x[1],
    "beta" => information.best_sol.x[2],
    "rho" => information.best_sol.x[3],
    "in_s" => information.best_sol.x[4]
	)
	
	Wandb.log(lg, Dict("minError" => information.best_sol.f, "hyperparameters" => hyperparams_dict))
	#wandb.log({'acc': 0.9, 'epoch': 3, 'batch': 117})
end


optimize(fitness, [0.01 1*10^(-8) 0.01 0.1; 0.5 1*10^(-4) 2 10], PSO(); logger=custom_logger)

#= 
# plot some signals
p2 = plot(data[trainLen+2:trainLen+testLen+1], c = RGB(0,0.75,0), label = "Target signal", reuse = false)
plot!(transpose(Y), c = :blue, label = "Free-running predicted signal")
title!("Target and generated signals y(n) starting at n=0")

p3 = plot(transpose(X[1:20,1:200]), leg = false)
title!("Some reservoir activations x(n)")

p4 = bar(transpose(Wout), leg = false)
title!("Output weights Wout")

# display all 4 plots
plot(p1, p2, p3, p4, size=(1200,800)) 
=#
