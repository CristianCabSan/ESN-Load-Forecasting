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

#======#
# TODO #
#======#

Random.seed!(rand(1:1000000))

#= # PSO parameters
Population = 20
selfTrust = 1.8
neighbourTrust = 1.5
inertia = 0.8 =#

#alpha,		beta,		rho,				in_s
#leaking, 	reg coef, 	spectral radius, 	input scaling
lower_parameters = 0.001, 1*10^(-8), 0.01, 0.01
upper_parameters = 0.99, 1*10^(-4), 2, 1

low_alpha, low_beta, low_rho, low_in_s = lower_parameters
upper_alpha, upper_beta, upper_rho, upper_in_s = upper_parameters
bounds = [low_alpha low_beta low_rho low_in_s; upper_alpha upper_beta upper_rho upper_in_s]

custom_ga = GA(;
    N = 5,
    p_mutation  = 1e-5,
    p_crossover = 0.5,
    initializer = RandomPermutation(N=100),
    selection   = TournamentSelection(),
    crossover   = SBX(;bounds),
    mutation    = SlightMutation(),
    environmental_selection = ElitistReplacement()
)

# load the data
trainLen = 10000
testLen = 600
initLen = 1200
pre_data = readdlm("data.txt")
data = pre_data ./ 10

# generate the ESN reservoir
inSize = outSize = 1
resSize = 1000
density = 0.1
errores = Dict()
randomSeed = 42

function fitness(hyperparameters)
	#leaking, reg coef, spectral radius, input scaling
	alpha, beta, rho, in_s = hyperparameters
	Random.seed!(randomSeed)
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
		#u = data[trainLen+t+1]
	end

	# compute MSE for the first errorLen time steps
	errorLen = testLen
	mse = sum( abs2.( data[trainLen+2:trainLen+errorLen+1] .- 
		Y[1,1:errorLen] ) ) / errorLen
	errores[mse] = hyperparameters

    print("MSE: $mse")
	return mse
end





function main()
	# Start a new run, tracking hyperparameters in config
	#= lg = WandbLogger(project = "PSO-ESN",
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
	) =#

	function custom_logger(information)
		# Get the current best solution
		println("$information")	
		#= println("minimum: $(information.best_sol.f)")
		println("hyperparameters: $(information.best_sol.x)")
		hyperparams_dict = Dict(
		"alpha" => information.best_sol.x[1],
		"beta" => information.best_sol.x[2],
		"rho" => information.best_sol.x[3],
		"in_s" => information.best_sol.x[4]
		) =#
		
		#Wandb.log(lg, Dict("minError" => information.best_sol.f, "hyperparameters" => hyperparams_dict))
	end

	optimize(fitness, bounds, custom_ga; logger=custom_logger)
	#close(lg)
end

while(true)
	main()
end