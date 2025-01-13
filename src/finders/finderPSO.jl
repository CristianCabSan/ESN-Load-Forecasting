import Random
using LinearAlgebra
import SparseArrays
using Distributions
using Metaheuristics
using Plots
using Wandb
using Dates
using Logging
using CSV
using DataFrames

Random.seed!(rand(1:1000000))

# Load the data
data_name = "data10secs_with_timestamps.csv"
resources_dir = joinpath(@__DIR__, "..", "..", "resources")
data_path = joinpath(resources_dir, data_name)
data = CSV.read(data_path, DataFrame)

values = data[:, 2]
timestampYear = data[:, 9]
timestampDay = data[:, 10]

trainLen = 10*1440
testLen = 600
initLen = 1200

# Generate the ESN reservoir
inSize = 3
outSize = 1
resSize = 1000
density = 0.1
errores = Dict()
randomSeed = 42

maxCounter = 20

function fitness(hyperparameters)
    # Leaking, regularization coefficient, spectral radius, input scaling
    alpha, beta, rho, in_s = hyperparameters
    Random.seed!(randomSeed)
    
    # Update input size to account for three inputs
    Win = (rand(resSize, 1 + inSize) .- 0.5) .* 1
    W = SparseArrays.sprand(resSize, resSize, density, x -> rand(Uniform(-in_s, in_s), x))
    W = Array(W)

    # Normalizing and setting spectral radius
    rhoW = maximum(abs.(eigvals(W)))
    W .*= (rho / rhoW)

    # Allocated memory for the design (collected states) matrix
    X = zeros(1 + inSize + resSize, trainLen - initLen)
    # Set the corresponding target matrix directly
    Yt = transpose(values[initLen + 2:trainLen + 1])

    # Run the reservoir with the data and collect X
    x = zeros(resSize, 1)
    for t = 1:trainLen
        # Use all three inputs: data, timestampYear, and timestampDay
        u = [values[t]; timestampYear[t]; timestampDay[t]]
        x = (1 - alpha) .* x .+ alpha .* tanh.(Win * [1; u] .+ W * x)
        if t > initLen
            X[:, t - initLen] = [1; u; x]
        end
    end

    # Train the output by ridge regression using Julia backslash solver
    Wout = transpose((X * transpose(X) + beta * I) \ (X * transpose(Yt)))

    # Run the trained ESN in a generative mode
    global Y = zeros(outSize, testLen)
    u = [values[trainLen + 1]; timestampYear[trainLen + 1]; timestampDay[trainLen + 1]]
    for t = 1:testLen
        x = (1 - alpha) .* x .+ alpha .* tanh.(Win * [1; u] .+ W * x)
        y = Wout * [1; u; x]
        global Y[:, t] = y
        # Generative mode: use the predicted output as input for the next step
        u = [y[1]; timestampYear[trainLen + t + 1]; timestampDay[trainLen + t + 1]]
    end

    # Compute MSE for the first errorLen time steps
    errorLen = testLen
    global mse = sum(abs2.(values[trainLen + 2:trainLen + errorLen + 1] .- Y[1, 1:errorLen])) / errorLen

    errores[mse] = hyperparameters
    global p2 = plot(values[trainLen:trainLen + testLen + 2], c = RGB(0, 0.75, 0), label = "Target signal", reuse = false)
    plot!(transpose(Y), c = :blue, label = "Free-running predicted signal")
    title!(p2, "Target and generated signals with timestamps \n MSE = $(mse)")

    return mse
end


function custom_logger(information)
	# Get the current best solution
	println("$information")	
	println("minimum: $(information.best_sol.f)")
	println("hyperparameters: $(information.best_sol.x)")
	hyperparams_dict = Dict(
	"alpha" => information.best_sol.x[1],
	"beta" => information.best_sol.x[2],
	"rho" => information.best_sol.x[3],
	"in_s" => information.best_sol.x[4]
	)
	
	# Logs the information on Wandb
	#Wandb.log(lg, Dict("minError" => information.best_sol.f, "hyperparameters" => hyperparams_dict))

	# If minimun doesnt change in counter iterations the PSO is forcefully stopped
	current_minimum = information.best_sol.f
	# Check if this is the first run or if the minimum has changed
	if lastMinimum != 0 && lastMinimum == current_minimum
		# Decrement the counter if the minimum hasn't improved
		global counter -= 1
		if counter == 0
			println("########################## STOP ######################################")
			custom_pso.status.stop = true
		end
	else
		# Reset the counter if the minimum has changed
		global counter = maxCounter
	end
	
	# Update the last recorded minimum for the next iteration
	global lastMinimum = current_minimum
end

function main()
	# Execution control parameters
	global counter = maxCounter
    global lastMinimum = 0

	# alpha,		beta,		rho,				in_s
	# leaking, 	reg coef, 	spectral radius, 	input scaling
	variation = rand(0.1:0.0001:2.0) #Adds a random variation for each iteration
	lower_base_parameters = 0.001, 1*10^(-8), 0.01, 0.01
	upper_base_parameters = 0.99, 1*10^(-4), 2, 1

	lower_parameters = variation .* lower_base_parameters
	upper_parameters = variation .* upper_base_parameters
	# PSO parameters
	Population = 30
	selfTrust = 1.8
	neighbourTrust = 1.5
	inertia = 0.8

	global custom_pso = PSO(;
		N  = Population,
		C1 = selfTrust,	
		C2 = neighbourTrust,
		ω  = inertia,	
		options = Options(iterations = 150)
	)

	# # Start a new run, tracking hyperparameters in config
	# global lg = WandbLogger(project = "PSO-ESN",
	# name = "Ajustes ESN-$(now())",
	# config = Dict(
	# 	"Population" => Population,
	# 	"selfTrust" => selfTrust,
	# 	"neighbourTrust" => neighbourTrust,
	# 	"inertia" => inertia,
	# 	"lower_parameters" => lower_parameters,
	# 	"upper_parameters" => upper_parameters,

	# 	"trainLen" => trainLen,
	# 	"testLen" => testLen,
	# 	"initLen" => initLen,

	# 	"resSize" => resSize,
	# 	"density" => density,
	# 	"randomSeed" => randomSeed
	# 	)
	# )

	low_alpha, low_beta, low_rho, low_in_s = lower_parameters
	upper_alpha, upper_beta, upper_rho, upper_in_s = upper_parameters

	optimize(fitness, [low_alpha low_beta low_rho low_in_s; upper_alpha upper_beta upper_rho upper_in_s], custom_pso; logger=custom_logger)
	# close(lg)
	sleep(30) # Ensures log is closed before restarting
end

while(true)
	main()
end
