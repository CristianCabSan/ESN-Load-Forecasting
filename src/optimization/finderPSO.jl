module PSOModule
	using Random
	using LinearAlgebra
	using SparseArrays
	using Distributions
	using Metaheuristics
	using Plots
	using Wandb
	using CSV
	using Dates
	using DataFrames

	export finderPSO

	# Parameters for the PSO stop when the minimum doesn't change
	global counter = 5
	global maxCounter = 5 # Maximum number of iterarions without change before stopping
	global lastMinimum = 0

	function load_data_timestamps(fileName)
		resources_dir = joinpath(@__DIR__, "..", "..", "resources")
		dataPath = joinpath(resources_dir, fileName)
		data = CSV.read(dataPath, DataFrame)

		values = data[:, 2]
		timestampYear = data[:, 9]
		timestampDay = data[:, 10]

		return values, timestampYear, timestampDay
	end

	function load_data(fileName)
		resources_dir = joinpath(@__DIR__, "..", "..", "resources")
		dataPath = joinpath(resources_dir, fileName)
		data = CSV.read(dataPath, DataFrame)

		values = data[:, 2]

		return values
	end

	function set_PSO(population, selfTrust, neighbourTrust, inertia, maxIterations)
		global custom_pso = PSO(;
			N  = population,
			C1 = selfTrust,	
			C2 = neighbourTrust,
			ω  = inertia,	
			options = Options(iterations = maxIterations)
		)
	end

	function fitness_timestamps(hyperparameters, values, timestampYear, timestampDay, inSize, outSize, resSize, density, trainLen, testLen, initLen, randomSeed, errors)

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

		errors[mse] = hyperparameters
		#= global p2 = plot(values[trainLen:trainLen + testLen + 2], c = RGB(0, 0.75, 0), label = "Target signal", reuse = false)
		plot!(transpose(Y), c = :blue, label = "Free-running predicted signal")
		title!(p2, "Target and generated signals with timestamps \n MSE = $(mse)") =#
		

		return mse
	end

	function fitness(hyperparameters, values, inSize, outSize, resSize, density, trainLen, testLen, initLen, randomSeed, errors)

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
			u = values[t]
			x = (1 - alpha) .* x .+ alpha .* tanh.(Win * [1; u] .+ W * x)
			if t > initLen
				X[:, t - initLen] = [1; u; x]
			end
		end

		# Train the output by ridge regression using Julia backslash solver
		Wout = transpose((X * transpose(X) + beta * I) \ (X * transpose(Yt)))

		# Run the trained ESN in a generative mode
		global Y = zeros(outSize, testLen)
		u = values[trainLen+1]
		for t = 1:testLen
			x = (1 - alpha) .* x .+ alpha .* tanh.(Win * [1; u] .+ W * x)
			y = Wout * [1; u; x]
			global Y[:, t] = y
			# Generative mode: use the predicted output as input for the next step
			u = values[trainLen+t+1]
		end

		# Compute MSE for the first errorLen time steps
		errorLen = testLen
		global mse = sum(abs2.(values[trainLen + 2:trainLen + errorLen + 1] .- Y[1, 1:errorLen])) / errorLen

		errors[mse] = hyperparameters
		#= global p2 = plot(values[trainLen:trainLen + testLen + 2], c = RGB(0, 0.75, 0), label = "Target signal", reuse = false)
		plot!(transpose(Y), c = :blue, label = "Free-running predicted signal")
		title!(p2, "Target and generated signals with timestamps \n MSE = $(mse)") =#
		

		return mse
	end

	function custom_logger(information)
		

		global then
		# Get the current best solution
		println("\n$information")	
		println("minimum: $(information.best_sol.f)")
		println("hyperparameters: $(information.best_sol.x)")
		println("Current date and time: ", now())
		println("Time elapsed ", round((now() - then).value / (1000 * 60), digits=2), " minutes")
		then = now()

		hyperparams_dict = Dict(
		"alpha" => information.best_sol.x[1],
		"beta" => information.best_sol.x[2],
		"rho" => information.best_sol.x[3],
		"in_s" => information.best_sol.x[4]
		)
		
		if log
			# Logs the information on Wandb
			Wandb.log(lg, Dict("minError" => information.best_sol.f, "hyperparameters" => hyperparams_dict))
		end

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

	function finderPSO(
		fileName="data10secs_with_timestamps_random_days.csv", 
		trainLen=2*8640, testLen=1*8640, initLen=1*8640, 
		inSize=3, outSize=1, resSize=1000, density=0.1, 
		population=30, selfTrust=1.8, neighbourTrust=1.5, inertia=0.8, maxIterations=30,
		lowerBounds=[0.001, 1*10^(-8), 0.01, 0.01], upperBounds=[0.99, 1*10^(-4), 2, 1], variedBounds=false,
		log_ = false
		)

		println("Current date and time: ", now())
		global then = now()
		global log = log_

		# Sets random seed, later seed is fixed
		Random.seed!(rand(1:1000000))

		# Sets the PSO parameters
		set_PSO(population, selfTrust, neighbourTrust, inertia, maxIterations)

		# Generate the error dictionary
		errors = Dict()

		# Adds a variance if variedBounds parameter is set
		if variedBounds
			variation = rand(0.1:0.0001:2.0) #Adds a random variation for each iteration
			lowerBounds = variation .* lowerBounds
			upperBounds = variation .* upperBounds 
		end

		# Fixed seed
		global randomSeed = 42

		# Logs the run if the log parameter is set
		if log
			global lg = WandbLogger(
				project = "PSO-ESN",
				name = "Ajustes ESN-$(now())",
				config = Dict(
					"Population" => population,
					"selfTrust" => selfTrust,
					"neighbourTrust" => neighbourTrust,
					"inertia" => inertia,
					"lowerBounds" => lowerBounds,
					"upperBounds" => upperBounds,

					"trainLen" => trainLen,
					"testLen" => testLen,
					"initLen" => initLen,

					"resSize" => resSize,
					"density" => density,
					"randomSeed" => randomSeed
				)
			) 
		end

		# leaking, 	reg coef, 	spectral radius, 	input scaling
		low_alpha, low_beta, low_rho, low_in_s = lowerBounds
		upper_alpha, upper_beta, upper_rho, upper_in_s = upperBounds

		# Load the data
		if inSize == 1
			values = load_data(fileName)
			fitness_closure = hyperparameters -> fitness(hyperparameters, values, inSize, outSize, resSize, density, trainLen, testLen, initLen, randomSeed, errors)
		else
			values, timestampYear, timestampDay = load_data_timestamps(fileName)
			fitness_closure = hyperparameters -> fitness_timestamps(hyperparameters, values, timestampYear, timestampDay, inSize, outSize, resSize, density, trainLen, testLen, initLen, randomSeed, errors)
		end
		
		optimize(fitness_closure, [low_alpha low_beta low_rho low_in_s; upper_alpha upper_beta upper_rho upper_in_s], custom_pso; logger=custom_logger)

		# Closes the run
		if log 
			# Ensures enough time to close the log correctly
			close(lg)
			sleep(30)
		end
	end

end