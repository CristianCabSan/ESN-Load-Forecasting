import Random
using LinearAlgebra
import SparseArrays
using Distributions
using Metaheuristics
using Plots
using Dates
using CSV
using DataFrames

# Load the data
fileName = "data1secTimestampsRandomDays.csv"

trainLen = 50*86400 #6 values/min * 6 min/hour * 24 hour/day = 8640 values/day
testLen = 3600*12
initLen = 1*86400

# generate the ESN reservoir
inSize = 1
outSize = 1
resSize = 500
density = 0.1
errors = Dict()
randomSeed = 42



#hyperparameters
alpha = 0.06880643389985228 #Leaking Rate
beta = 0.18459163742831247   #Regularization Coef
rho = 2.7949644298276812   #Input Scaling
in_s =  0.45610339765370234   #Spectral Radius
hyperparameters = alpha, beta, in_s, rho

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
	global mape = mean(abs.((values[trainLen+2 : trainLen+errorLen+1] .- Y[1, 1:errorLen]) ./ values[trainLen+2 : trainLen+errorLen+1])) * 100

	errors[mse] = hyperparameters
	global p1 = plot(values[trainLen:trainLen + testLen + 2], c = RGB(0, 0.75, 0), label = "Target signal", reuse = false)
	plot!(transpose(Y), c = :blue, label = "Free-running predicted signal")
	title!(p1, "Target and generated signals \n MSE = $(mse)") 
	display(p1)

	return mse, mape
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
		u = y
	end

	# Compute MSE for the first errorLen time steps
	errorLen = testLen
	global mse = sum(abs2.(values[trainLen + 2:trainLen + errorLen + 1] .- Y[1, 1:errorLen])) / errorLen
	global mape = mean(abs.((values[trainLen+2 : trainLen+errorLen+1] .- Y[1, 1:errorLen]) ./ values[trainLen+2 : trainLen+errorLen+1])) * 100

	errors[mse] = hyperparameters
	global p1 = plot(values[trainLen:trainLen + testLen + 2], c = RGB(0, 0.75, 0), label = "Target signal", reuse = false)
	plot!(transpose(Y), c = :blue, label = "Free-running predicted signal")
	title!(p1, "Target and generated signals \n MSE = $(mse)")
	display(p1)

	return mse, mape
end

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

if inSize == 1
	values = load_data(fileName)
	fitness(hyperparameters, values, inSize, outSize, resSize, density, trainLen, testLen, initLen, randomSeed, errors)
else
	values, timestampYear, timestampDay = load_data_timestamps(fileName)
	fitness_timestamps(hyperparameters, values, timestampYear, timestampDay, inSize, outSize, resSize, density, trainLen, testLen, initLen, randomSeed, errors)
end



#savefig("Documentacion/Starts in day $((paddingLen-397)÷1440)/$testLen testLen/interpolationComparision($(trainLen ÷ 1440) days)($testLen testLen).png")


