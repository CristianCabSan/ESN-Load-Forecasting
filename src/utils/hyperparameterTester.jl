#=
A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data 
in "plain" Julia.
from https://mantas.info/code/simple_esn/
(c) 2020 Mantas Lukoševičius
Distributed under MIT license https://opensource.org/licenses/MIT
=#

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

# generate the ESN reservoir
inSize = 3
outSize = 1
resSize = 1000
density = 0.1
errores = Dict()
randomSeed = 42

#hyperparameters
alpha = 1.61845   #Leaking Rate
beta = 0.00016348   #Regularization Coef
in_s = 3.2696     #Input Scaling
rho =  0.744364   #Spectral Radius
hyperparameters = alpha, beta, in_s, rho

# plot some of it
function fitness(hyperparameters, values)
	inSizeCustom = 1
	#leaking, reg coef, spectral radius, input scaling
	alpha, beta, rho, in_s = hyperparameters
	Random.seed!(randomSeed)
	Win = (rand(resSize, 1+inSizeCustom) .- 0.5) .* 1
	W = SparseArrays.sprand(resSize, resSize, density, x-> rand(Uniform(-in_s,in_s), x ))
	W = Array(W)

	# normalizing and setting spectral radius
	#print("Computing spectral radius...")
	rhoW = maximum(abs.(eigvals(W)))
	#println("done.")
	W .*= (rho / rhoW)

	# allocated memory for the design (collected states) matrix
	X = zeros(1+inSizeCustom+resSize, trainLen-initLen)
	# set the corresponding target matrix directly
	Yt = transpose(values[initLen+2:trainLen+1]) 

	# run the reservoir with the data and collect X
	x = zeros(resSize, 1)
	for t = 1:trainLen
		# Reemplazar 1,1 por los datos de referencias
		u = values[t]
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
	global Y = zeros(outSize, testLen)
	u = values[trainLen+1]
	for t = 1:testLen 
		x = (1-alpha).*x .+ alpha.*tanh.( Win*[1;u] .+ W*x )
		y = Wout*[1;u;x]
		#y2 = data-ReferenceYear
		#y3 = data-ReferenceDay
		global Y[:,t] = y
		# generative mode:
		u = y
		# this would be a predictive mode:
		#u = data[trainLen+t+1]
	end

	# compute MSE for the first errorLen time steps
	errorLen = testLen
	global mse = sum( abs2.( values[trainLen+2:trainLen+errorLen+1] .- 
		Y[1,1:errorLen] ) ) / errorLen
	
	errores[mse] = hyperparameters
	global p1 = plot(values[trainLen:trainLen+testLen+2], c = RGB(0,0.75,0), label = "Target signal", reuse = false)
	plot!(transpose(Y), c = :blue, label = "Free-running predicted signal")
	title!(p1, "Target and generated signals without timestamps \n MSE = $(mse)")
	
	return mse
end

function fitnessTimestamps(hyperparameters, values, timestampYear, timestampDay)
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

#println("trainLen: $trainLen")
println("With timestamps: $(fitnessTimestamps(hyperparameters, values, timestampYear, timestampDay))")
println("Without timestamps: $(fitness(hyperparameters, values))")


# display all 4 plots
plot(p1,p2, size=(1200,800))
#savefig("Documentacion/Starts in day $((paddingLen-397)÷1440)/$testLen testLen/interpolationComparision($(trainLen ÷ 1440) days)($testLen testLen).png")

