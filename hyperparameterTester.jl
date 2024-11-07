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

# load the data
trainLen = 90*1440 #1440 minutes = 1 day
testLen = 150
initLen = 1200
paddingLen = (365*1440)+397 #To control the day it starts feeding data from (397 so it starts at 00:00)
raw_data1 = readdlm("data.txt")
pre_data1 = raw_data1[paddingLen:end]

raw_data2 = readdlm("formattedData.txt", ':')
pre_data2 = raw_data2[paddingLen:end, 3]

data1 = pre_data1 ./ 10
data2 = pre_data2 ./ 10

p1 = plot(data1[1:trainLen], leg = false, title = "A sample of data", reuse=false)
p3 = plot(data2[1:trainLen], leg = false, title = "A sample of data", reuse=false)

# generate the ESN reservoir
inSize = outSize = 1
resSize = 1000
density = 0.1
errores = Dict()
randomSeed = 42

#hyperparameters
alpha = 0.842538   #Leaking Rate
beta = 5.91997e-5   #Regularization Coef
in_s = 1.35975      #Input Scaling
rho = 0.449798     #Spectral Radius
hyperparameters = alpha, beta, in_s, rho

# plot some of it
function fitness(hyperparameters, data)
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
	global Y = zeros(outSize, testLen)
	u = data[trainLen+1]
	for t = 1:testLen 
		x = (1-alpha).*x .+ alpha.*tanh.( Win*[1;u] .+ W*x )
		y = Wout*[1;u;x]
		global Y[:,t] = y
		# generative mode:
		u = y
		# this would be a predictive mode:
		#u = data[trainLen+t+1]
	end

	# compute MSE for the first errorLen time steps
	errorLen = testLen
	global mse = sum( abs2.( data[trainLen+2:trainLen+errorLen+1] .- 
		Y[1,1:errorLen] ) ) / errorLen
	
	errores[mse] = hyperparameters
	global p2 = plot(data[trainLen:trainLen+testLen+2], c = RGB(0,0.75,0), label = "Target signal", reuse = false)
	plot!(transpose(Y), c = :blue, label = "Free-running predicted signal")
	title!(p2, "Target and generated signals without interpolation \n MSE = $(mse)")
	
	return mse
end

function fitness2(hyperparameters)
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
	Yt = transpose(data2[initLen+2:trainLen+1]) 

	# run the reservoir with the data and collect X
	x = zeros(resSize, 1)
	for t = 1:trainLen
		u = data2[t]
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
	u = data2[trainLen+1]
	for t = 1:testLen 
		x = (1-alpha).*x .+ alpha.*tanh.( Win*[1;u] .+ W*x )
		y = Wout*[1;u;x]
		global Y[:,t] = y
		# generative mode:
		u = y
		# this would be a predictive mode:
		#u = data2[trainLen+t+1]
	end

	# compute MSE for the first errorLen time steps
	errorLen = testLen
	global mse = sum( abs2.( data2[trainLen+2:trainLen+errorLen+1] .- 
		Y[1,1:errorLen] ) ) / errorLen
	
	errores[mse] = hyperparameters
	global p4 = plot(data2[trainLen:trainLen+testLen+2], c = RGB(0,0.75,0), label = "Target signal", reuse = false)
	plot!(transpose(Y), c = :blue, label = "Free-running predicted signal")
	title!(p4, "Target and generated signals with interpolation \n MSE = $(mse)")
	
	return mse
end
println("trainLen: $trainLen")
println("Without interpolation: $(fitness(hyperparameters, data1))")
println("With interpolation: $(fitness2(hyperparameters))")

# display all 4 plots
plot(p1,p2,p3,p4, size=(1200,800), ylim = (0,1))
savefig("Documentacion/Starts in day $((paddingLen-397)÷1440)/$testLen testLen/interpolationComparision($(trainLen ÷ 1440) days)($testLen testLen).png")

