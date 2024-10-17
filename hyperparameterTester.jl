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
trainLen = 2000
testLen = 600
initLen = 200
data = readdlm("data.txt")

p1 = plot(data[1:1000], leg = false, title = "A sample of data", reuse=false)
plot(p1, size=(1200,800))

# generate the ESN reservoir
inSize = outSize = 1
resSize = 1000
density = 0.1
errores = Dict()
randomSeed = 42
Random.seed!(randomSeed)

#hyperparameters
alpha = 0.18582897495305825     #Leaking Rate
beta = 0.00002141456204105258   #Regularization Coef
in_s = 1.5268437152991396       #Input Scaling
rho = 0.08658646768396282       #Spectral Radius
hyperparameters = alpha, beta, in_s, rho

# plot some of it


Random.seed!(42)
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
		#u = data[trainLen+t+1]
	end

	# compute MSE for the first errorLen time steps
	errorLen = testLen
	mse = sum( abs2.( data[trainLen+2:trainLen+errorLen+1] .- 
		Y[1,1:errorLen] ) ) / errorLen
	
	errores[mse] = hyperparameters
	#p2 = plot(data[trainLen+2:trainLen+testLen+1], c = RGB(0,0.75,0), label = "Target signal", reuse = false)
	plot!(transpose(Y), c = :blue, label = "Free-running predicted signal")
	title!("Target and generated signals y(n) starting at n=0")
	
	return mse
end

print(fitness(hyperparameters))

# plot some signals
#=
p2 = plot(data[trainLen+2:trainLen+testLen+1], c = RGB(0,0.75,0), label = "Target signal", reuse = false)
plot!(transpose(Y), c = :blue, label = "Free-running predicted signal")
title!("Target and generated signals y(n) starting at n=0")

p3 = plot( transpose(X[1:20,1:200]), leg = false )
title!("Some reservoir activations x(n)")

p4 = bar(transpose(Wout), leg = false)
title!("Output weights Wout")

# display all 4 plots
plot(p1, p2, p3, p4, size=(1200,800))
=#
