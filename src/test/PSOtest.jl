include("../finders/finderPSO.jl")
using .PSOModule: finderPSO

# Length parameters
trainLen = 7*8640 #6 values/min * 60 min/hour * 24 hour/day = 8640 values/day
testLen = 360
initLen = 1*8640

# Reservoir parameters
inSize = 3
outSize = 1
resSize = 50
density = 0.1

# PSO parameters
population = 500
selfTrust = 1.8
neighbourTrust = 1.5
inertia = 0.8
maxIterations = 20

# alpha,		beta,		rho,				in_s
# leaking, 	reg coef, 	spectral radius, 	input scaling
lowerBounds = 0.001, 1*10^(-8), 0.01, 0.01
upperBounds = 0.99, 1*10^(-4), 2, 1

variedBounds = false

# Logging parameters
log = false

# Run the main function
finderPSO(	
    "data10secs_with_timestamps_random_days.csv", 
    trainLen, testLen, initLen, 
    inSize, outSize, resSize, density,
    population, selfTrust, neighbourTrust, inertia, maxIterations,
    lowerBounds, upperBounds,variedBounds,
    log
    )