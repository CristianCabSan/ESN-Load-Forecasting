include("../finders/finderPSO.jl")
using .PSOModule: finderPSO

# Length parameters
trainLen = 10*8640 #1 value/sec * 60 secs/min * 60 min/hour * 24 hour/day = 86400 values/day
testLen = 360 # 1 value/sec * 60 secs/min * 60 min/hour = 1 hour prediction
initLen = 1*600

# Reservoir parameters
inSize = 3
outSize = 1
resSize = 500
density = 0.1

# PSO parameters
population = 50
selfTrust = 1.8
neighbourTrust = 1.5
inertia = 0.8
maxIterations = 20

# alpha,		beta,		rho,				in_s
# leaking, 	reg coef, 	spectral radius, 	input scaling
lowerBounds = 0.01, 1*10^(-8), 0.1, 0.01
upperBounds = 1.0 , 1*10^(-1), 1.5, 1

# If variedBounds is true, the bounds are multiplied by a random number between 0.1 and 2.0
variedBounds = true

# Logging parameters
log = true

# Run the main function
finderPSO(	
    "data10secs_with_timestamps_random_days.csv", 
    trainLen, testLen, initLen, 
    inSize, outSize, resSize, density,
    population, selfTrust, neighbourTrust, inertia, maxIterations,
    lowerBounds, upperBounds,variedBounds,
    log
    )