using CSV
using DataFrames
using Dates
using Interpolations

#= 
    Takes a .csv file and generates intermediate values through interpolation 
    to make a new gap of "inerpolation_interval" seconds.
=#

# Set interpolation interval (in seconds)
interpolation_interval = 1  

# Load the data
data_name = "data1min.csv"
resources_dir = joinpath(@__DIR__, "..", "..", "resources")
data_path = joinpath(resources_dir, data_name)
data = CSV.read(data_path, DataFrame)

# Convert DateTime column to DateTime type
data.DateTime = DateTime.(data.DateTime)

# Convert DateTime to numeric values (seconds since start_time)
start_time = data.DateTime[1]
data_numeric_time = Float64.((data.DateTime .- start_time) ./ Millisecond(1)) ./ 1000.0

# Generate the new DateTime range
new_times = collect(start_time:Second(interpolation_interval):data.DateTime[end])
new_numeric_times = Float64.((new_times .- start_time) ./ Millisecond(1)) ./ 1000.0

# Interpolate each column except DateTime
cols_to_interpolate = names(data)[2:end]
interp_data = DataFrame(DateTime = new_times)

for col in cols_to_interpolate
    f = LinearInterpolation(data_numeric_time, data[!, col], extrapolation_bc=Line())
    interp_data[!, col] = f.(new_numeric_times)
end

# Save the updated DataFrame to a new CSV file
CSV.write(joinpath(resources_dir, "data$(interpolation_interval)sec.csv"), interp_data)
