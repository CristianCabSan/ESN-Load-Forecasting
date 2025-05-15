using DelimitedFiles
using CSV
using DataFrames

#= 
    Takes a directory with the REDD dataset and sums the two mains of each house.
    Returns a CSV file with the total power of the mains and the timestamp of each value.
=#

# Define the path to the resources directory
resources_dir = joinpath(@__DIR__, "..", "..", "resources")

# One iterarion for each house
for h in 1:6
    # Read both mains of house h
    house_path = joinpath(resources_dir, "redd", "low_freq","house_$(h)")
    data1 = readdlm(joinpath(house_path, "channel_1.dat"))
    data2 = readdlm(joinpath(house_path, "channel_2.dat"))

    # Verify that both files have the same length
    @assert size(data1, 1) == size(data2, 1) "Los archivos no tienen la misma longitud"

    # Aggregate the values of both mains
    suma_data = [(Int(data1[i, 1]), data1[i, 2] + data2[i, 2]) for i in eachindex(data1[:, 1])]

    # Convert to DataFrame
    df = DataFrame(timestamp = [x[1] for x in suma_data],
                   total_power = [x[2] for x in suma_data])

    # Saves as CSV
    destination_path = joinpath(resources_dir, "house$(h)_mains_total.csv")
    CSV.write(destination_path, df)
end