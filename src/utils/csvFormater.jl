using CSV
using DataFrames

data_name = "household_power_consumption.txt"

resources_dir = joinpath(@__DIR__, "..", "..", "resources")
data_path = joinpath(resources_dir, data_name)

data = CSV.read(data_path, DataFrame; delim = ";")

CSV.write("$resources_dir/data.csv", data)