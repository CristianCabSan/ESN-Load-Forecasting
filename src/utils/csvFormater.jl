using CSV
using DataFrames
using Dates

#= 
    Takes a .txt file with at least Date and Time columns and outputs a .csv file. 
    The Date and Time columns are merged into a single DateTime column in the resulting file.
=#

# Get the data
data_name = "household_power_consumption.txt"
resources_dir = joinpath(@__DIR__, "..", "..", "resources")
data_path = joinpath(resources_dir, data_name)
data = CSV.read(data_path, DataFrame; delim = ";")
df = DateFormat("dd/mm/yyyy")
dates = Date.(data.Date, df)
times = data.Time

# Creates and adds the new column as the first one
dateTimes = dates .+ times
data[!, :DateTime] = dateTimes
data = select(data, :DateTime, Not(:DateTime))

# Remove the original Date and Time columns
select!(data, Not([:Date, :Time]))

# Save the updated DataFrame to a new CSV file
CSV.write("$resources_dir/data.csv", data)
