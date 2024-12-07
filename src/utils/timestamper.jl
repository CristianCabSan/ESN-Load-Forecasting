using CSV
using Dates
using DataFrames

#= 
    Takes a .csv file wiht a DateTime column and generates two new ones.
    Seconds_from_year_start: Represents the number of seconds elapsed since the start of the year 
    corresponding to the DateTime value in each row.
    
    Seconds_from_day_start: Represents the number of seconds elapsed since the start of the day 
    corresponding to the DateTime value in each row.
=#

# Load the data
data_name = "data10secs.csv"
resources_dir = joinpath(@__DIR__, "..", "..", "resources")
data_path = joinpath(resources_dir, data_name)
data = CSV.read(data_path, DataFrame)

# Generates and adds the new columns
data[!, :Miliseconds_from_year_start] = [
    Dates.value(DateTime(row.DateTime) - DateTime(year(DateTime(row.DateTime)), 1, 1))
    for row in eachrow(data)
]

data[!, :Miliseconds_from_day_start] = [
    Dates.value(DateTime(row.DateTime) - DateTime(year(DateTime(row.DateTime)), month(DateTime(row.DateTime)), day(DateTime(row.DateTime))))
    for row in eachrow(data)
]

# Save the updated DataFrame to a new CSV file
CSV.write(joinpath(resources_dir, "data10secs_with_timestamps.csv"), data)