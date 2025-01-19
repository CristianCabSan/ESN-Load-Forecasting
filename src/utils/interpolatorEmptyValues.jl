using Plots
using DataFrames
using CSV

#= 
    Takes a .csv file and interpolates missing values (marked with "?") on 
    all numeric columns (not DateTime) with the average of valid neighbours.
=#

# Get the data
data_name = "data.csv"
resources_dir = joinpath(@__DIR__, "..", "..", "resources")
data_path = joinpath(resources_dir, data_name)
data = CSV.read(data_path, DataFrame)

# Interpolate "?" values with the average of the valid previous and next values
function interpolate_values(column)
    # Convert only if the column contains strings
    if eltype(column) <: AbstractString
        column = Vector{Union{Float64, Missing}}(
            [value == "?" ? missing : parse(Float64, value) for value in column]
        )
    else
        column = Vector{Union{Float64, Missing}}(column)
    end

    # Replace missing values with the average of valid neighbors
    for i in 1:length(column)
        if ismissing(column[i])
            prev_value = find_previous_valid(column, i)
            next_value = find_next_valid(column, i)
            column[i] = (prev_value + next_value) / 2
        end
    end

    return Float64.(column)  # Convert to Float64
end

function find_previous_valid(column, idx)
    for i in reverse(1:idx-1)
        if !ismissing(column[i])
            return column[i]
        end
    end
    return NaN  # If no valid previous value is found
end

function find_next_valid(column, idx)
    for i in idx+1:length(column)
        if !ismissing(column[i])
            return column[i]
        end
    end
    return NaN  # If no valid next value is found
end

# Interpolate consumption values
for col in names(data)
    if col != "DateTime"
        data[!, col] = interpolate_values(data[!, col])
    end
end

# Save the updated DataFrame to a new CSV file
CSV.write("$resources_dir/data1min.csv", data)
