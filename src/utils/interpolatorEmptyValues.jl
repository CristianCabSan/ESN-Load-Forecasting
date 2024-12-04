using Plots
using DataFrames
using CSV

data_name = "data.csv"

resources_dir = joinpath(@__DIR__, "..", "..", "resources")
data_path = joinpath(resources_dir, data_name)

# Read the CSV file into a DataFrame
data = CSV.read(data_path, DataFrame)
dates = data.Date
times = data.Time
consumption = String.(data.Global_active_power)  # Convert to String for processing

# Interpolate "?" values with the average of the valid previous and next values
function interpolate_consumption(consumption)
    consumption_float = Vector{Union{Float64, Missing}}(
        [value == "?" ? missing : parse(Float64, value) for value in consumption]
    )

    # Replace missing values with the average of valid neighbors
    for i in 1:length(consumption_float)
        if ismissing(consumption_float[i])
            prev_value = find_previous_valid(consumption_float, i)
            next_value = find_next_valid(consumption_float, i)
            consumption_float[i] = (prev_value + next_value) / 2
        end
    end

    return Float64.(consumption_float)  # Convert to Float64
end

function find_previous_valid(consumption, idx)
    for i in reverse(1:idx-1)
        if !ismissing(consumption[i])
            return consumption[i]
        end
    end
    return NaN  # If no valid previous value is found
end

function find_next_valid(consumption, idx)
    for i in idx+1:length(consumption)
        if !ismissing(consumption[i])
            return consumption[i]
        end
    end
    return NaN  # If no valid next value is found
end

# Interpolate consumption values
data.Global_active_power = interpolate_consumption(consumption)  # Replace column with interpolated values

# Save the updated DataFrame to a new CSV file
output_file_path = joinpath(resources_dir, "data1min.csv")
CSV.write(output_file_path, data)

println("File successfully saved at: $output_file_path")
