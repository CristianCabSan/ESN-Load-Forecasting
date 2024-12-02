using Plots
using DelimitedFiles
using Dates

# Load the data from the text file, omitting the header
script_dir = @__DIR__

# Move up two levels to the project root and construct the path to the resources folder
resources_dir = joinpath(script_dir, "..", "..", "resources")

# Construct the full path to the file
file_name = "household_power_consumption.txt"
file_path = joinpath(resources_dir, file_name)

# Check if the file exists, read it into `data` if it does, or print an error
if isfile(file_path)
    data = readdlm(file_path, ';', String; header=true)[1]
else
    println("File not found at path: $file_path")
    exit(1)  # Exit if file is not found
end

# Extract the relevant columns (first column is date, second is time, third is consumption)
dates = data[1:end, 1]
times = data[1:end, 2]
consumption = data[1:end, 3]

# Interpolate "?" values with the average of the valid previous and next values
function interpolate_consumption(consumption)
    consumption_float = Vector{Union{Float64, String}}(consumption)  # Mutable to process

    for i in 1:length(consumption_float)
        if consumption_float[i] == "?"  # If it is an invalid value
            prev_value = find_previous_valid(consumption_float, i)
            next_value = find_next_valid(consumption_float, i)
            consumption_float[i] = (prev_value + next_value) / 2  # Replace with the average
        elseif typeof(consumption_float[i]) == String
            # Convert valid values to Float64 if they are strings
            consumption_float[i] = parse(Float64, consumption_float[i])
        end
    end

    return Float64.(consumption_float)  # Convert to a vector of Float64
end

function find_previous_valid(consumption_float, idx)
    for i in reverse(1:idx-1)
        if consumption_float[i] != "?" && typeof(consumption_float[i]) == String
            return parse(Float64, consumption_float[i])
        elseif consumption_float[i] != "?"
            return consumption_float[i]  # Already Float64
        end
    end
    return NaN  # If no valid previous value is found
end

function find_next_valid(consumption_float, idx)
    for i in idx+1:length(consumption_float)
        if consumption_float[i] != "?" && typeof(consumption_float[i]) == String
            return parse(Float64, consumption_float[i])
        elseif consumption_float[i] != "?"
            return consumption_float[i]  # Already Float64
        end
    end
    return NaN  # If no valid next value is found
end

# Interpolate consumption values
consumption_float = interpolate_consumption(consumption)

# Combine the date and time columns into a single formatted timestamp column
date_time = [join([dates[i], times[i]], ";") for i in 1:length(dates)]

# Create a file with the format Date;Timestamp;Value
output_file_path = joinpath(resources_dir, "data1min.txt")
open(output_file_path, "w") do file
    for (dt, c) in zip(date_time, consumption_float)
        println(file, "$dt;$c")  # Required format Date;Timestamp;Value
    end
end

println("File successfully saved at: $output_file_path")
