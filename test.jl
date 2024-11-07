using Plots
using DelimitedFiles
using Dates

# Load the data from the text file
data = readdlm("household_power_consumption.txt", ';', String)

# Extract the relevant columns (assuming the second column is Time and the third is Global_active_power)
time = data[2:end, 2]  # All rows, second column
consumption = data[2:end, 3]  # All rows, third column

# Interpolate "?" values with the average of the previous and next valid values
function interpolate_consumption(consumption)
    consumption_float = Vector{Union{Float64, String}}(consumption)  # Mutable array for processing

    for i in 1:length(consumption_float)
        if consumption_float[i] == "?"
            prev_value = find_previous_valid(consumption_float, i)
            next_value = find_next_valid(consumption_float, i)
            # Replace with the arithmetic mean of the previous and next values
            consumption_float[i] = (prev_value + next_value) / 2
        else
            # Convert valid strings to Float64
            consumption_float[i] = parse(Float64, consumption_float[i])
        end
    end
    
    return Float64.(consumption_float)  # Convert to Float64 array
end

function find_previous_valid(consumption_float, idx)
    for i in reverse(1:idx-1)
        # Check if the current value is not "?" and is not NaN
        if consumption_float[i] != "?" && !isnan(consumption_float[i])
            return consumption_float[i]  # Return the valid Float64 value directly
        end
    end
    return NaN  # Fallback if no previous valid value is found
end
function find_next_valid(consumption_float, idx)
    for i in idx+1:length(consumption_float)
        if consumption_float[i] != "?" && consumption_float[i] != NaN
            return parse(Float64, consumption_float[i])
        end
    end
    return NaN  # Fallback if no next valid value is found
end

# Interpolate consumption values
consumption_float = interpolate_consumption(consumption)

# Format the time to HH:MM
trimmed_time = [join(split(t, ":")[1:2], ":") for t in time]

# Iterate over the specified ranges of both arrays
#= for (t, c) in zip(trimmed_time[19720:19730], consumption_float[19720:19730])
    println("Time: $t     Consumption: $c")
end =#

# Open a file for writing
open("formattedData.txt", "w") do file
    # Ensure both arrays are the same length
    if length(trimmed_time) != length(consumption_float)
        println("Error: trimmed_time and consumption_float must be of the same length.")
        return
    end

    # Iterate over the indices of both arrays using eachindex
    for i in eachindex(trimmed_time)
        # Write each line in the format "trimmed_time[i]:consumption_float[i]"
        println(file, "$(trimmed_time[i]):$(consumption_float[i])")
    end
end

# Create the plot
p = plot(trimmed_time[1000:2440], consumption_float[1000:2440], 
         xlabel="Time (HH:MM)", ylabel="Global Active Power (kilowatts)", 
         title="Global Active Power vs Time", legend=false)

# Customize the plot appearance
plot!(size=(1200, 800), guidefont=font(10), tickfont=font(8))

# Display the plot
display(p)


 