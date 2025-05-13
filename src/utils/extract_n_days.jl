using DataFrames
import CSV
import Random
import Dates

#= 
    Takes a .csv file with a DateTime column and selects n unique days randomly.
    Returns a new CSV file with the selected rows.
=#


n = 365 # Number of unique days to select
output_name = "data10secs_with_timestamps_random_days.csv"
data_name = "data10secs_with_timestamps.csv"

# Get the resources folder path
resources_dir = joinpath(@__DIR__, "..", "..", "resources")
data_path = joinpath(resources_dir, data_name)

# Read the data
df = CSV.read(data_path, DataFrame, dateformat="yyyy-mm-ddTHH:MM:SS.s")

function select_random_days(data::DataFrame, n::Int)
    selected_rows = DataFrame()
    used_dates = Set()  # Keep track of selected dates to avoid repetition

    for i in 1:n
        time = @elapsed begin
            # Get available days that haven't been used
            available_rows = filter(row -> Date(row.DateTime) ∉ used_dates, data)
            
            # Handles edge case of more days selected than available
            if isempty(available_rows)
                println("No more unique days available.")
                break
            end

            # Select a random row and gets its date
            rand_row = rand(eachrow(available_rows))  
            rand_datetime = rand_row.DateTime
            rand_date = Date(rand_datetime)

            # Mark this date as used
            push!(used_dates, rand_date)

            # Select all rows with the same date
            matching_rows = filter(row -> Date(row.DateTime) == rand_date, data)

            # Append to the final DataFrame
            append!(selected_rows, matching_rows)
        end
        println("Vuelta: $i -------- Tiempo: $time")
    end

    return selected_rows
end

# Save the result to a new CSV
newData = select_random_days(df, n)
CSV.write("$resources_dir\\$output_name", newData)
