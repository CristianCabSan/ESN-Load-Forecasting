using CSV
using DataFrames
using Dates

#=
    Takes a .txt file with at least Date and Time columns and outputs a .csv file. 
    The Date and Time columns are merged into a single DateTime column in the resulting file.
    You can choose whether to keep all columns or only DateTime and Global_active_power.
=#

# Set to false to keep only DateTime and Global_active_power
keep_all_columns = false  

# Get the data
data_name = "household_power_consumption.txt"
resources_dir = joinpath(@__DIR__, "..", "..", "resources")
data_path = joinpath(resources_dir, data_name)
data = CSV.read(data_path, DataFrame; delim = ";")

# Convert Date and Time to DateTime
df = DateFormat("dd/mm/yyyy")
dates = Date.(data.Date, df)
times = data.Time
dateTimes = dates .+ times
data[!, :DateTime] = dateTimes
data = select(data, :DateTime, Not(:DateTime))  # Move DateTime to the first column

# Remove the original Date and Time columns
select!(data, Not([:Date, :Time]))

# Filter columns based on flag
if keep_all_columns
    nothing  # Do nothing
else
    # Keep only DateTime and Global_active_power
    data = select(data, [:DateTime, :Global_active_power])
end

# Save to CSV
CSV.write("$resources_dir/data.csv", data)
