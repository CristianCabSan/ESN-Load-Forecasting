using CSV, Dates

# Load the dataset (replace "input_data.csv" with your actual filename)
data = CSV.read("input_data.csv", DataFrame; delim=';', header=false)
rename!(data, [:Date, :Time, :Consumption])

# Parse Date and Time into a single DateTime object
data.Timestamp = DateTime.(data.Date .* " " .* data.Time, "dd/mm/yyyy HH:MM:SS")

# Add a column for the start of the year (relative timestamp from Jan 1 of the same year)
data.StartOfYearTimestamp = Int.((data.Timestamp .- DateTime(year.(data.Timestamp))) ./ Millisecond(1)) ./ 1000

# Add a column for the start of the day (relative timestamp in seconds from the start of the same day)
data.StartOfDayTimestamp = Int.((data.Timestamp .- floor.(data.Timestamp, Day)) ./ Second(1))

# Save the new dataset
output_filename = "processed_dataset_with_timestamps.csv"
CSV.write(output_filename, data)

println("Dataset processed and saved as $output_filename.")
