using DataFrames
using CSV
using Random
using Dates

#= 
    Takes a .csv file with a DateTime column and selects N unique days randomly.
    Returns a new CSV file with the selected rows.
=#

N = 365 # Number of unique days to select
data_name = "data1sec.csv"
output_name = "data1secRandomDays.csv"

# Get the resources folder path
resources_dir = joinpath(@__DIR__, "..", "..", "resources")
data_path = joinpath(resources_dir, data_name)

# Read the data
df = CSV.read(data_path, DataFrame, dateformat="yyyy-mm-ddTHH:MM:SS.s")

function select_random_days(data::DataFrame, N::Int)
    # Extraemos todas las fechas únicas (días)
    unique_days = unique(Date.(data.DateTime))
    
    # Si pedimos más días que los que hay, ajustamos
    N = min(N, length(unique_days))
    
    selected_days = Set{Date}()
    
    for i in 1:N
        time = @elapsed begin
            # Seleccionamos un día aleatorio no usado
            remaining_days = setdiff(unique_days, selected_days)
            if isempty(remaining_days)
                println("No more unique days available.")
                break
            end
            day = rand(remaining_days)
            push!(selected_days, day)
        end
        println("Vuelta: $i -------- Tiempo: $time")
    end

    # Filtramos el DataFrame solo una vez para conservar los días seleccionados
    selected_rows = filter(row -> Date(row.DateTime) in selected_days, data)
    
    return selected_rows
end

# Save the result to a new CSV
newData = select_random_days(df, N)
CSV.write(joinpath(resources_dir,output_name), newData)
