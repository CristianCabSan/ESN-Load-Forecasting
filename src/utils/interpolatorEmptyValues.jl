using Plots
using DelimitedFiles
using Dates

# Cargar los datos del archivo de texto, omitiendo el encabezado
data = readdlm("household_power_consumption.txt", ';', String; header=true)[1]

# Extraer las columnas relevantes (segunda columna es hora, tercera es el consumo)
time = data[2:end, 2]  # Todas las filas excepto el encabezado, segunda columna
consumption = data[2:end, 3]  # Todas las filas excepto el encabezado, tercera columna

# Interpolar valores "?" con el promedio de los valores válidos anteriores y posteriores
function interpolate_consumption(consumption)
    consumption_float = Vector{Union{Float64, String}}(consumption)  # Mutable para procesar

    for i in 1:length(consumption_float)
        if consumption_float[i] == "?"  # Si es un valor inválido
            prev_value = find_previous_valid(consumption_float, i)
            next_value = find_next_valid(consumption_float, i)
            consumption_float[i] = (prev_value + next_value) / 2  # Reemplazar por el promedio
        elseif typeof(consumption_float[i]) == String
            # Convertir los valores válidos a Float64 si son cadenas
            consumption_float[i] = parse(Float64, consumption_float[i])
        end
    end

    return Float64.(consumption_float)  # Convertir a un vector de Float64
end

function find_previous_valid(consumption_float, idx)
    for i in reverse(1:idx-1)
        if consumption_float[i] != "?" && typeof(consumption_float[i]) == String
            return parse(Float64, consumption_float[i])
        elseif consumption_float[i] != "?"
            return consumption_float[i]  # Ya es Float64
        end
    end
    return NaN  # Si no se encuentra un valor previo válido
end

function find_next_valid(consumption_float, idx)
    for i in idx+1:length(consumption_float)
        if consumption_float[i] != "?" && typeof(consumption_float[i]) == String
            return parse(Float64, consumption_float[i])
        elseif consumption_float[i] != "?"
            return consumption_float[i]  # Ya es Float64
        end
    end
    return NaN  # Si no se encuentra un valor siguiente válido
end

# Interpolar valores de consumo
consumption_float = interpolate_consumption(consumption)

# Formatear el tiempo a HH:MM
trimmed_time = [join(split(t, ":")[1:2], ":") for t in time]

# Crear un archivo con formato HH:MM;Valor
open("formattedData.txt", "w") do file
    for (t, c) in zip(trimmed_time, consumption_float)
        println(file, "$t;$c")  # Formato requerido HH:MM;Valor
    end
end
