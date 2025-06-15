using Images, FileIO, Statistics

println("Introduce el nombre de las carpetas a cargar (separadas por un espacio):")
folders = readline()
folders = split(strip(folders), " ") |> x -> filter(!isempty, map(strip, x))

println("Carpetas a cargar: ", folders)

if isempty(folders)
    println("No se introdujeron carpetas. Saliendo del programa.")
    exit()
end

folder_stats = Dict{String, Dict{String, Float64}}()

for folder in folders
    path = "parasite-dataset/$folder"
    if !isdir(path)
        println("La carpeta '$folder' no existe")
        continue
    end

    println("\nProcesando carpeta: $folder")

    # Inicializar acumuladores de características por carpeta
    all_features = Dict(
        "red_mean" => Float64[],
        "green_mean" => Float64[],
        "blue_mean" => Float64[],
        "red_std" => Float64[],
        "green_std" => Float64[],
        "blue_std" => Float64[],
        "red_green_ratio" => Float64[],
        "red_blue_ratio" => Float64[],
        "green_blue_ratio" => Float64[],
        "intensity_mean" => Float64[],
        "intensity_std" => Float64[],
        "dark_pixel_ratio" => Float64[],
        "center_mean" => Float64[],
        "periphery_mean" => Float64[],
        "center_periphery_ratio" => Float64[]
    )

    images = [load("$path/$file") for file in readdir(path) if endswith(file, ".jpg") || endswith(file, ".png")]
    if isempty(images)
        println("No se encontraron imágenes en '$folder'")
        continue
    end

    for img in images
        # EXTRAER CANALES RGB
        red_channel = float32.(channelview(img)[1, :, :])
        green_channel = float32.(channelview(img)[2, :, :]) 
        blue_channel = float32.(channelview(img)[3, :, :])

        # Cálculos básicos de color
        push!(all_features["red_mean"], mean(red_channel))
        push!(all_features["green_mean"], mean(green_channel))
        push!(all_features["blue_mean"], mean(blue_channel))
        
        push!(all_features["red_std"], std(red_channel))
        push!(all_features["green_std"], std(green_channel))
        push!(all_features["blue_std"], std(blue_channel))

        # Cálculos de relaciones de color
        push!(all_features["red_green_ratio"], mean(red_channel) / (mean(green_channel) + 1e-10))
        push!(all_features["red_blue_ratio"], mean(red_channel) / (mean(blue_channel) + 1e-10))
        push!(all_features["green_blue_ratio"], mean(green_channel) / (mean(blue_channel) + 1e-10))

        # CONVERTIR A ESCALA DE GRIS
        img_gray = Gray.(img)
        gray_values = Float64.(img_gray)

        # Cálculos de intensidad
        push!(all_features["intensity_mean"], mean(gray_values))
        push!(all_features["intensity_std"], std(gray_values))

        # RATIO DE PÍXELES OSCUROS
        threshold = mean(gray_values) - 0.5 * std(gray_values)
        dark_pixels = sum(gray_values .< threshold)
        push!(all_features["dark_pixel_ratio"], dark_pixels / length(gray_values))

        # CENTRO VS PERIFERIA
        h, w = size(gray_values)
        center_region = gray_values[div(h, 3):2*div(h, 3), div(w, 3):2*div(w, 3)]
        periphery_mask = trues(size(gray_values))
        periphery_mask[div(h, 3):2*div(h, 3), div(w, 3):2*div(w, 3)] .= false
        periphery = gray_values[periphery_mask]

        push!(all_features["center_mean"], mean(center_region))
        push!(all_features["periphery_mean"], mean(periphery))
        push!(all_features["center_periphery_ratio"], mean(center_region) / (mean(periphery) + 1e-10))
    end

    # Promediar las métricas por carpeta
    folder_stats[folder] = Dict(
        "blue_mean" => mean(all_features["blue_mean"]),
        "blue_std" => mean(all_features["blue_std"]),
        "center_mean" => mean(all_features["center_mean"]),
        "center_periphery_ratio" => mean(all_features["center_periphery_ratio"]),
        "dark_pixel_ratio" => mean(all_features["dark_pixel_ratio"]),
        "green_blue_ratio" => mean(all_features["green_blue_ratio"]),
        "green_mean" => mean(all_features["green_mean"]),
        "green_std" => mean(all_features["green_std"]),
        "intensity_mean" => mean(all_features["intensity_mean"]),
        "intensity_std" => mean(all_features["intensity_std"]),
        "periphery_mean" => mean(all_features["periphery_mean"]),
        "red_blue_ratio" => mean(all_features["red_blue_ratio"]),
        "red_green_ratio" => mean(all_features["red_green_ratio"]),
        "red_mean" => mean(all_features["red_mean"]),
        "red_std" => mean(all_features["red_std"])
    )
end

# Mostrar resultados por carpeta
println("\nEstadísticas globales por carpeta:")
for (folder, stats) in folder_stats
    println("\nCarpeta: $folder")
    for (name, value) in sort(collect(stats); by=first)
        println("  $(lpad(name, 30)): ", round(value, digits=5))
    end
end
