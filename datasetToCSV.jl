using Images, FileIO, Statistics, CSV, DataFrames

# Function to extract selected features from an image
function extract_selected_features(img)
    features = Dict()
    
    # COLOR FEATURES
    # Extract RGB channels
    red_channel = float32.(channelview(img)[1, :, :])
    green_channel = float32.(channelview(img)[2, :, :]) 
    blue_channel = float32.(channelview(img)[3, :, :])
    
    # Basic color statistics
    features["red_mean"] = Float64(mean(red_channel))
    features["green_mean"] = Float64(mean(green_channel))
    features["blue_mean"] = Float64(mean(blue_channel))
    
    features["red_std"] = Float64(std(red_channel))
    features["green_std"] = Float64(std(green_channel))
    features["blue_std"] = Float64(std(blue_channel))
    
    # Color ratios
    features["red_green_ratio"] = features["red_mean"] / (features["green_mean"] + 1e-10)
    features["red_blue_ratio"] = features["red_mean"] / (features["blue_mean"] + 1e-10)
    features["green_blue_ratio"] = features["green_mean"] / (features["blue_mean"] + 1e-10)
    
    # Convert to grayscale for other features
    img_gray = Gray.(img)
    gray_values = Float64.(img_gray)
    
    # INTENSITY FEATURES
    # Basic intensity statistics
    features["intensity_mean"] = Float64(mean(gray_values))
    features["intensity_std"] = Float64(std(gray_values))
    
    # Dark pixel ratio (using adaptive threshold)
    threshold = features["intensity_mean"] - 0.5 * features["intensity_std"]
    dark_pixels = sum(gray_values .< threshold)
    features["dark_pixel_ratio"] = Float64(dark_pixels) / Float64(length(gray_values))
    
    # Center vs periphery contrast
    h, w = size(img_gray)
    center_region = gray_values[div(h,3):2*div(h,3), div(w,3):2*div(w,3)]
    periphery_mask = trues(size(gray_values))
    periphery_mask[div(h,3):2*div(h,3), div(w,3):2*div(w,3)] .= false
    periphery = gray_values[periphery_mask]
    
    features["center_mean"] = Float64(mean(center_region))
    features["periphery_mean"] = Float64(mean(periphery))
    features["center_periphery_ratio"] = features["center_mean"] / (features["periphery_mean"] + 1e-10)
    
    # TEXTURE FEATURES
    # Local homogeneity (smoothness)
    try
        # Apply a local mean filter
        local_mean = imfilter(gray_values, ones(3,3)/9)
        # Calculate local variance
        local_var = (gray_values .- local_mean).^2
        # Homogeneity is inverse of variance
        features["local_homogeneity"] = Float64(1 / (1 + mean(local_var)))
    catch
        features["local_homogeneity"] = 0.0
    end
    
    # Simple entropy calculation (measure of randomness)
    try
        # Create a histogram with 32 bins
        hist_counts = zeros(32)
        min_val = minimum(gray_values)
        max_val = maximum(gray_values)
        bin_width = (max_val - min_val) / 32
        
        for val in gray_values
            bin_idx = min(32, max(1, floor(Int, (val - min_val) / bin_width) + 1))
            hist_counts[bin_idx] += 1
        end
        
        # Normalize histogram
        hist_norm = hist_counts ./ sum(hist_counts)
        
        # Calculate entropy
        entropy_val = 0.0
        for p in hist_norm
            if p > 0
                entropy_val -= p * log2(p)
            end
        end
        features["texture_entropy"] = Float64(entropy_val)
        
        # Calculate energy (uniformity)
        features["texture_energy"] = Float64(sum(hist_norm.^2))
    catch
        features["texture_entropy"] = 0.0
        features["texture_energy"] = 0.0
    end
    
    # SHAPE FEATURES - REMOVED
    # The aspect_ratio, circularity, and compactness features have been removed
    # since they are the same for all examples
    
    # Ensure all values are simple numeric types (not Gray{Float64})
    for (key, value) in features
        if typeof(value) != Float64
            features[key] = Float64(value)
        end
    end
    
    return features
end

# Create output directory if it doesn't exist
output_dir = "parasiteCSV"
if !isdir(output_dir)
    mkdir(output_dir)
    println("Creada carpeta '$output_dir' para guardar los archivos CSV")
end

# Main program
println("Introduce el nombre de las carpetas a cargar (separadas por un espacio):")
println("Ejemplo: babesia trypanosoma leishmania")
folders = readline()
folders = split(strip(folders), " ") |> x -> filter(!isempty, map(strip, x))

println("Carpetas a cargar: ", folders)

if isempty(folders)
    println("No se introdujeron carpetas. Saliendo del programa.")
    exit()
end

# Get all unique feature names across all parasites
all_feature_names = Set{String}()
all_parasite_features = []  # Will store tuples of (features, parasite_name)

# Process each folder
for folder in folders
    path = "parasite-dataset/$folder"
    if !isdir(path)
        println("La carpeta '$folder' no existe")
        continue
    end
    
    println("\n=== Procesando parásito: $folder ===")
    
    # Get all image files
    image_files = filter(file -> endswith(lowercase(file), ".jpg") || 
                                 endswith(lowercase(file), ".png") || 
                                 endswith(lowercase(file), ".jpeg"), 
                         readdir(path))
    
    println("Encontradas $(length(image_files)) imágenes en la carpeta '$folder'")
    
    # Process each image
    success_count = 0
    fail_count = 0
    
    # Show progress less frequently
    progress_interval = max(1, div(length(image_files), 10))
    
    for (i, file) in enumerate(image_files)
        full_path = "$path/$file"
        
        # Show progress less frequently
        if i % progress_interval == 0 || i == 1
            print("\rProcesando: $(round(Int, 100 * i / length(image_files)))% completado ($(i)/$(length(image_files)))")
        end
        
        try
            img = load(full_path)
            features = extract_selected_features(img)
            
            # Add to all feature names
            for key in keys(features)
                push!(all_feature_names, key)
            end
            
            # Store features with parasite name
            push!(all_parasite_features, (features, folder))
            
            success_count += 1
        catch e
            fail_count += 1
            # Only print errors for the first few failures to avoid flooding the console
            if fail_count <= 3
                println("\nError procesando $file: $e")
            elseif fail_count == 4
                println("\nOmitiendo mensajes de error adicionales...")
            end
        end
    end
    
    println("\nTerminado procesamiento de '$folder'. Total: $(length(image_files)), Éxito: $success_count, Fallidas: $fail_count")
end

# Convert Set to Array and then sort
all_feature_names = sort(collect(all_feature_names))

# Create a single CSV file with all parasites
if !isempty(all_parasite_features)
    # Create a matrix for all samples
    num_samples = length(all_parasite_features)
    num_features = length(all_feature_names)
    data_matrix = zeros(num_samples, num_features)
    parasite_labels = String[]
    
    for (i, (features, parasite_name)) in enumerate(all_parasite_features)
        for (j, feature) in enumerate(all_feature_names)
            data_matrix[i, j] = get(features, feature, 0.0)
        end
        push!(parasite_labels, parasite_name)
    end
    
    # Create a mapping from parasite names to numeric labels
    unique_parasites = unique(parasite_labels)
    parasite_to_num = Dict(parasite => i for (i, parasite) in enumerate(unique_parasites))
    
    # Save the mapping to a reference file
    map_file = joinpath(output_dir, "parasite_label_map.txt")
    open(map_file, "w") do io
        println(io, "Numeric_Label,Parasite_Name")
        for (parasite, num) in parasite_to_num
            println(io, "$num,$parasite")
        end
    end
    println("\nMapeo de etiquetas guardado en: $map_file")
    
    # Convert string labels to numeric labels
    numeric_labels = [parasite_to_num[parasite] for parasite in parasite_labels]
    
    # Save to a single CSV file with numeric labels
    output_file = joinpath(output_dir, "all_parasites.csv")
    
    # Write the matrix to CSV without headers
    open(output_file, "w") do io
        for i in 1:size(data_matrix, 1)
            # Join all features and add numeric label at the end
            line = join(data_matrix[i, :], ",") * "," * string(numeric_labels[i])
            println(io, line)
        end
    end
    
    println("\n=== Resumen final ===")
    println("Se ha creado un único archivo CSV con todos los parásitos: $output_file")
    println("Dimensiones de la matriz: $(size(data_matrix, 1)) filas × $(size(data_matrix, 2)) columnas (más la columna de etiquetas numéricas)")
    println("Características eliminadas: aspect_ratio, circularity, compactness")
    println("Mapeo de etiquetas:")
    for (parasite, num) in parasite_to_num
        println("  $parasite => $num")
    end
    
    # Save feature names to a separate file for reference (optional)
    feature_file = joinpath(output_dir, "feature_names.txt")
    open(feature_file, "w") do io
        for (i, feature) in enumerate(all_feature_names)
            println(io, "$i,$feature")
        end
    end
    println("Nombres de características guardados en $feature_file (para referencia)")
else
    println("\nNo se extrajeron características con éxito para ningún parásito")
end