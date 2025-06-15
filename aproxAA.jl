# Script principal para seleccionar y ejecutar modelos de AA clásico
using Random
using Statistics
using DelimitedFiles

# Incluir los archivos de los modelos
include("modelos/rna.jl")
include("modelos/svm.jl")
include("modelos/knn.jl")
include("modelos/dome.jl")
include("modelos/decisionTree.jl")

# Incluir funciones auxiliares
include("fonts/functionsSol.jl")

# Fijar la semilla aleatoria para garantizar la repetibilidad
Random.seed!(1234)

# Función para cargar y preprocesar los datos
function cargarDatos(rutaArchivo, separador=',')
    # Cargar los datos
    datos = readdlm(rutaArchivo, separador)
    
    # Separar características y etiquetas
    entradas = Float64.(datos[:, 1:end-1])  # Convertir explícitamente a Float64
    etiquetas = Float64.(datos[:, end])     # Convertir explícitamente a Float64
    
    return entradas, etiquetas
end

# Función para determinar el número máximo de folds posible
function maxFoldsPosibles(etiquetas)
    if isa(etiquetas, AbstractArray{Bool})
        # Para etiquetas booleanas
        numPositivos = sum(etiquetas)
        numNegativos = length(etiquetas) - numPositivos
        return min(numPositivos, numNegativos)
    else
        # Para etiquetas categóricas
        clases = unique(etiquetas)
        minPatrones = typemax(Int)
        
        for clase in clases
            numPatrones = sum(etiquetas .== clase)
            minPatrones = min(minPatrones, numPatrones)
        end
        
        return minPatrones
    end
end

# Función principal
function main()
    # Cargar los datos (reemplazar con tu ruta)
    rutaDataset = "parasiteCSV/all_parasites.csv"
    X, y = cargarDatos(rutaDataset)
    
    # Normalizar los datos
    X_norm = normalizeZeroMean(X)
    
    # Determinar el número máximo de folds posible
    max_folds = maxFoldsPosibles(y)
    k_folds = min(3, max_folds)  # Usar 3 folds o menos si es necesario
    
    println("Usando $k_folds folds para validación cruzada (máximo posible: $max_folds)")
    
    # Crear índices para validación cruzada
    cv_indices = crossvalidation(y, k_folds)
    
    # Menú para seleccionar el modelo
    println("Seleccione el modelo a ejecutar:")
    println("1. Redes Neuronales Artificiales")
    println("2. k-Nearest Neighbors")
    println("3. Árboles de Decisión")
    println("4. Support Vector Machines")
    println("5. DoME")
    
    # Leer la opción del usuario
    opcion = parse(Int, readline())
    
    # Ejecutar el modelo seleccionado
    if opcion == 1
        ejecutarModeloRNA(X_norm, y, cv_indices)
    elseif opcion == 2
        ejecutarModeloKNN(X_norm, y, cv_indices)
    elseif opcion == 3
        ejecutarModeloArbolDecision(X_norm, y, cv_indices)
    elseif opcion == 4
        ejecutarModeloSVM(X_norm, y, cv_indices)
    elseif opcion == 5
        ejecutarModeloDoME(X_norm, y, cv_indices)
    else
        println("Opción no válida")
    end
end

# Ejecutar la función principal
main()