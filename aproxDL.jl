# aproxDL.jl
# Aproximación basada en Deep Learning para clasificación de parásitos

using Flux
using Flux.Losses
using Flux: onehotbatch, onecold, adjust!
using Images
using FileIO
using Statistics: mean, std
using Random
using DelimitedFiles
using LinearAlgebra

# Asegurarnos de que la función load está disponible
import FileIO: load

# Incluir las funciones de la práctica 1
include("fonts/functionsSol.jl")

# Fijar la semilla aleatoria para garantizar la repetibilidad
Random.seed!(1234)

# 1. Arquitectura más simple
function arquitecturaSimple(width, height, numClases)
    return Chain(
        Conv((3, 3), 1=>16, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(32 * div(width, 4) * div(height, 4), numClases),
        softmax
    )
end

# 2. Arquitectura más profunda
function arquitecturaProfunda(width, height, numClases)
    return Chain(
        Conv((3, 3), 1=>16, pad=1, relu),
        Conv((3, 3), 16=>32, pad=1, relu),
        MaxPool((2,2)),
        Conv((3, 3), 32=>64, pad=1, relu),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(64 * div(width, 4) * div(height, 4), numClases),
        softmax
    )
end

# 3. Arquitectura Bottleneck
function arquitecturaBottleneck(width, height, numClases)
    return Chain(
        Conv((1, 1), 1=>16, relu),  # proyección
        Conv((3, 3), 16=>32, pad=1, relu),
        MaxPool((2,2)),
        Conv((1, 1), 32=>32, relu),
        Conv((3, 3), 32=>64, pad=1, relu),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(64 * div(width, 4) * div(height, 4), numClases),
        softmax
    )
end

# 4. Arquitectura con diferentes tamaños de kernel
function arquitecturaMultiKernel(width, height, numClases)
    return Chain(
        Conv((5, 5), 1=>16, pad=(2,2), relu),
        MaxPool((2,2)),
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((1, 1), 32=>32, relu),  # 1x1 convolution para reducción de dimensionalidad
        Conv((3, 3), 32=>64, pad=(1,1), relu),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(64 * div(width, 8) * div(height, 8), numClases),
        softmax
    )
end


# Función para cargar imágenes desde carpetas
function cargarImagenes(rutaBase, tamañoImagen=(64, 64))
    println("Cargando imágenes desde: ", rutaBase)
    
    # Verificar que la carpeta existe
    if !isdir(rutaBase)
        error("La carpeta $rutaBase no existe")
    end
    
    # Obtener las carpetas (clases) dentro de la ruta base
    carpetas = filter(x -> isdir(joinpath(rutaBase, x)), readdir(rutaBase))
    
    if isempty(carpetas)
        error("No se encontraron carpetas en $rutaBase")
    end
    
    println("Clases encontradas: ", carpetas)
    
    # Mapear nombres de carpetas a índices numéricos
    claseAIndice = Dict(clase => i for (i, clase) in enumerate(carpetas))
    
    # Arrays para almacenar imágenes y etiquetas
    imagenes = []
    etiquetas = []
    
    # Para cada carpeta (clase)
    for carpeta in carpetas
        rutaCarpeta = joinpath(rutaBase, carpeta)
        indiceClase = claseAIndice[carpeta]
        
        # Obtener archivos de imagen en la carpeta
        archivosImagen = filter(x -> lowercase(splitext(x)[2]) in [".jpg", ".jpeg", ".png"], 
                               readdir(rutaCarpeta))
        
        println("Procesando clase $carpeta (índice $indiceClase): $(length(archivosImagen)) imágenes")
        
        # Cargar cada imagen
        contadorExito = 0
        contadorError = 0
        
        for archivo in archivosImagen
            rutaArchivo = joinpath(rutaCarpeta, archivo)
            try
                # Cargar imagen usando FileIO.load explícitamente
                img = FileIO.load(rutaArchivo)
                
                # Mostrar información sobre la imagen original
                if contadorExito == 0
                    println("  Primera imagen: tamaño=$(size(img)), tipo=$(typeof(img))")
                end
                
                # Convertir a escala de grises si es necesario
                if eltype(img) <: Colorant
                    img = Gray.(img)
                end
                
                # Redimensionar a tamaño fijo
                img = imresize(img, tamañoImagen)
                
                # Convertir a Float32 y normalizar entre 0 y 1
                imgArray = Float32.(img)
                
                # Verificar que la imagen tiene el tamaño correcto después del procesamiento
                if size(imgArray) != tamañoImagen
                    println("  Advertencia: La imagen $archivo tiene tamaño $(size(imgArray)) después del redimensionamiento")
                    continue
                end
                
                # Añadir a los arrays
                push!(imagenes, imgArray)
                push!(etiquetas, indiceClase)
                contadorExito += 1
                
                # Mostrar progreso cada 50 imágenes
                if contadorExito % 50 == 0
                    println("  Procesadas $contadorExito imágenes de la clase $carpeta")
                end
            catch e
                contadorError += 1
                if contadorError <= 5  # Limitar el número de errores mostrados
                    println("  Error al procesar $rutaArchivo: $e")
                elseif contadorError == 6
                    println("  (Se omitirán más mensajes de error para esta clase)")
                end
            end
        end
        
        println("  Procesadas con éxito $contadorExito imágenes de la clase $carpeta (errores: $contadorError)")
        
        # Verificar que se cargaron algunas imágenes para esta clase
        if contadorExito == 0
            println("  ADVERTENCIA: No se pudo cargar ninguna imagen para la clase $carpeta")
        end
    end
    
    totalImagenes = length(imagenes)
    println("Total de imágenes cargadas: ", totalImagenes)
    
    if totalImagenes == 0
        error("No se pudo cargar ninguna imagen. Verifica la ruta y los formatos de archivo.")
    end
    
    # Convertir etiquetas a vector
    etiquetas = Int.(etiquetas)
    
    # Obtener clases únicas
    clases = sort(unique(etiquetas))
    
    # Verificar que tenemos al menos una imagen por clase
    for clase in clases
        if count(==(clase), etiquetas) == 0
            println("ADVERTENCIA: La clase $clase no tiene imágenes")
        end
    end
    
    return imagenes, etiquetas, clases, carpetas
end

# Función para convertir array de imágenes al formato WHCN
function convertirArrayImagenesWHCN(imagenes, tamañoImagen=(64, 64))
    numPatrones = length(imagenes)
    
    # Verificar que hay imágenes
    if numPatrones == 0
        error("No hay imágenes para convertir a formato WHCN")
    end
    
    # Importante que sea un array de Float32
    nuevoArray = Array{Float32, 4}(undef, tamañoImagen[1], tamañoImagen[2], 1, numPatrones)
    
    for i in 1:numPatrones
        # Verificar dimensiones
        if size(imagenes[i]) != tamañoImagen
            error("La imagen $i tiene tamaño $(size(imagenes[i])), pero se esperaba $tamañoImagen")
        end
        
        # Copiar los datos
        nuevoArray[:, :, 1, i] .= imagenes[i]
    end
    
    # Verificar que los valores están en el rango [0,1]
    minVal = minimum(nuevoArray)
    maxVal = maximum(nuevoArray)
    
    if minVal < 0 || maxVal > 1
        println("ADVERTENCIA: Los valores de las imágenes están fuera del rango [0,1]: [$minVal, $maxVal]")
        println("Normalizando valores...")
        nuevoArray = (nuevoArray .- minVal) ./ (maxVal - minVal + eps(Float32))
    end
    
    return nuevoArray
end

# Función para dividir datos en entrenamiento, validación y test
function dividirDatos(imagenes, etiquetas)
    numPatrones = length(etiquetas)
    indices = shuffle(1:numPatrones)
    
    # 70% entrenamiento, 20% validación, 10% test
    numTrain = round(Int, numPatrones * 0.7)
    numVal = round(Int, numPatrones * 0.2)
    
    indicesTrain = indices[1:numTrain]
    indicesVal = indices[numTrain+1:numTrain+numVal]
    indicesTest = indices[numTrain+numVal+1:end]
    
    return indicesTrain, indicesVal, indicesTest
end

# Definir una arquitectura CNN simple para pruebas iniciales
function arquitecturaSimple(width, height, numClases)
    return Chain(
        Conv((3, 3), 1=>16, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        x -> reshape(x, :, size(x, 4)),
        Dense(32 * div(width, 4) * div(height, 4), numClases),
        softmax
    )
end

# Función para crear y entrenar una CNN
function entrenarCNN(trainImgs, trainLabels, valImgs, valLabels, testImgs, testLabels, clases, arquitectura)
    numClases = length(clases)
    
    # Verificar que hay suficientes datos
    if size(trainImgs, 4) < 10
        error("Muy pocas imágenes de entrenamiento: $(size(trainImgs, 4))")
    end
    
    # Convertir etiquetas a formato one-hot
    trainLabelsOneHot = onehotbatch(trainLabels, clases)
    valLabelsOneHot = onehotbatch(valLabels, clases)
    testLabelsOneHot = onehotbatch(testLabels, clases)
    
    # Crear batches para entrenamiento
    batch_size = min(32, size(trainImgs, 4))  # Ajustar batch_size si hay pocas imágenes
    gruposIndicesBatch = Iterators.partition(1:size(trainImgs, 4), batch_size)
    println("Se han creado ", length(gruposIndicesBatch), " grupos de índices para distribuir los patrones en batches")
    
    train_set = [(trainImgs[:, :, :, indicesBatch], onehotbatch(trainLabels[indicesBatch], clases)) for indicesBatch in gruposIndicesBatch]
    
    # Conjuntos de validación y test
    validation_set = (valImgs, valLabelsOneHot)
    test_set = (testImgs, testLabelsOneHot)
    
    # Definir la arquitectura de la CNN
    ann = arquitectura(size(trainImgs, 1), size(trainImgs, 2), numClases)
    
    # Definir la función de pérdida
    loss(ann, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) : Losses.crossentropy(ann(x), y)
    
    # Función para calcular la precisión
    accuracy(batch) = mean(onecold(ann(batch[1]), clases) .== onecold(batch[2], clases))
    
    # Optimizador
    eta = 0.01
    opt_state = Flux.setup(Adam(eta), ann)
    
    # Entrenamiento
    println("Comenzando entrenamiento...")
    mejorPrecision = -Inf
    criterioFin = false
    numCiclo = 0
    numCicloUltimaMejora = 0
    mejorModelo = nothing
    
    # Métricas para seguimiento
    historicoAccVal = []
    
    try
        while !criterioFin
            # Entrenar un ciclo
            Flux.train!(loss, ann, train_set, opt_state)
            
            numCiclo += 1
            
            # Calcular precisión en validación
            precisionValidacion = accuracy(validation_set)
            push!(historicoAccVal, precisionValidacion)
            
            # Mostrar progreso
            println("Ciclo ", numCiclo, ": Precisión en validación: ", round(100*precisionValidacion, digits=2), "%")
            
            # Si se mejora la precisión en validación, evaluar en test
            if precisionValidacion > mejorPrecision
                mejorPrecision = precisionValidacion
                precisionTest = accuracy(test_set)
                println("   Mejora en validación -> Precisión en test: ", round(100*precisionTest, digits=2), "%")
                mejorModelo = deepcopy(ann)
                numCicloUltimaMejora = numCiclo
            end
            
            # Si no se ha mejorado en 5 ciclos, bajar la tasa de aprendizaje
            if (numCiclo - numCicloUltimaMejora >= 5) && (eta > 1e-5)
                eta /= 10.0
                println("   No se ha mejorado la precisión en validación en 5 ciclos, se baja la tasa de aprendizaje a ", eta)
                adjust!(opt_state, eta)
            end
            
            # Criterios de parada
            if precisionValidacion >= 0.999
                println("   Se para el entrenamiento por haber llegado a una precisión de validación de 99.9%")
                criterioFin = true
            end
            
            if numCiclo - numCicloUltimaMejora >= 7
                println("   Se para el entrenamiento por no haber mejorado la precisión en validación durante 10 ciclos")
                criterioFin = true
            end
            
            if numCiclo >= 20
                println("   Se para el entrenamiento por haber alcanzado el máximo de 100 ciclos")
                criterioFin = true
            end
        end
    catch e
        println("Error durante el entrenamiento: ", e)
        if mejorModelo === nothing
            mejorModelo = ann  # Usar el modelo actual si no hay uno mejor
        end
    end
    
    # Si no se pudo entrenar un modelo, devolver valores por defecto
    if mejorModelo === nothing
        println("ADVERTENCIA: No se pudo entrenar un modelo. Devolviendo valores por defecto.")
        return ann, 0.0, zeros(Int, numClases, numClases), 0.0, 0.0, 0.0, 0.0, [], historicoAccVal
    end
    
    # Calcular matriz de confusión con el mejor modelo
    #try
        predicciones = onecold(mejorModelo(test_set[1]), clases)
        etiquetasReales = testLabels
        
        matrizConfusion = zeros(Int, numClases, numClases)
        for i in 1:length(predicciones)
            matrizConfusion[etiquetasReales[i], predicciones[i]] += 1
        end
        
        # Calcular métricas
        accuracy, recall, precision, f1 = calcularMetricas(matrizConfusion)
        
        return mejorModelo, mejorPrecision, matrizConfusion, accuracy, recall, precision, f1, [], historicoAccVal
    #=catch e
        println("Error al calcular la matriz de confusión: ", e)
        return mejorModelo, mejorPrecision, zeros(Int, numClases, numClases), 0.0, 0.0, 0.0, 0.0, [], historicoAccVal
    end=#
end

# Función para calcular métricas a partir de la matriz de confusión
function calcularMetricas(matrizConfusion)
    numClases = size(matrizConfusion, 1)
    
    # Accuracy global
    accuracy = sum(diag(matrizConfusion)) / sum(matrizConfusion)
    
    # Métricas por clase
    recalls = zeros(numClases)
    precisions = zeros(numClases)
    f1s = zeros(numClases)
    
    for c in 1:numClases
        # Verdaderos positivos
        tp = matrizConfusion[c, c]
        
        # Falsos negativos (suma de la fila c excluyendo tp)
        fn = sum(matrizConfusion[c, :]) - tp
        
        # Falsos positivos (suma de la columna c excluyendo tp)
        fp = sum(matrizConfusion[:, c]) - tp
        
        # Calcular métricas
        recalls[c] = tp / (tp + fn + 1e-10)
        precisions[c] = tp / (tp + fp + 1e-10)
        f1s[c] = 2 * recalls[c] * precisions[c] / (recalls[c] + precisions[c] + 1e-10)
    end
    
    # Promedios
    avgRecall = mean(recalls)
    avgPrecision = mean(precisions)
    avgF1 = mean(f1s)
    
    return accuracy, avgRecall, avgPrecision, avgF1
end

# Función para seleccionar la arquitectura deseada
function seleccionarArquitectura(numArquitectura)
    arquitecturas = Dict(
        1 => ("Arquitectura Simple", arquitecturaSimple),
        2 => ("Arquitectura Profunda", arquitecturaProfunda),
        3 => ("Arquitectura Bottleneck", arquitecturaBottleneck),
        4 => ("Arquitectura MultiKernel", arquitecturaMultiKernel),
    )
    
    # Validar la selección
    if !haskey(arquitecturas, numArquitectura)
        # Si la opción no es válida, mostrar las opciones disponibles
        println("Opción no válida. Por favor, seleccione una de las siguientes arquitecturas:")
        for (num, (nombre, _)) in arquitecturas
            println("$num: $nombre")
        end
        return nothing, ""
    end
    
    nombreArq, funcionArq = arquitecturas[numArquitectura]
    println("Seleccionada: $nombreArq")
    
    return funcionArq, nombreArq
end

# Función principal
function main(numArquitectura=1)
    println("=== Aproximación basada en Deep Learning para clasificación de parásitos ===")
    
    # Verificar que los paquetes necesarios están disponibles
    println("Verificando paquetes necesarios...")
    try
        # Intentar cargar una imagen de prueba para verificar que FileIO funciona
        println("Función load disponible: ", isdefined(FileIO, :load))
    catch e
        println("Error al verificar paquetes: ", e)
    end
    
    # Definir tamaño de imagen
    tamañoImagen = (64, 64)  # Tamaño fijo para todas las imágenes
    
    # Cargar imágenes - Usar ruta absoluta o relativa correcta
    # Intenta con diferentes rutas posibles
    rutasPosibles = [
        "parasite-dataset",
        "./parasite-dataset",
        "../parasite-dataset",
        joinpath(dirname(@__FILE__), "parasite-dataset")
    ]
    
    imagenes = []
    etiquetas = []
    clases = []
    nombresClases = []
    
    # Probar cada ruta posible
    for ruta in rutasPosibles
        try
            println("Intentando cargar desde: $ruta")
            if isdir(ruta)
                imagenes, etiquetas, clases, nombresClases = cargarImagenes(ruta, tamañoImagen)
                println("Imágenes cargadas correctamente desde $ruta")
                break
            else
                println("La carpeta $ruta no existe")
            end
        catch e
            println("Error al cargar desde $ruta: $e")
        end
    end
    
    if isempty(imagenes)
        error("No se pudieron cargar imágenes desde ninguna de las rutas probadas. Verifica la ubicación de la carpeta 'parasite-dataset'.")
    end
    
    # Mostrar información sobre las imágenes cargadas
    println("Número de imágenes cargadas: ", length(imagenes))
    println("Número de clases: ", length(clases))
    println("Distribución de clases:")
    for (i, clase) in enumerate(clases)
        numImagenes = count(==(clase), etiquetas)
        println("  Clase $clase ($(nombresClases[i])): $numImagenes imágenes")
    end
    
    # Dividir en entrenamiento, validación y test
    indicesTrain, indicesVal, indicesTest = dividirDatos(imagenes, etiquetas)
    
    # Separar datos
    imagenesTrain = [imagenes[i] for i in indicesTrain]
    etiquetasTrain = etiquetas[indicesTrain]
    imagenesVal = [imagenes[i] for i in indicesVal]
    etiquetasVal = etiquetas[indicesVal]
    imagenesTest = [imagenes[i] for i in indicesTest]
    etiquetasTest = etiquetas[indicesTest]
    
    # Convertir a formato WHCN
    println("Convirtiendo imágenes al formato WHCN...")
    trainImgs = convertirArrayImagenesWHCN(imagenesTrain, tamañoImagen)
    valImgs = convertirArrayImagenesWHCN(imagenesVal, tamañoImagen)
    testImgs = convertirArrayImagenesWHCN(imagenesTest, tamañoImagen)
    
    println("Tamaño de la matriz de entrenamiento: ", size(trainImgs))
    println("Tamaño de la matriz de validación: ", size(valImgs))
    println("Tamaño de la matriz de test: ", size(testImgs))
    
    # Valores mínimo y máximo
    println("Valores mínimo y máximo de las entradas: (", minimum(trainImgs), ", ", maximum(trainImgs), ")")
    
    # Seleccionar la arquitectura
    arquitectura, nombreArquitectura = seleccionarArquitectura(numArquitectura)
    
    if arquitectura === nothing
        return nothing, nothing, 0.0, 0.0, 0.0, 0.0
    end
    
    println("\n=== Evaluando arquitectura: $nombreArquitectura ===")
    
    # Entrenar y evaluar el modelo
    modelo, precision, matrizConfusion, accuracy, recall, precision, f1, _, historicoAccVal = 
        entrenarCNN(trainImgs, etiquetasTrain, valImgs, etiquetasVal, testImgs, etiquetasTest, clases, arquitectura)
    
    # Mostrar matriz de confusión
    println("\nMatriz de confusión:")
    display(matrizConfusion)
    
    # Mostrar métricas
    println("\nMétricas:")
    println("  Accuracy: ", round(accuracy * 100, digits=2), "%")
    println("  Recall: ", round(recall * 100, digits=2), "%")
    println("  Precision: ", round(precision * 100, digits=2), "%")
    println("  F1-Score: ", round(f1 * 100, digits=2), "%")
    
    # Guardar mapeo de clases
    open("parasite_class_mapping.txt", "w") do io
        println(io, "Índice,Nombre")
        for (i, nombre) in enumerate(nombresClases)
            println(io, "$i,$nombre")
        end
    end
    
    return modelo, matrizConfusion, accuracy, recall, precision, f1
end

# Script de ejecución interactiva para seleccionar arquitectura
function ejecutarSeleccionInteractiva()
    println("=== Selector de Arquitecturas CNN para Clasificación de Parásitos ===")
    println("Por favor, seleccione una arquitectura:")
    println("1: Arquitectura Simple")
    println("2: Arquitectura Profunda")
    println("3: Arquitectura Bottleneck")
    println("4: Arquitectura Ancha")
    println("0: Salir")

    println("Ingrese el número de la arquitectura deseada (0 para salir):")
    numArquitectura = parse(Int, readline())

    if numArquitectura == 0
        println("Saliendo del selector de arquitecturas.")
        return
    end

    if numArquitectura < 1 || numArquitectura > 5
        println("Número no válido. Por favor, seleccione un número entre 1 y 5.")
        return
    end


    # Llamar a la función principal con la arquitectura seleccionada
    modelo, matrizConfusion, accuracy, recall, precision, f1 = main(numArquitectura)
    if modelo !== nothing
        println("Modelo entrenado y evaluado con éxito.")
    else
        println("No se pudo entrenar el modelo.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    ejecutarSeleccionInteractiva()
end