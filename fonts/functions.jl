# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

import Pkg;
using PyCall
using Statistics
using DelimitedFiles
using Statistics
using Flux
using Flux.Losses


function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    # Obtener el número de clases únicas
    uniqueClasses = unique(classes) #para comparar el numero, de la otra manera comparas vectores
    numClasses = length(uniqueClasses)
    numPatterns = length(feature)

    # Si solo hay dos clases, crear un vector booleano
    if numClasses <= 2
        # Compara cada elemento de feature con la primera clase y genera un vector booleano
        newMatrix = reshape(feature .== uniqueClasses[1], numPatterns, 1)
    else
        # Crear una matriz booleana de tamaño numClasses x numPattterns
        newMatrix = falses(numPatterns, numClasses)

        # Para cada clase, marcar en la matriz las posiciones correspondientes con un bucle for
        for i in 1:numClasses
            newMatrix[:, i] .= feature .== uniqueClasses[i]
        end
    end
    return newMatrix
end;

# Sobrecarga de la función para el caso en el que no se proporcionen las clases únicas
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

# Sobrecarga de la función para el caso en que feature es un vector de booleanos
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, size(feature, 1), 1);


# dims = 1 significa que lo va a hacer las operaciones por las filas (columnas)
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})

    # Hace los mínimos de cada una de las columnas
    min = minimum(dataset, dims=1)
    # Hace los máximos de cada una de las columnas
    max = maximum(dataset, dims=1)
    return (min, max)
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    # dims = 1 significa que lo va a hacer las operaciones por las filas (columnas)
    # Hace las medias de cada una de las columnas
    mean_value = mean(dataset, dims=1)
    # Hace las desviaciones tipicas de cada una de las columnas
    std_value = std(dataset, dims=1)
    return (mean_value, std_value)
end;


function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    # Hace la normalización min-max
    #Nota: hay que ver el caso en el que se divide por 0?
    dataset[:, vec(normalizationParameters[1] .== normalizationParameters[2])] .= 0;
    dataset .= (dataset .- normalizationParameters[1]) ./ (normalizationParameters[2] .- normalizationParameters[1])

    return dataset
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    # Hace la normalización min-max
    normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))

    return dataset
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    aux = copy(dataset)
    normalizeMinMax!(aux, normalizationParameters)

    return aux
end;

normalizeMinMax(dataset::AbstractArray{<:Real,2}) = normalizeMinMax(dataset, calculateMinMaxNormalizationParameters(dataset));



function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    safeStd = ifelse.(normalizationParameters[2] .== 0, one(eltype(dataset)), normalizationParameters[2]) #evita divisiones entre 0
    dataset .= (dataset .- normalizationParameters[1]) ./ safeStd
end;

normalizeZeroMean!(dataset::AbstractArray{<:Real,2}) = normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset));

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    aux = copy(dataset)
    normalizeZeroMean!(aux, normalizationParameters)

    return aux
end;

normalizeZeroMean(dataset::AbstractArray{<:Real,2}) = normalizeZeroMean(dataset, calculateZeroMeanNormalizationParameters(dataset));



function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs .>= threshold
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numColumns = size(outputs, 2)

    if (numColumns == 1)
        vector = classifyOutputs(outputs[:], threshold=threshold)

        # Reshape para ponerlo en columnas, se usa outputs para obtener las filas originales
        return reshape(vector, size(outputs, 1), 1)
    else
        # Saber en que columna esta el maximo de cada patron
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)

        # Creas matriz booleana inicializada a falses
        outputs = falses(size(outputs))

        # Pones a true en la columna que corresponde a cada patron
        outputs[indicesMaxEachInstance] .= true

    end

    return outputs

end;

# Accuracy para vectores booleanos
accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs .== targets);

# Para matrices bidimensionales de valores booleanos
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if (size(outputs, 2) == 1 && size(targets, 2) == 1)
        return accuracy(outputs[:, 1], targets[:, 1])

    else

        # Saber los valores i,j son iguales en las matrices
        trueValues = outputs .== targets

        # Matriz de filas tienen todos los valores a true y cuales no cumplen eso
        rowMatching = all(trueValues, dims=2)

        # Se calcula la media de los valores true
        return mean(rowMatching)

    end


end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = outputs .>= threshold
    return accuracy(outputs, targets)
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    
    if (size(outputs, 2) == 1 && size(targets, 2) == 1)
        outputs_tar = outputs .>= threshold
        return accuracy(outputs_tar[:, 1], targets[:, 1])

    else

        return accuracy(classifyOutputs(outputs), targets)
    end
end;

# Problemas de clasificación
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann = Chain()

    numInputsLayer = numInputs
    for (i, numOutputsLayer) = enumerate(topology)
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]))
        numInputsLayer = numOutputsLayer
    end

    if numOutputs > 2
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
    end
    
    return ann
end;


function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    X, Y = dataset
    X = Float32.(X)'
    Y = Float32.(Y)'

    numInputs, numOutputs = size(X, 1), size(Y, 1)

    ann = buildClassANN(numInputs, topology, numOutputs, transferFunctions=transferFunctions)

    # Define the loss function
    function loss(model, x, y)
        ŷ = model(x)
        return Flux.logitcrossentropy(ŷ, y)
    end

    # Create an optimizer
    opt = Flux.setup(Flux.Optimise.Descent(learningRate), ann)

    # Prepare training
    losses = Float32[]
    push!(losses, loss(ann, X, Y))

    # Training loop
    for epoch in 1:maxEpochs
        grads = Flux.gradient(model -> loss(model, X, Y), ann)
        Flux.update!(opt, ann, grads[1])
        
        current_loss = loss(ann, X, Y)
        push!(losses, current_loss)

        if current_loss < minLoss
            break
        end
    end

    return ann, losses
end

#Misma funcion pero con un vector target convertido a matriz de una columna
function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    targetVector = reshape(targets, (:,1))

    return trainClassANN(topology, (inputs, targetVector), transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)

end;

# # Se debería de probar los métodos train para comprobar lo que se propone en la pagina 17

# # ~Parámetros base para el experimento~
# maxEpochs = 1000;           # Número máximo de épocas de entrenamiento

# # Definición de los conjuntos de parámetros a probar:
# topology = [5];      # Ejemplos: una capa oculta de 5 neuronas, una sola de 10 y dos capas con 5 y 3 neuronas
# learningRates = 0.01;           # Tasas de aprendizaje a probar (dentro del rango 0.001 - 0.1)
# normalizationOptions = [true];   # Probar con datos normalizados y sin normalizar

# # ----------------------------------------------------------------------------------------------
# # Cargamos la base de datos.
# dataset = readdlm("iris.data",',');

# # Separamos las entradas y las salidas deseadas.
# inputs = dataset[:,1:4];
# targets = dataset[:,5];

# # Convertimos los datos de entrada a Array{Float32,2} y procesamos los targets.
# inputs = Float32.(inputs);                      # Convertir a Float32
# targets = oneHotEncoding(targets);              # Aplicar one-hot encoding
# targets = Bool.(targets);                       # Convertir a booleanos

# @assert size(inputs,1) == size(targets,1) "Error: Las matrices de entradas y salidas no tienen el mismo número de filas";

# normalizeZeroMean!(inputs);                     # Normalizar las entradas

# # Entrenamiento y creación de la RNA
# (trainedANN, lossValues) = trainClassANN(topology, (inputs, targets), maxEpochs = maxEpochs, learningRate = learningRates);

# # Salidas de la RNA
# outputs = trainedANN(inputs');
# outputs = outputs';

# accuracySet = accuracy(outputs, targets);





# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    permutedIndexes = randperm(N)

    numTest = round(Int, N * P)

    testIndex = permutedIndexes[1:numTest]
    trainIndex = permutedIndexes[(numTest + 1):end]

    return (trainIndex, testIndex)
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    #Separo los indices de entrenamiento y validacion
    (trainvalIndex, testIndex) = holdOut(N, Ptest)
    NewPval = Pval/(1-Ptest)
    (trainIndex, valIndex) = holdOut(length(trainvalIndex), NewPval)
    
    #Seleccion de los indices
    trainIndex = trainvalIndex[trainIndex]
    valIndex = trainvalIndex[valIndex]

    return (trainIndex, valIndex, testIndex)
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, size(trainingDataset[1], 2)), falses(0, size(trainingDataset[2], 2))),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, size(trainingDataset[1], 2)), falses(0, size(trainingDataset[2], 2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    train_inputs = convert(Array{Float32,2}, trainingDataset[1])'
    train_targets = convert(Array{Float32,2}, trainingDataset[2])'
    val_inputs = convert(Array{Float32,2}, validationDataset[1])'
    val_targets = convert(Array{Float32,2}, validationDataset[2])'
    test_inputs = convert(Array{Float32,2}, testDataset[1])'
    test_targets = convert(Array{Float32,2}, testDataset[2])'

    ann = buildClassANN(size(train_inputs, 1), topology, size(train_targets, 1), transferFunctions = transferFunctions)

    loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

    opt_state = Flux.setup(Adam(learningRate), ann)

	trainingLosses = Float32[loss(ann, train_inputs, train_targets)]
    validationLosses = Float32[]
    testLosses = Float32[]

	if !isempty(val_inputs)
		validationLosses = [loss(ann, val_inputs, val_targets)]
    end

    if !isempty(test_inputs)
        testLosses = [loss(ann, test_inputs, test_targets)]
	end

	numEpoch = 0
	numEpochVal = 0

	bestAnn = deepcopy(ann)

    bestValidationLoss = isempty(val_inputs) ? nothing : validationLosses[end]

    for numEpoch in 1:maxEpochs
        Flux.train!(loss, ann, [(train_inputs, train_targets)], opt_state)
    
        currentTrainingLoss = loss(ann, train_inputs, train_targets)
        push!(trainingLosses, currentTrainingLoss)
    
        if !isempty(val_inputs)
            currentValidationLoss = loss(ann, val_inputs, val_targets)
            push!(validationLosses, currentValidationLoss)
        end
    
        if !isempty(test_inputs)
            currentTestLoss = loss(ann, test_inputs, test_targets)
            push!(testLosses, currentTestLoss)
        end
    
        if !isempty(val_inputs)
            if validationLosses[end] < bestValidationLoss
                bestValidationLoss = validationLosses[end]
                bestAnn = deepcopy(ann)
                numEpochVal = 0  
            else
                numEpochVal += 1
            end
    
            if numEpochVal >= maxEpochsVal
                break
            end
        end
    
        if currentTrainingLoss <= minLoss
            break
        end
    end
    

    if isempty(val_inputs) 
        return ann, trainingLosses, validationLosses, testLosses
    else 
        return bestAnn, trainingLosses, validationLosses, testLosses
    end
end


function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, size(trainingDataset[1], 2)), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef, 0, size(trainingDataset[1], 2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    
    y_train = reshape(trainingDataset[2], :, 1)

    valDataset = isempty(validationDataset[1]) ? (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,1)) :
                  (validationDataset[1], reshape(validationDataset[2], :, 1))
    
    testDataset = isempty(testDataset[1]) ? (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,1)) :
                  (testDataset[1], reshape(testDataset[2], :, 1))

    return trainClassANN(topology, (trainingDataset[1], y_train),
                         validationDataset=valDataset, testDataset=testDataset,
                         transferFunctions=transferFunctions, maxEpochs=maxEpochs,
                         minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)
end



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


#4.1

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    vn = sum((.!outputs) .& (.!targets))
    vp = sum(outputs .& targets)
    fn = sum((.!outputs) .& targets)
    fp = sum(outputs .& (.!targets))

    total = length(targets);
    accuracy = (vn + vp) / total;
    error = 1. - accuracy;
    recall = vp / (vp + fn);
    specificity = vn / (vn + fp);
    precision = vp / (vp + fp);
    negativePredictiveValue = vn / (vn + fn);

    #Casos particulares

    #Si no hay positivos
    if(vp + fn) == 0
        recall = 1.0;
    end

    #Si no hay negativos
    if(vn + fp) == 0
        specificity = 1.0;
    end

    #Si no hay positivos predichos
    if(vp + fp) == 0
        precision = 1.0;
    end

    #Si no hay negativos predichos
    if(vn + fn) == 0
        negativePredictiveValue = 1.0;
    end

    #f1Score si se divide por 0
    recall = isnan(recall) ? 0.0 : recall;
    specificity = isnan(specificity) ? 0.0 : specificity;
    precision = isnan(precision) ? 0.0 : precision;
    negativePredictiveValue = isnan(negativePredictiveValue) ? 0.0 : negativePredictiveValue;

    if(precision + recall) == 0
        f1Score = 0.0;

    else
        f1Score = (2 * (precision * recall)) / (precision + recall);
    end

    #La matrix de confusion
    #confusion_matrix = Array{Int,64}(undef, 2, 2);
    confusion_matrix = [vn fp; fn vp];


    return (accuracy, error, recall, specificity, precision, negativePredictiveValue, f1Score, confusion_matrix);

end;


function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    bool_outputs = outputs.>threshold;
    return confusionMatrix(bool_outputs, targets);

end;


function printConfusionMatrix(outputs::AbstractArray{Bool,1}, 
    targets::AbstractArray{Bool,1})

    (accuracy, error, recall, specificity, precision, negativePredictiveValue, f1Score, confusion_matrix) = confusionMatrix(outputs, targets);

    println("Métricas de Clasificación:")
    println("Precisión (Accuracy): ", accuracy)
    println("Tasa de Error (Error Rate): ", error)
    println("Sensibilidad (Recall): ", recall)
    println("Especificidad: ", specificity)
    println("Valor Predictivo Positivo (Precision): ", precision)
    println("Valor Predictivo Negativo (NPV): ", negativePredictiveValue)
    println("F1-score: ", f1Score)
    println("Matriz de Confusión:")
    println(confusion_matrix)

end;


function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, 
    targets::AbstractArray{Bool,1}; threshold::Real=0.5)

    bool_outputs = outputs .> threshold;
    printConfusionMatrix(bool_outputs, targets);
end;


#4.2
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    classesCnt = size(outputs, 2);
    @assert(classesCnt == size(targets,2));
    @assert(classesCnt != 2);

    if (classesCnt == 1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    end
    
    recall = zeros(classesCnt);
    specificity = zeros(classesCnt);
    precision = zeros(classesCnt);
    negativePredictiveValue = zeros(classesCnt);
    f1Score = zeros(classesCnt);

    for i in 1:classesCnt
        (_, _, r, s, p, n, f, _) = confusionMatrix(outputs[:,i], targets[:,i]);
        recall[i] = r;
        specificity[i] = s;
        precision[i] = p;
        negativePredictiveValue[i] = n;
        f1Score[i] = f;
    end

    confusion_mtx = [sum(targets[:, i] .& outputs[:, j]) for i in 1:classesCnt, j in 1:classesCnt]
    patternsEachClass = vec(sum(targets, dims=1));

    if (weighted)
        weight = patternsEachClass ./ sum(patternsEachClass);

        recall = sum(weight .* recall);
        specificity = sum(weight .* specificity);
        precision = sum(weight .* precision);
        negativePredictiveValue = sum(weight .* negativePredictiveValue);
        f1Score = sum(weight .* f1Score);

    else
        nomEmpty = sum(patternsEachClass .> 0);

        recall = sum(recall) / nomEmpty;
        specificity = sum(specificity) / nomEmpty;
        precision = sum(precision) / nomEmpty;
        negativePredictiveValue = sum(negativePredictiveValue) / nomEmpty;
        f1Score = sum(f1Score) / nomEmpty;

    end

    a = accuracy(outputs, targets);
    errorRate = 1. - a;

    return(a, errorRate, recall, specificity, precision, negativePredictiveValue, f1Score, confusion_mtx);
end;


function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    return confusionMatrix(classifyOutputs(outputs, threshold=threshold), targets, weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert all(in.(outputs, Ref(classes))) && all(in.(targets, Ref(classes))) "Todas las etiquetas deben estar en el conjunto de clases."
    
    function oneHotEncoding(labels, classes)
        return hcat([labels .== c for c in classes]...)
    end
    
    outputs_one_hot = oneHotEncoding(outputs, classes)
    targets_one_hot = oneHotEncoding(targets, classes)
    
    return confusionMatrix(outputs_one_hot, targets_one_hot; weighted=weighted)
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(outputs, targets))  # Calcula automáticamente las clases únicas
    return confusionMatrix(outputs, targets, classes; weighted=weighted)
end;

using SymDoME


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #Coger la tupla
    trainingInputs, trainingTargets = trainingDataset;
    
    #Pasar a float las entradas de entrenamiento y tests
    trainingInputs = Float64.(trainingInputs);
    testInputs = Float64.(testInputs);

    _, _, _, model = dome(trainingInputs, trainingTargets; maximumNodes = maximumNodes);

    return evaluateTree(model, testInputs);

end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #Coger la tupla
    trainingInputs, trainingTargets = trainingDataset;
    numClasses = size(trainingTargets, 2);

    if numClasses == 1
        binaryTargets = vec(trainingTargets);
        
        trainingDome = trainClassDoME((trainingInputs, binaryTargets), testInputs, maximumNodes);
        
        return reshape(trainingDome,:, 1);
    end

    #Mas de una columna
    testSize = size(testInputs, 1);
    finalMatrix = Array{Float64,2}(undef, testSize, numClasses);

    for i in 1:numClasses
        binaryTargets = vec(trainingTargets[:,i]);
        trainingDome = trainClassDoME((trainingInputs, binaryTargets), testInputs, maximumNodes);
        finalMatrix[:,i] = trainingDome;
    end

    return finalMatrix;

end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #Coger la tupla
    trainingInputs, trainingTargets = trainingDataset;

    classes = unique(trainingTargets);
    testOutputs = Array{eltype(trainingTargets), 1}(undef, size(testInputs, 1));

    testOutputsDoME = trainClassDoME((trainingInputs, oneHotEncoding(trainingTargets, classes)),testInputs, maximumNodes);

    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0);

    if(length(classes) <= 2)
        testOutputsBool = vec(testOutputsBool);
        testOutputs[testOutputsBool] .= classes[1];

        if(length(classes) == 2)
            testOutputs[.!testOutputsBool] .= classes[2];
        end
    
    else
        for numClass in 1:length(classes)
            testOutputs[testOutputsBool[:,numClass]] .= classes[numClass];
        end
    end

    return testOutputs;


end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random: seed!

#En la validacion cruzada se divide en k subcojuntos el conjunto de datos
#Unos van para el conjunto de entrenamiento y otro para los del test garatinzando que todos los datos vayan a los dos subcojuntos
#Se repite el proceso varias veces para promediar el resultado y obtener una evaluacion robusta


function crossvalidation(N::Int64, k::Int64)
    #1. Ordenar de 1 a k
    kVec = collect(1:k);

    #2. Crear un vector nuevo con repeticiones de este vector hasta que la longitud sea mayor o igual a N
    reps = Int64(ceil(N/k));
    nVec = repeat(kVec, reps);

    #3. De este vector, tomar los N primeros valores.
    nVec = nVec[1:N];

    #4. Desordenar
    return shuffle!(nVec);

end;


function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    #1.
    index = zeros(Int64, length(targets));

    #Pillas indices pos y neg
    posIndex = findall(targets);
    negIndex = findall(.!targets);

    #Mirar si hay < k patrones

    kPositive = length(posIndex);
    kNegative = length(negIndex);

    @assert kPositive >= k "Error, menos de k patrones positivos"
    @assert kNegative >= k "Error, menos de k patrones negativos"
    
    index[posIndex] .= crossvalidation(kPositive, k);
    index[negIndex] .= crossvalidation(kNegative, k);

    return index;

end;


function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #1.
    N = size(targets, 1);
    numClasses = size(targets, 2);
    index = zeros(Int64, N);

    #2.
    for i in 1:numClasses 
        sample = sum(targets[:,i]);
        @assert sample >= k "Error, menos de k patrones"
        index[findall(targets[:,i])] .= crossvalidation(sample, k);
    end

    return index;

end;


function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    #Formato one-hot
    boolTargets = oneHotEncoding(targets);

    return crossvalidation(boolTargets, k);

end;


function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
    validationRatio::Real=0, maxEpochsVal::Int=20)

    inputs, targets = dataset

    # Calculate the number of folds
    numFolds = maximum(crossValidationIndices)

    # Calculate unique classes
    classes = unique(targets)
    
    # Perform one-hot encoding
    oneHotTargets = oneHotEncoding(targets, classes)
    
    # Get the number of classes
    numClasses = size(oneHotTargets, 2)

    # Initialize arrays for metrics
    accuracy = zeros(numFolds)
    errorRate = zeros(numFolds)
    recall = zeros(numFolds)
    specificity = zeros(numFolds)
    precision = zeros(numFolds)
    negativePredictiveValue = zeros(numFolds)
    F1s = zeros(numFolds)
    
    # Initialize confusion matrix
    confMatrix = zeros(numClasses, numClasses)

    # Loop through each fold
    for fold in 1:numFolds
        # Split data into training and test sets
        testIndexes = findall(crossValidationIndices .== fold)
        trainIndexes = findall(crossValidationIndices .!= fold)

        trainInputs = inputs[trainIndexes, :]
        trainTargets = oneHotTargets[trainIndexes, :]
        testInputs = inputs[testIndexes, :]
        testTargets = oneHotTargets[testIndexes, :]

        # Initialize arrays for metrics for each execution
        foldAccuracy = zeros(numExecutions)
        foldErrorRate = zeros(numExecutions)
        foldRecall = zeros(numExecutions)
        foldSpecificity = zeros(numExecutions)
        foldPrecision = zeros(numExecutions)
        foldNPV = zeros(numExecutions)
        foldF1s = zeros(numExecutions)
        
        # Initialize confusion matrix for each execution
        foldMatrix = zeros(numClasses, numClasses, numExecutions)

        # Loop through each execution
        for exec in 1:numExecutions
            # If validation ratio > 0, create validation set
            if validationRatio > 0
                trainSize = size(trainInputs, 1)
                
                # Calculate validation ratio for holdOut
                validationRatio2 = validationRatio * numFolds / (numFolds - 1)
                
                # Use holdOut to split training data into training and validation sets
                trainIndices, valIndices = holdOut(trainSize, validationRatio2)

                validationInputs = trainInputs[valIndices, :]
                validationTargets = trainTargets[valIndices, :]
                trainInputs2 = trainInputs[trainIndices, :]
                trainTargets2 = trainTargets[trainIndices, :]

                # Train the neural network with validation
                bestAnn, _, _, _ = trainClassANN(topology, (trainInputs2, trainTargets2),
                    validationDataset=(validationInputs, validationTargets),
                    transferFunctions=transferFunctions, maxEpochs=maxEpochs,
                    minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)
            else
                # Train the neural network without validation
                bestAnn, _, _, _ = trainClassANN(topology, (trainInputs, trainTargets),
                    transferFunctions=transferFunctions, maxEpochs=maxEpochs,
                    minLoss=minLoss, learningRate=learningRate)
            end

            # Get outputs from the neural network
            outputs = collect(bestAnn(testInputs')')

            # Calculate metrics
            if numClasses == 2
                # Binary classification
                outputs = vec(outputs)
                testTargets2 = vec(testTargets)
                foldAccuracy[exec], foldErrorRate[exec], foldRecall[exec], foldSpecificity[exec], foldPrecision[exec],
                    foldNPV[exec], foldF1s[exec], foldMatrix[:, :, exec] = confusionMatrix(outputs, testTargets2)
            else
                # Multi-class classification
                foldAccuracy[exec], foldErrorRate[exec], foldRecall[exec], foldSpecificity[exec], foldPrecision[exec],
                    foldNPV[exec], foldF1s[exec], foldMatrix[:, :, exec] = confusionMatrix(outputs, testTargets)
            end
        end

        # Calculate mean metrics for the fold
        accuracy[fold] = mean(foldAccuracy)
        errorRate[fold] = mean(foldErrorRate)
        recall[fold] = mean(foldRecall)
        specificity[fold] = mean(foldSpecificity)
        precision[fold] = mean(foldPrecision)
        negativePredictiveValue[fold] = mean(foldNPV)
        F1s[fold] = mean(foldF1s)

        # Calculate mean confusion matrix for the fold
        foldMeanMatrix = dropdims(mean(foldMatrix, dims=3), dims=3)
        
        # Add to global confusion matrix
        confMatrix += foldMeanMatrix
    end

    # Calculate overall mean and standard deviation for each metric
    accuracy_mean = mean(accuracy)
    println(accuracy_mean)
    accuracy_std = std(accuracy)
    println(accuracy_std)
    errorRate_mean = mean(errorRate)
    errorRate_std = std(errorRate)
    recall_mean = mean(recall)
    recall_std = std(recall)
    specificity_mean = mean(specificity)
    specificity_std = std(specificity)
    precision_mean = mean(precision)
    precision_std = std(precision)
    npv_mean = mean(negativePredictiveValue)
    npv_std = std(negativePredictiveValue)
    f1_mean = mean(F1s)
    f1_std = std(F1s)

    # Return metrics and confusion matrix
    return (accuracy_mean, accuracy_std), 
           (errorRate_mean, errorRate_std),
           (recall_mean, recall_std), 
           (specificity_mean, specificity_std),
           (precision_mean, precision_std), 
           (npv_mean, npv_std), 
           (f1_mean, f1_std), 
           confMatrix
end;





# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})
    
    if modelType == :ANN
        return ANNCrossValidation(
            modelHyperparameters["topology"], 
            dataset, 
            crossValidationIndices, 
            learningRate = get(modelHyperparameters, "learningRate", 0.01),
            numExecutions = get(modelHyperparameters, "numExecutions", 50),
            maxEpochs = get(modelHyperparameters, "maxEpochs", 1000),
            validationRatio = get(modelHyperparameters, "validationRatio", 0),
            maxEpochsVal = get(modelHyperparameters, "maxEpochsVal", 20)
        )
    end
    
    # Extraer entradas y targets del dataset
    inputs, rawTargets = dataset
    targetsString = string.(rawTargets)
    classes = unique(targetsString)
    
    accuracyVec = Float64[]
    errorRateVec = Float64[]
    recallVec = Float64[]
    specificityVec = Float64[]
    vppVec = Float64[]
    vpnVec = Float64[]
    f1Vec = Float64[]
    
    numClasses = length(classes)
    globalMatrix = zeros(Float64, numClasses, numClasses)
    
    numFolds = maximum(crossValidationIndices)
    
    for fold in 1:numFolds
        trainIndex = findall(crossValidationIndices .!= fold)
        testIndex  = findall(crossValidationIndices .== fold)
    
        trainInputs  = inputs[trainIndex, :]
        trainTargets = targetsString[trainIndex]    # targets extraídos del dataset
        testInputs   = inputs[testIndex, :]
        testTargets  = targetsString[testIndex]
    
        testPrediction = nothing
        if modelType == :DoME
            testPrediction = trainClassDoME((trainInputs, trainTargets), testInputs, modelHyperparameters["maximumNodes"])
        elseif modelType == :SVC
            C = modelHyperparameters["C"]
            kernel_str = modelHyperparameters["kernel"]
            if kernel_str == "linear"
                model = SVMClassifier(kernel=LIBSVM.Kernel.Linear, cost=Float64(C))
            elseif kernel_str == "rbf"
                model = SVMClassifier(kernel=LIBSVM.Kernel.RadialBasis, 
                    cost=Float64(C), 
                    gamma=Float64(get(modelHyperparameters, "gamma", 0.01)))
            elseif kernel_str == "sigmoid"
                model = SVMClassifier(kernel=LIBSVM.Kernel.Sigmoid, 
                    cost=Float64(C), 
                    gamma=Float64(get(modelHyperparameters, "gamma", 0.01)),
                    coef0=Float64(get(modelHyperparameters, "coef0", 0)))
            elseif kernel_str == "poly"
                model = SVMClassifier(kernel=LIBSVM.Kernel.Polynomial, cost=Float64(C),
                    gamma=Float64(get(modelHyperparameters, "gamma", 0.01)),
                    coef0=Float64(get(modelHyperparameters, "coef0", 0)),
                    degree=Int32(get(modelHyperparameters, "degree", 3)))
            end
    
            # Se crea el objeto machine usando trainTargets extraído directamente
            mach = machine(model, MLJ.table(trainInputs), categorical(trainTargets))
            MLJ.fit!(mach, verbosity=0)
            testPrediction = MLJ.predict(mach, MLJ.table(testInputs))
    
        elseif modelType == :DecisionTreeClassifier
            model = DTClassifier(max_depth=modelHyperparameters["max_depth"], rng=Random.MersenneTwister(1))
            mach = machine(model, MLJ.table(trainInputs), categorical(trainTargets))
            MLJ.fit!(mach, verbosity=0)
            testPrediction = mode.(MLJ.predict(mach, MLJ.table(testInputs)))
        elseif modelType == :KNeighborsClassifier
            model = kNNClassifier(K=modelHyperparameters["n_neighbors"])
            mach = machine(model, MLJ.table(trainInputs), categorical(trainTargets))
            MLJ.fit!(mach, verbosity=0)
            testPrediction = mode.(MLJ.predict(mach, MLJ.table(testInputs)))
        end 
    
        # Convertir los targets de test a String para que coincida con la codificación
        testTargetsString = string.(testTargets)
    
        (acc, err, rec, spec, prec, npv, f1, cm) = confusionMatrix(testPrediction, testTargetsString, classes)
    
        push!(accuracyVec, acc)
        push!(errorRateVec, err)
        push!(recallVec, rec)
        push!(specificityVec, spec)
        push!(vppVec, prec)
        push!(vpnVec, npv)
        push!(f1Vec, f1)
    
        globalMatrix += cm
    end
    
    return ((mean(accuracyVec), std(accuracyVec)),
            (mean(errorRateVec), std(errorRateVec)),
            (mean(recallVec), std(recallVec)),
            (mean(specificityVec), std(specificityVec)),
            (mean(vppVec), std(vppVec)),
            (mean(vpnVec), std(vpnVec)),
            (mean(f1Vec), std(f1Vec)),
            globalMatrix)
    

end;