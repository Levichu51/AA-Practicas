# Modelo 1: Redes Neuronales Artificiales
using Random
using Statistics
using DelimitedFiles

# Función principal para ejecutar experimentos con RNA
function ejecutarModeloRNA(inputs, targets, crossValidationIndices)
    println("=== Modelo 1: Redes Neuronales Artificiales ===")
    
    # Definir las topologías a probar
    topologies = [
        [3],           # Una capa oculta con 3 neuronas
        [5],           # Una capa oculta con 5 neuronas
        [7],           # Una capa oculta con 7 neuronas
        [10],          # Una capa oculta con 10 neuronas
        [3, 2],        # Dos capas ocultas: 3 y 2 neuronas
        [5, 3],        # Dos capas ocultas: 5 y 3 neuronas
        [7, 5],        # Dos capas ocultas: 7 y 5 neuronas
        [10, 5]        # Dos capas ocultas: 10 y 5 neuronas
    ]
    
    # Parámetros comunes para el entrenamiento (Tener solo 1 descomentado)
    learning_rate = 0.01; num_executions = 20; max_epochs = 500; min_loss = 0.1
    #learning_rate = 0.005; num_executions = 30; max_epochs = 500; min_loss = 0.1
    #learning_rate = 0.05; num_executions = 30; max_epochs = 500; min_loss = 0.05
    
    # Almacenar resultados
    results = []
    
    # Probar cada topología
    for topology in topologies
        println("========== Evaluando topología: ", topology , " ===========")
        
        try
            # Crear diccionario de hiperparámetros para esta topología
            hyperparameters = Dict(
                "topology" => topology,
                "numExecutions" => num_executions,
                "maxEpochs" => max_epochs,
                "minLoss" => min_loss,
                "learningRate" => learning_rate
            )
            
            # Realizar validación cruzada usando modelCrossValidation
            accuracy, error_rate, recall, specificity, precision, npv, f1, conf_matrix = 
                modelCrossValidation(
                    :ANN,                   # Tipo de modelo
                    hyperparameters,        # Hiperparámetros
                    (inputs, targets),      # Dataset
                    crossValidationIndices  # Índices de validación cruzada
                )

            println("Matriz de confusión de test:")
            display(conf_matrix)
            println()

            # Guardar resultados
            push!(results, (
                topology=topology,
                accuracy=accuracy,
                error_rate=error_rate,
                recall=recall,
                specificity=specificity,
                precision=precision,
                npv=npv,
                f1=f1,
                conf_matrix=conf_matrix
            ))
            
            # Mostrar resultados
            println("  Accuracy: ", accuracy[1], " ± ", accuracy[2])
            println("  Sensibilidad (Recall): ", recall[1], " ± ", recall[2])
            println("  VPP (Precision): ", precision[1], " ± ", precision[2])
            println("  F1-Score: ", f1[1], " ± ", f1[2])
            println()
        catch e
            println("  Error al evaluar topología ", topology, ": ", e)
            println("  Saltando a la siguiente topología...")
        end
    end
    
    # Encontrar la mejor topología basada en recall
    best_idx = argmax([r.recall[1] for r in results])
    best_topology = results[best_idx].topology
    best_recall = results[best_idx].recall
    
    println("========= Resultados Finales =========")
    println("Mejor topología (por recall): ", best_topology)
    println("Accuracy: ", results[best_idx].accuracy[1], " ± ", results[best_idx].accuracy[2])
    println("Sensibilidad (Recall): ", best_recall[1], " ± ", best_recall[2])
    println("VPP (Precision): ", results[best_idx].precision[1], " ± ", results[best_idx].precision[2])
    println("F1-Score: ", results[best_idx].f1[1], " ± ", results[best_idx].f1[2])
    
    return results, best_idx
end