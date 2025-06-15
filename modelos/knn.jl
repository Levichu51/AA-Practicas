# Modelo 2: k-Nearest Neighbors
using Random
using Statistics
using DelimitedFiles

# Función principal para ejecutar experimentos con kNN
function ejecutarModeloKNN(inputs, targets, crossValidationIndices)
    println("=== Modelo 2: k-Nearest Neighbors ===")
    
    # Definir los valores de k a probar
    # Usamos valores impares para evitar empates en la votación
    k_values = [
        1, 
        3, 
        5, 
        7, 
        9, 
        11, 
        13, 
        15
    ]
    
    # Almacenar resultados
    results = []
    
    # Probar cada valor de k
    for k in k_values
        println("========== Evaluando k = ", k, " ===========")
        
        try
            # Crear diccionario de hiperparámetros para este valor de k
            hyperparameters = Dict(
                "n_neighbors" => k
            )
            
            # Realizar validación cruzada usando modelCrossValidation
            accuracy, error_rate, recall, specificity, precision, npv, f1, conf_matrix = 
                modelCrossValidation(
                    :KNeighborsClassifier,  # Tipo de modelo
                    hyperparameters,        # Hiperparámetros
                    (inputs, targets),      # Dataset
                    crossValidationIndices  # Índices de validación cruzada
                )

            println("Matriz de confusión de test:")
            display(conf_matrix)
            println()
            
            # Guardar resultados
            push!(results, (
                k=k,
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
            println("  Error al evaluar k = ", k, ": ", e)
            println("  Saltando al siguiente valor de k...")
        end
    end
    
    # Encontrar el mejor valor de k basado en recall
    best_idx = argmax([r.recall[1] for r in results])
    best_k = results[best_idx].k
    best_recall = results[best_idx].recall
    
    println("========= Resultados Finales =========")
    println("Mejor valor de k (por recall): ", best_k)
    println("Accuracy: ", results[best_idx].accuracy[1], " ± ", results[best_idx].accuracy[2])
    println("Sensibilidad (Recall): ", best_recall[1], " ± ", best_recall[2])
    println("VPP (Precision): ", results[best_idx].precision[1], " ± ", results[best_idx].precision[2])
    println("F1-Score: ", results[best_idx].f1[1], " ± ", results[best_idx].f1[2])
    println("\n")
    
    return results, best_idx
end