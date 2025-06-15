# Modelo 5: DoME (Domain Mapping Estimator)
using Random
using Statistics
using DelimitedFiles

# Función principal para ejecutar experimentos con DoME
function ejecutarModeloDoME(inputs, targets, crossValidationIndices)
    println("=== Modelo 5: DoME (Domain Mapping Estimator) ===")
    
    # Definir los valores de número máximo de nodos a probar
    max_nodes_values = [15, 30, 45, 60, 75, 90, 105, 120]
    
    # Almacenar resultados
    results = []
    
    # Probar cada valor de número máximo de nodos
    for max_nodes in max_nodes_values
        println("========== Evaluando número máximo de nodos = ", max_nodes, " ===========")
        
        try
            # Crear diccionario de hiperparámetros para este valor de número máximo de nodos
            hyperparameters = Dict(
                "maximumNodes" => max_nodes
            )
            
            # Realizar validación cruzada usando modelCrossValidation
            accuracy, error_rate, recall, specificity, precision, npv, f1, conf_matrix = 
                modelCrossValidation(
                    :DoME,                  # Tipo de modelo (símbolo, no string)
                    hyperparameters,        # Hiperparámetros
                    (inputs, targets),      # Dataset
                    crossValidationIndices  # Índices de validación cruzada
                )
            
            println("Matriz de confusión de test:")
            display(conf_matrix)
            println()
            
            # Guardar resultados
            push!(results, (
                max_nodes=max_nodes,
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
            println("  Error al evaluar número máximo de nodos = ", max_nodes, ": ", e)
            println("  Saltando al siguiente valor...")
        end
    end
    
    # Encontrar el mejor valor de número máximo de nodos basado en recall
    best_idx = argmax([r.recall[1] for r in results])
    best_max_nodes = results[best_idx].max_nodes
    best_recall = results[best_idx].recall
    
    println("========= Resultados Finales =========")
    println("Mejor número máximo de nodos (por recall): ", best_max_nodes)
    println("Accuracy: ", results[best_idx].accuracy[1], " ± ", results[best_idx].accuracy[2])
    println("Sensibilidad (Recall): ", best_recall[1], " ± ", best_recall[2])
    println("VPP (Precision): ", results[best_idx].precision[1], " ± ", results[best_idx].precision[2])
    println("F1-Score: ", results[best_idx].f1[1], " ± ", results[best_idx].f1[2])
    println("\n")
    
    return results, best_idx
end