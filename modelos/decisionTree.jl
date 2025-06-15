# Modelo 3: Árboles de Decisión
using Random
using Statistics
using DelimitedFiles

# Función principal para ejecutar experimentos con Árboles de Decisión
function ejecutarModeloArbolDecision(inputs, targets, crossValidationIndices)
    println("=== Modelo 3: Árboles de Decisión ===")
    
    # Definir las profundidades máximas a probar
    max_depths = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Almacenar resultados
    results = []
    
    # Probar cada profundidad máxima
    for depth in max_depths
        println("========== Evaluando profundidad máxima = ", depth, " ===========")
        
        try
            # Crear diccionario de hiperparámetros para esta profundidad
            hyperparameters = Dict(
                "max_depth" => depth
            )
            
            # Realizar validación cruzada usando modelCrossValidation
            accuracy, error_rate, recall, specificity, precision, npv, f1, conf_matrix = 
                modelCrossValidation(
                    :DecisionTreeClassifier,  # Tipo de modelo
                    hyperparameters,          # Hiperparámetros
                    (inputs, targets),        # Dataset
                    crossValidationIndices    # Índices de validación cruzada
                )
            
            # Imprimir la matriz de confusión para diagnóstico
            println("Matriz de confusión de test:")
            display(conf_matrix)
            println()
            
            # Guardar resultados
            push!(results, (
                depth=depth,
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
            println("  Error al evaluar profundidad = ", depth, ": ", e)
            println("  Saltando a la siguiente profundidad...")
        end
    end
    
    # Encontrar la mejor profundidad basada en recall
    best_idx = argmax([r.recall[1] for r in results])
    best_depth = results[best_idx].depth
    best_recall = results[best_idx].recall
    
    println("========= Resultados Finales =========")
    println("Mejor profundidad (por recall): ", best_depth)
    println("Accuracy: ", results[best_idx].accuracy[1], " ± ", results[best_idx].accuracy[2])
    println("Sensibilidad (Recall): ", best_recall[1], " ± ", best_recall[2])
    println("VPP (Precision): ", results[best_idx].precision[1], " ± ", results[best_idx].precision[2])
    println("F1-Score: ", results[best_idx].f1[1], " ± ", results[best_idx].f1[2])
    println("\n")

    return results, best_idx
end