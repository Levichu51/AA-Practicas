# Modelo 4: Support Vector Machines
using Random
using Statistics
using DelimitedFiles

# Función principal para ejecutar experimentos con SVM
function ejecutarModeloSVM(inputs, targets, crossValidationIndices)
    println("=== Modelo 4: Support Vector Machines ===")
    
    # Definir exactamente 8 configuraciones de hiperparámetros
    hyperparameter_configs = [
        # 1. Linear kernel con C pequeño
        Dict("name" => "1", "kernel" => "linear", "C" => 0.1),
        
        # 2. Linear kernel con C grande
        Dict("name" => "2", "kernel" => "linear", "C" => 100.0),
        
        # 3. RBF kernel con C pequeño y gamma por defecto
        Dict("name" => "3", "kernel" => "rbf", "C" => 0.1, "gamma" => 1.0 / size(inputs, 2)),
        
        # 4. RBF kernel con C grande y gamma por defecto
        Dict("name" => "4", "kernel" => "rbf", "C" => 100.0, "gamma" => 1.0 / size(inputs, 2)),
        
        # 5. Sigmoid kernel con C pequeño
        Dict("name" => "5", "kernel" => "sigmoid", "C" => 0.1, "gamma" => 1.0 / size(inputs, 2), "coef0" => 0.0),
        
        # 6. Sigmoid kernel con C grande
        Dict("name" => "6", "kernel" => "sigmoid", "C" => 10.0, "gamma" => 0.1, "coef0" => 1.0),

        # 7. Polynomial kernel con C pequeño
        Dict("name" => "7", "kernel" => "poly", "C" => 0.1, "degree" => 2, "gamma" => 1.0 / size(inputs, 2), "coef0" => 0.0),

        # 8. Polynomial kernel con C grande
        Dict("name" => "8", "kernel" => "poly", "C" => 10.0, "degree" => 3, "gamma" => 0.1, "coef0" => 1.0)
    ]
    
    # Almacenar resultados
    results = []
    
    # Probar cada configuración de hiperparámetros
    for config in hyperparameter_configs
        println("========== Evaluando: ", config["name"], " ===========")
        
        try
            # Realizar validación cruzada usando modelCrossValidation
            accuracy, error_rate, recall, specificity, precision, npv, f1, conf_matrix = 
                modelCrossValidation(
                    :SVC,                   # Tipo de modelo
                    config,                 # Hiperparámetros
                    (inputs, targets),      # Dataset
                    crossValidationIndices  # Índices de validación cruzada
                )

            println("Matriz de confusión de test:")
            display(conf_matrix)
            println()
            
            # Guardar resultados
            push!(results, (
                name=config["name"],
                kernel=config["kernel"],
                C=config["C"],
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
            println("  Error al evaluar configuración: ", e)
            println("  Saltando a la siguiente configuración...")
        end
    end
    
    # Encontrar la mejor configuración basada en recall
    best_idx = argmax([r.recall[1] for r in results])
    best_config = results[best_idx]
    
    println("========= Resultados Finales =========")
    println("Mejor configuración (por recall):")
    println("Configuración: ", best_config.name)
    println("Kernel: ", best_config.kernel)
    println("C: ", best_config.C)
    println("Accuracy: ", best_config.accuracy[1], " ± ", best_config.accuracy[2])
    println("Sensibilidad (Recall): ", best_config.recall[1], " ± ", best_config.recall[2])
    println("VPP (Precision): ", best_config.precision[1], " ± ", best_config.precision[2])
    println("F1-Score: ", best_config.f1[1], " ± ", best_config.f1[2])
    
    return results, best_idx
end