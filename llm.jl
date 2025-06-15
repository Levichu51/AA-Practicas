using Transformers
using Transformers.HuggingFace
using Transformers.TextEncoders
using Flux

#Cargar config
config = HuggingFace.load_config(
"BSC-LT/salamandra-2b-instruct")

#Modelo
modelType = HuggingFace.getconfigname(config)

#Tareas que soporta
HuggingFace.get_model_type(modelType)

#Cargar el codificador
textEncoder = HuggingFace.load_tokenizer(
"BSC-LT/salamandra-2b-instruct")

#Tamaño vocabulario
length(textEncoder.vocab)

#Token ocupa la palabra x
lookup(textEncoder.vocab, "▁aprendizaje")

#Que token es x num
lookup(textEncoder.vocab, 110929)

#Token ocupa palabra que no existe
lookup(textEncoder.vocab, "dvfdihvw387y32f34f")

#Token ocupa el inicio del texto
lookup(textEncoder.vocab, textEncoder.startsym)

#Token ocupa fin del texto
lookup(textEncoder.vocab, textEncoder.endsym)

#Token ocupa el relleno pad
lookup(textEncoder.vocab, textEncoder.padsym)

#Codificar un texto
tokens = encode(textEncoder,"En un lugar de la Mancha, de cuyo").token

isa(tokens, AbstractArray)

size(tokens)

length(tokens.onehots[1])

#Decodificar un token
decode(textEncoder, tokens[:,1])
decode(textEncoder, tokens[:,2])

decode(textEncoder, tokens.onehots[1])
decode(textEncoder, tokens.onehots[2])

#Deco un conjunto de tokens
decode(textEncoder, tokens)

#Vector de strings para concatenarlos
join(decode(textEncoder, tokens))


#Otras posibilidades
Flux.onecold(tokens)
Flux.onecold.(tokens.onehots)
findfirst.(tokens.onehots)

join(decode(textEncoder, Flux.onecold(tokens)))
join(decode(textEncoder,
Flux.onecold.(tokens.onehots)))



#Cargar Modelo
model = HuggingFace.load_model(
"BSC-LT/salamandra-2b-instruct",
:forcausallm;
config = config)

Base.summarysize(model)/(1024^3)


while true
#Aplicar Modelo
    input = (; token = tokens)
    outputs = model(input)

    #Logits
    size(outputs.logit)
    logits = outputs.logit[:, end, 1]

    #Otra manera
    logits = view(outputs.logit, :, size(outputs.logit,2), 1)

    #Logits a probabilidades
    probs = softmax(logits)

    #Palabra mas probable 
    newTokenId = argmax(probs)

    #Token al final de la cadena de tokens 
    push!(tokens.onehots, newTokenId)

    #Decode y print
    print(decode(textEncoder, newTokenId))

end;


#Para que genere textos distintos, selecciona aleatoriamente entre los tokens mas probables
ids = partialsortperm(probs, 1:k, rev=true)
highest_probs = view(probs, ids)
newTokenId = sample(
ids,
ProbabilityWeights(highest_probs))

#Convierte los logits en probabilidades
probs = softmax(logits ./ temperature)


#Criterio de parada
if newTokenId == textEncoder.endsym
end;

#Se podría parar de generar texto cuando el modelo devuelva “###” El modelo cree que la respuesta ha terminado y ahora tiene que generar la siguiente instrucción
if newTokenId == findfirst(lookup(textEncoder, "###")) ||
    newTokenId == findfirst(lookup(textEncoder, "_###"))
end;


#Para usar GPU
using CUDA

#Disp disponibles y seleccionar
CUDA.devices()
CUDA.device!(0)

#Que solo haga operaciones vectoriales
CUDA.allowscalar(false)

#Activar grafica, cargar modelo y pasarle el input a la grafica
enable_gpu(true)
model = todevice(model);
input = (; token = tokens) |> todevice






