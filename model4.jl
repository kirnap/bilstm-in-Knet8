# Only forward lstm implementation
# param[2k-1] = weight, param[2k] = bias for the kth layer
# param[end] = embedding 
function initparams(atype, layerconfig, embedsize, vocab, winit)
    parameters = Array(Any, 2*length(layerconfig) + 3)
    parameters[end] = winit * randn(vocab, embedsize) # embedding
    parameters[end-1] = winit * randn(layerconfig[end], vocab) # final layer weight
    parameters[end-2] = zeros(1, vocab)

    # lstm parameters
    input = embedsize
    for k = 1:length(layerconfig)
        parameters[2k-1] = winit * randn(input+layerconfig[k], 4*layerconfig[k])
        parameters[2k] = zeros(1, 4*layerconfig[k])
        parameters[2k][1:layerconfig[k]] = 1 # forget gate bias
        input = layerconfig[k]
    end
    return map(p->convert(atype, p), parameters)
end


# state[2k-1,2k]: hidden and cell for the k'th lstm layer
function initstates(atype, hidden, batchsize)
    state = Array(Any, 2*length(hidden))
    for k = 1:length(hidden)
        state[2k-1] = zeros(batchsize,hidden[k])
        state[2k] = zeros(batchsize,hidden[k])
    end
    return map(s->convert(atype,s), state)
end


function lstm(weight, bias, hidden, cell, input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end


# multi layer lstm forward function
function forward(parameters, states, input)
    x = input
    for i=1:2:length(states)
        (states[i], states[i+1]) = lstm(parameters[i], parameters[i+1], states[i], states[i+1], x)
        x = states[i]
    end
    return x
end

# read the input from 1 to n-1
# first hidden layer is zero corresponding <s> prediction
function loss(parameters, states, sequence)
    total = 0.0
    count = 0
    atype = typeof(AutoGrad.getval(parameters[1]))
    hlayers = length(states) / 2
    hlayers = convert(Int, hlayers)

    # forward lstm
    fhiddens = Array(Any, length(sequence))
    t1 = states[1:end]
    for i=1:length(sequence)-1
        input = convert(atype, sequence[i])
        x = input * parameters[end]
        fhiddens[i+1] = forward(parameters[1:2*hlayers], t1, x)
    end
    fhiddens[1] = convert(atype, zeros(size(fhiddens[2])))

    # merge layer
    for i=1:length(fhiddens)
        ypred = fhiddens[i] * parameters[end-1] .+ parameters[end-2]
        ynorm = logp(ypred, 2)
        ygold = convert(atype, sequence[i])
        count += size(ygold, 1)
        total += sum(ygold .* ynorm)
    end
    return - total / count
end

lossgradient = grad(loss)
