# 1:2*length(layerconfig) -> forward lstm
# 2*lenghth(layerconfig)+1:4*length(layerconfig) -> bacward lstm
# final 3 or 4 is embedding and final layer prediction parameters
function initparams(atype, layerconfig, embedsize, vocab, winit; single_embedding=false)
    if !single_embedding
        parameters = Array(Any, 4*length(layerconfig)+4)
        parameters[end] = winit * randn(vocab, embedsize) # backward lstm embedding
        parameters[end-1] = winit * randn(vocab, embedsize) # forward lstm embedding
        parameters[end-2] =  winit * randn(layerconfig[end]*2, vocab)# final layer weight
        parameters[end-3] = zeros(1, vocab) # final layer bias
    else
        parameters = Array(Any, 4*length(layerconfig)+3)
        parameters[end] = winit * randn(vocab, embedsize) # embedding shared among forward and backward lstm
        parameters[end-1] =  winit * randn(layerconfig[end], vocab)# final layer weight
        parameters[end-2] = zeros(1, vocab) # final layer bias
    end

    len = length(layerconfig)
    # forward layer parameters
    input = embedsize
    for k=1:len
        parameters[2k-1] = winit * randn(input+layerconfig[k], 4*layerconfig[k])
        parameters[2k] = zeros(1, 4*layerconfig[k])
        parameters[2k][1:layerconfig[k]] = 1 # forget gate bias
        input = layerconfig[k]
    end
    # bacward layer parameters
    input = embedsize
    for k=1:len
        parameters[2k-1+2len] = winit * randn(input+layerconfig[k], 4*layerconfig[k])
        parameters[2k+2len] = zeros(1, 4*layerconfig[k])
        parameters[2k+2len][1:layerconfig[k]] = 1 # forget gate bias
        input = layerconfig[k]
    end
    return map(p->convert(atype, p), parameters)
end


# 1:2*length(layerconfig) -> forward lstm
# 2*length(layerconfig)+1:4*length(layerconfig) -> backward lstm
function initstates(atype, layerconfig, batchsize)
    states = Array(Any, 4*length(layerconfig))
    len = length(layerconfig)

    # forward layer initialization
    for k=1:len
        states[2k-1] = zeros(batchsize, layerconfig[k])
        states[2k] = zeros(batchsize, layerconfig[k])
    end

    # backward layer initialization
    for k=1:len
        states[2k-1+2len] = zeros(batchsize, layerconfig[k])
        states[2k+2len] = zeros(batchsize, layerconfig[k])
    end
    return map(s->convert(atype, s), states)
end


function resetstate!(states, atype)
    for k=1:length(states)
        states[k] = zeros(size(states))
        convert(atype, states[k])
    end
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


# forward reads from 1 to n-1, backward reads from n to 2
# that loss assumes double embedding
function loss(parameters, states, sequence)
    total = 0.0
    count = 0
    atype = typeof(AutoGrad.getval(parameters[1]))
    hlayers = length(states) / 4
    hlayers = convert(Int, hlayers)

    # forward lstm
    fhiddens = Array(Any, length(sequence))
    for i=1:length(sequence)-1
        input = convert(atype, sequence[i])
        x = input * parameters[end-1]
        fhiddens[i+1] = forward(parameters[1:2*hlayers], states[1:2*hlayers], x)
    end
    padding = zeros(size(fhiddens[2]))
    fhiddens[1] = convert(atype, padding)

    # bacward lstm
    bhiddens = Array(Any, length(sequence))
    for i=length(sequence):-1:2
        input = convert(atype, sequence[i])
        x = input * parameters[end]
        bhiddens[i-1] = forward(parameters[2*hlayers+1:4*hlayers], states[2*hlayers+1:4*hlayers], x)
    end
    bhiddens[end] = convert(atype, padding)
    for i=1:length(fhiddens)
        ypred = hcat(fhiddens[i], bhiddens[i]) * parameters[end-2] .+ parameters[end-3]
        ynorm = logp(ypred, 2)
        ygold = convert(atype, sequence[i])
        count += size(ygold, 1)
        total += sum(ygold .* ynorm)
    end
    return - total / count
end

lossgradient = grad(loss)
