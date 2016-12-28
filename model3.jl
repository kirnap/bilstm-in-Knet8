# Single embedding implementation of blstm
# 1:2*length(layerconfig) -> forward lstm
# 2*lenghth(layerconfig)+1:4*length(layerconfig) -> bacward lstm
# final 3 is embedding and final layer prediction parameters
function initparams(atype, layerconfig, embedsize, vocab, winit)
    parameters = Array(Any, 4*length(layerconfig)+3)
    parameters[end] = winit * randn(vocab, embedsize) # embedding shared among forward and backward lstm
    parameters[end-1] =  winit * randn(layerconfig[end] * 2, vocab)# final layer weight
    parameters[end-2] = zeros(1, vocab) # final layer bias
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
# sequence corresponds the n minitbatches in the literature
function loss(parameters, states, sequence)
    total = 0.0
    count = 0
    atype = typeof(AutoGrad.getval(parameters[1]))
    hlayers = length(states) / 4
    hlayers = convert(Int, hlayers)

    # forward lstm
    fhiddens = Array(Any, length(sequence))
    t1 = states[1:2*hlayers]
    for i=1:length(sequence)-1
        input = convert(atype, sequence[i])
        x = input * parameters[end]
        fhiddens[i+1] = forward(parameters[1:2*hlayers], t1, x)
        #(prev_1 == t1) && println("State is not being modified")
        #(fhiddens[i+1] == t1[1]) && println("I get the correct hidden")
        #prev_1 = t1[1:end]
    end
    fhiddens[1] = convert(atype, zeros(size(fhiddens[2])))

    # backward lstm
    bhiddens = Array(Any, length(sequence))
    t2 = states[2*hlayers+1:4*hlayers]
    for i=length(sequence):-1:2
        input = convert(atype, sequence[i])
        x = input * parameters[end]
        bhiddens[i-1] = forward(parameters[2*hlayers+1:4*hlayers], t2, x)
    end
    bhiddens[end] = convert(atype, zeros(size(bhiddens[2])))

    # merge layer
    for i=1:length(fhiddens)
        ypred = hcat(fhiddens[i], bhiddens[i]) * parameters[end-1] .+ parameters[end-2]
        ynorm = logp(ypred, 2)
        ygold = convert(atype, sequence[i])
        count += size(ygold, 1)
        total += sum(ygold .* ynorm)
    end
    return - total / count
end

lossgradient = grad(loss)
