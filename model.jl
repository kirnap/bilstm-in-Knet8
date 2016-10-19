

"""
parameters: holds the weight and bias for kth lstm layer as parameters[2k-1,2k]: weight and bias.
nlayers: Number of stacked lstm layers
states: Holds the cell and hidden states as state[2k-1,2k]: hidden and cell
"""
type LSTM
    parameters::Array
    nlayers::Int
    states::Array

    function LSTM(atype, layerconfig, embedsize, batchsize, winit)
        nlayers = length(layerconfig)
        parameters = initweights(atype, layerconfig, embedsize, winit)
        states = initstate(atype, layerconfig, batchsize)
        new(parameters, nlayers, states)
    end
end


# No need for embedding matrix TODO: lookup table implementation
"""
Initialized the weights and biases for the stacked LSTM.
"""
function initweights(atype, hidden, embed, winit)
    param = Array(Any, 2*length(hidden))
    input = embed
    for k = 1:length(hidden)
        param[2k-1] = winit*randn(input+hidden[k], 4*hidden[k])
        param[2k]   = zeros(1, 4*hidden[k])
        param[2k][1:hidden[k]] = 1 # forget gate bias
        input = hidden[k]
    end
    return map(p->convert(atype,p), param)
end


"""
state[2k-1]: hidden, state[2k]: cell
"""
function initstate(atype, hidden, batchsize)
    state = Array(Any, 2*length(hidden))
    for k = 1:length(hidden)
        state[2k-1] = zeros(batchsize,hidden[k])
        state[2k] = zeros(batchsize,hidden[k])
    end
    return map(s->convert(atype,s), state)
end


"""
forward implementation for single layer lstm
"""
function lstmforward(weight, bias, hidden, cell, input)
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


# TODO: Merge layer will handle the final prediction parameters, namely weight and bias
