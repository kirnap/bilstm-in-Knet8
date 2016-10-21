@doc """ Bi-directional LSTM for language modeling, training procedure:
         Suppose an input sequence of x1-x2-x3,
         The backward lstm gets input as </s>-x3-x2,
         The forward lstm gets input as <s>-x1-x2 """

"""
parameters: holds the weight and bias for kth lstm layer as parameters[2k-1,2k]: weight and bias.
nlayers: Number of stacked lstm layers
states: Holds the cell and hidden states as state[2k-1,2k]: hidden and cell
"""
type LSTM
    parameters::Array
    nlayers::Int
    states::Array
    atype::DataType

    function LSTM(atype, layerconfig, embedsize, batchsize, winit)
        nlayers = length(layerconfig)
        parameters = initweights(atype, layerconfig, embedsize, winit)
        states = initstate(atype, layerconfig, batchsize)
        new(parameters, nlayers, states, atype)
    end
end


# No need for embedding matrix TODO: lookup table implementation
"""
Initialized the weights and biases for the stacked LSTM.
atype: Array type of the machine, namely KnetArray or Array regarding of gpu usage
hidden: Array of hidden layer configurations, e.g, [128 256] 2 layers of lstm.
embed: Embedding size of the model
winit: initialization parameter of the model
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
forward implementation for a single layer lstm
"""
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


"""
Go one step forward and return the final hidden state of the lstm for a given input
parameters[2k-1] is weight, parameters[2k] is bias for the kth layer
states[2k-1] is the hidden and states[2k] is the cell for the kth layer
"""
function sforw(parameters, states, input)
    x = input
    for i=1:2:length(states)
        (states[i], states[i+1]) = lstm(parameters[i], parameters[i+1], states[i], states[i+1], x)
        x = states[i]
    end
    return x
end


"""
Go forward and collect hidden states of the lstm for a sequence
sequence is padded from begining with token <s> and end with token </s>
returns hiddenlayers: contains the final hidden states of the forward and backward lstms in correct order,
"""
function bilsforw(parameters, states, sequence; forwardlstm=true)
    hiddenlayers = Array(Any, length(sequence))

    # if forward first item else the last item will be zeros of the same size with other items
    traverse = (forwardlstm ? (2:length(sequence)) : (length(sequence)-1:-1:1))

    for i=traverse
        result = sforw(parameters, states, sequence[i])
        hiddenlayers[i] = result
    end

    padindex = (forwardlstm ? 1 : length(hiddenlayers))
    padding = convert(atype, zeros(eltype(hiddenlayers[3]), size(hiddenlayers[3])))
    hiddenlayers[padindex] = padding
    return hiddenlayers
end


function loss(paramdict, statedict, sequence)
    fhiddens = bilsforw(paramdict["forw"], statedict["forw"], sequence)
    bhiddens = bilsforw(paramdict["back"], statedict["back"], sequence; forwardlstm=false)
    # TODO : Merge them and find the predictions, accumulate the loss
    
    
end
