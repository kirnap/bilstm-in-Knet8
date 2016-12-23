# 1:2*length(layerconfig) -> forward lstm
# 2*lenghth(layerconfig)+1:4*length(layerconfig) -> bacward lstm
# final 3 or 4 is embedding and final layer prediction parameters
function initparams(atype, layerconfig, embedsize, vocab, winit; single_embedding=false)
    if !single_embedding
        parameters = Array(Any, 4*length(layerconfig)+4)
        parameters[end] = winit * randn(vocab, embedsize) # backward lstm embedding
        parameters[end-1] = winit * randn(vocab, embedsize) # forward lstm embedding
        parameters[end-2] =  winit * randn(layerconfig[end], vocab)# final layer weight
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
