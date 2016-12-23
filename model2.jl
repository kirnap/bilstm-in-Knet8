using Knet
include("model.jl")
include("preprocess.jl")


# 1:2*length(layerconfig) -> forward lstm
# 2*lenghth(layerconfig)+1:4*length(layerconfig) -> bacward lstm
function initparams(atype, layerconfig, embedsize, vocab, winit; single_embedding=false)
    if !single_embedding
        parameters = Array(Any, 4*length(layerconfig)+4)
        parameters[end-1] = winit * randn(vocab, embedsize) # forward lstm embedding
        parameters[end] = winit * randn(vocab, embedsize) # backward lstm embedding
    else
        parameters = Array(Any, 4*length(layerconfig)+3)
        parameters[end] = winit * randn(vocab, embedsize) # embedding shared among forward and backward lstm
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
    parameters[end-2] =  winit * randn(layerconfig[end], vocab)# final layer weight
    parameters[end-3] = zeros(1, vocab) # final layer bias
    return map(p->convert(atype, p), parameters)
end
