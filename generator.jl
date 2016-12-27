# open the following 3 lines in case you need to use stand-alone substitute generation
#using Knet, JLD
#include("model2.jl")
#include("preprocess.jl")

function gensub(parameters, states, sequence; lp=false)
    result = Array(Any, length(sequence))
    hlayers = length(states) / 4
    hlayers = convert(Int, hlayers)

    # forward lstm
    fhiddens = Array(Any, length(sequence))
    t1 = states[1:2*hlayers]
    for i=1:length(sequence)-1
        input = oftype(parameters[1], sequence[i])
        x = input * parameters[end-1]
        fhiddens[i+1] = forward(parameters[1:2*hlayers], t1, x)
    end
    fhiddens[1] = oftype(fhiddens[2], zeros(size(fhiddens[2])))

    # backward lstm
    bhiddens = Array(Any, length(sequence))
    t2 = states[2*hlayers+1:4*hlayers]
    for i=length(sequence):-1:2
        input = oftype(parameters[1], sequence[i])
        x = input * parameters[end]
        bhiddens[i-1] = forward(parameters[2*hlayers+1:4*hlayers], t2, x)
    end
    bhiddens[end] = oftype(bhiddens[end-1], zeros(size(bhiddens[2])))

    # merge layer
    for i=1:length(fhiddens)
        ypred = hcat(fhiddens[i], bhiddens[i]) * parameters[end-2] .+ parameters[end-3]
        ynorm = logp(ypred, 2)
        result[i] = (lp ? ynorm : exp(ynorm))
    end
    return result
end


function printsub(parameters, states, sequence, index_to_word; lp=true)
    pbvec = gensub(parameters, states, sequence; lp=lp)
    assert(length(pbvec) == length(sequence))
    for i=1:length(sequence)
        subs = zip(index_to_word, pbvec[i])
        index = find(x->x==true, sequence[i])
        word = index_to_word[index][1]
        print(word, "\t")
        for (sub, lprob) in subs
            print(sub," ", lprob,"\t")
        end
        println()
    end
    return pbvec
end

function test()
    x = load("initialmodel.jld")
    param = x["model"]
    state = x["initstates"]
    tdata = Data("readme_data.txt";batchsize=1)
    sequences = Any[]
    for item in tdata; push!(sequences, item);end;
    printsub(param, state, sequences[1], tdata.index_to_word)
end
