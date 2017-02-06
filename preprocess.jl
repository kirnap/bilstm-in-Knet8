# Sequential preprocessor for large sized text files
# Gather the sequences of the same length s.t. when it is called it brings batches of similar sequences with the help of iterables

import Base: start, next, done


const SOS = "<s>"
const EOS = "</s>"
const UNK = "<unk>"
const LLIMIT = 2


type Data
    word_to_index::Dict{AbstractString, Int}
    index_to_word::Vector{AbstractString}
    batchsize::Int
    serve_type::AbstractString
    sequences::Dict
end


function Data(datafile; word_to_index=nothing, vocabfile=nothing, serve_type="bitarray", batchsize=20)
    # TODO: right now Data type can only be a bit array because of Knet support, no need to following line
    # (serve_type == nothing) && error("Please specify the data serve type: onehot, bitarray or sequence")
    existing_vocab = (word_to_index != nothing)
    if !existing_vocab
        word_to_index = Dict{AbstractString, Int}(SOS=>1, EOS=>2, UNK=>3)
        vocabfile != nothing && info("Working with provided vocabfile : $vocabfile")
        vocabfile != nothing && (V = vocab_from_file(vocabfile))
    end

    stream = open(datafile)
    sequences = Dict{Int64, Array{Any, 1}}()
    for line in eachline(stream)
        words = Int32[]
        push!(words, word_to_index[SOS])
        for word in split(line)
            if !existing_vocab && vocabfile != nothing && !(word in V)
                word = UNK
            end
            if existing_vocab
                index = get(word_to_index, word, word_to_index[UNK])
            else
                index = get!(word_to_index, word, 1+length(word_to_index))
            end
            push!(words, index)
        end
        push!(words, word_to_index[EOS])
        skey = length(words)
        (skey == LLIMIT) && println("LLMIT needs to be checked") #'ceause we already put <s> and </s> tokens?
        (!haskey(sequences, skey)) && (sequences[skey] = Any[])
        push!(sequences[skey], words)
    end
    close(stream)
    vocabsize = length(word_to_index)
    index_to_word = Array(AbstractString, vocabsize)
    for (word, index) in word_to_index
        index_to_word[index] = word
    end
    Data(word_to_index, index_to_word, batchsize, serve_type, sequences)
end


function sentenbatch(nom::Array{Any,1}, from::Int, batchsize::Int, vocabsize::Int; serve_type="bitarr")
    total = length(nom)
    to = (from + batchsize - 1 < total) ? (from + batchsize - 1) : total

    # not to work with surplus sentences
    if (to-from + 1 < batchsize)
        warning("Surplus does not being cleaned correctly!")
        return (nothing, 1)
    end
    
    new_from = (to == total) ? 1 : (to + 1)
    seqlen = length(nom[1]) # TODO: get rid of length computation give it as an extra argument!
    sentences = nom[from:to]

    # If Knet supports lookup that part will work correctly, it is commented for speed purposes
    #if serve_type == "lookup"
    #    return (sentences, new_from)
    #end

    scount = batchsize # modified future code
    data = [ falses(scount, vocabsize) for i=1:seqlen ]
    for cursor=1:seqlen
        for row=1:scount
            index = sentences[row][cursor]
            data[cursor][row, index] = 1
        end   
    end
    return (data, new_from)
end


"""Removes the surplus sentences randomly"""
function clean_seqdict!(seqdict::Dict{Int64,Array{Any,1}}, batchsize::Int)
    for seqlen in keys(seqdict)
        remain = rem(length(seqdict[seqlen]), batchsize)
        while remain != 0
            index = rand(1:length(seqdict[seqlen]))
            deleteat!(seqdict[seqlen], index)
            remain -= 1
        end
        if isempty(seqdict[seqlen])
            delete!(seqdict, seqlen)
        end
    end
end


function start(s::Data)
    sdict = deepcopy(s.sequences)
    clean_seqdict!(sdict, s.batchsize)
    @assert (!isempty(sdict)) "There is not enough data with that batchsize $(s.batchsize)"
    slens = collect(keys(sdict))
    seqlen = pop!(slens)
    from = nothing
    vocabsize = length(s.word_to_index)
    state = (sdict, seqlen, slens,from, vocabsize)
    return state
end


function next(s::Data, state)
    (sdict, seqlen, slens, from, vocabsize) = state

    if from == nothing
        (item, new_from) = sentenbatch(sdict[seqlen], 1, s.batchsize, vocabsize)
    elseif from == 1
        seqlen = pop!(slens)
        (item, new_from) = sentenbatch(sdict[seqlen], from, s.batchsize, vocabsize)
    else
        (item, new_from) = sentenbatch(sdict[seqlen], from, s.batchsize, vocabsize)
    end
    from = new_from
    state = (sdict, seqlen, slens, from, vocabsize)
    return (item, state)
end


function done(s::Data, state)
    (sdict, seqlen, slens, from, vocabsize) = state
    return isempty(slens) && (from == 1)
end


""" Creates a set that contains all the words in that file, vocab file given as each vocab in a single line
    sorted_counted represents pure create_vocab.sh output
"""
function vocab_from_file(vocabfile; sorted_counted=false)
    V = Set{AbstractString}()
    open(vocabfile) do file
        for line in eachline(file)
            line = split(line)
            if sorted_counted
                (length(line)>1) && push!(V, line[2])
            else
                !isempty(line) && push!(V, line[1])
            end
        end
    end
    return V
end

""" Builds the kth sentence from a given sequence """
function ibuild_sentence(tdata::Data, sequence::Array{BitArray{2},1}, kth::Int)
    sentence = Array{Any, 1}()
    for i=1:length(sequence)
        z = find(x->x==true, sequence[i][kth,:])
        append!(sentence, z)
    end
    ret = map(x->tdata.index_to_word[x], sentence)
    return ret
end

# FUTURE CODE: if one day knet8 allows us to change batchsize on the fly, following lines will implement surplus batch implementation, this code snippet would be put on sentenbatch
    # (length(sentences) != batchsize) && (println("I am using the surplus sentences:) $from : $to"))
    # scount = length(sentences) # it can be either batchsize or the surplus sentences
    # data = [ falses(scount, vocabsize) for i=1:seqlen ]
    # for cursor=1:seqlen
    #     for row=1:scount
    #         index = sentences[row][cursor]
    #         data[cursor][row, index] = 1
    #     end   
    # end
