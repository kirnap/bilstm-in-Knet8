# Sequential preprocessor for large sized text files

# Rough TODO:
# Flexibility of vocab file and create the vocabulary while reading the file
# Gather the sequences of the same length s.t. when it is called it brings batches of similar sequences with the help of iterables

const SOS = "<s>"
const EOS = "</s>"
const UNK = "<unk>"
const LLIMIT = 2

type Data
    word_to_index::Dict{AbstractString, Int}
    index_to_word::Vector{AbstractString}
    batchsize::Int32
    serve_type::AbstractString
    sequences::Dict
end

function Data(datafile; word_to_index=nothing, vocabfile=nothing, serve_type=nothing, batchsize=20)
    (serve_type == nothing) && error("Please specify the data serve type: onehot, bitarray or sequence")
    existing_vocab = (word_to_index != nothing)
    if !existing_vocab
        word_to_index = Dict{AbstractString, Int}(SOS=>1, EOS=>2, UNK=>3)
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
        (!haskey(sequences, skey)) && (skey > LLIMIT) && (sequences[skey] = Any[])
        (skey != LLIMIT) && push!(sequences[skey], words)
    end
    close(stream)
    vocabsize = length(word_to_index)
    index_to_word = Array(AbstractString, vocabsize)
    for (word, index) in word_to_index
        index_to_word[index] = word
    end
    Data(word_to_index, index_to_word, batchsize, serve_type, sequences)
end




""" Creates a set that contains all the words in that file, vocab file given as each vocab in a single line """
function vocab_from_file(vocabfile)
    V = Set{AbstractString}()
    open(vocabfile) do file
        for line in eachline(file)
            line = split(line)
            # (length(line)>1) && push!(V, line[2]) # if the skeleton is given use that 
            !isempty(line) && push!(V, line[1])
        end
    end
    return V
end
