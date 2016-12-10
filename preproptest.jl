# This file is not readable yet, is used for unit testing purposes needs to be organizedo
include("preprocess.jl")


function test_datatype(datafile)
    serve_type = "bitarray"
    batchsize = 57
    data = Data(datafile; serve_type=serve_type, batchsize=batchsize)
    return data
end

""" test the sequence field of datatype"""
function test_sequence_field(testfile)
    # Based on the property of the testdata.txt we only have sentences of length 3-4-5, corresponding
    # lengths of 5-6-7 that is being hold on to the test_sentence_sequence
    # preprocess the data
    tdata = test_datatype(testfile)
    seq5 = tdata.sequences[5]
    seq6 = tdata.sequences[6]
    seq7 = tdata.sequences[7]

    sets = icreate_sets(testfile)
    set5 = sets[1]
    set6 = sets[2]
    set7 = sets[3]

    # Not fancy but works for now
    for i=1:length(seq5)
        assert(map(x->tdata.index_to_word[x], seq5[i])[2:end-1] in set5) # ignore <s> and </s> for now
    end
    for i=1:length(seq6)
        assert(map(x->tdata.index_to_word[x], seq6[i])[2:end-1] in set6) # ignore <s> and </s> for now
    end
    for i=1:length(seq7)
        assert(map(x->tdata.index_to_word[x], seq7[i])[2:end-1] in set7) # ignore <s> and </s> for now
    end
end


""" reserves all the batches of a given length-sequence for test purposes """
function icreate_batches(sentences::Array{Any,1}, batchsize::Int, vocabsize::Int)
    batches = Array{Any, 1}()
    (d, from) = sentenbatch(sentences, 1, batchsize, vocabsize) #what we expect from sentencebatch is to give us a sequence, from is 1 in the first call
    push!(batches, d)
    while from != 1
        (d, from) = sentenbatch(sentences, from, batchsize, vocabsize)
        push!(batches, d)
    end
    if nothing in batches
        info("Removing nothing entry")
        pop!(batches)
    end
    return batches
end


""" Returns the sets that contains all the same length sentences """
function icreate_sets(testfile::AbstractString)
    # Be careful in our testfile there exists only the length of 5-6-7 sentences!
    # Here in the length of 6 there are two sentences of the same : "part of a series"
    set5 = Set()
    set6 = Set()
    set7 = Set()

    open(testfile) do file
        for line in eachline(file)
            len = length(split(line))
            (len == 3) && push!(set5, split(line)) #push!(lines5, split(line))
            (len == 4) && push!(set6, split(line))
            (len == 5) && push!(set7, split(line))
        end
    end
    return (set5, set6, set7)
end


""" Builds the kth sentence from a given sequence """
function ibuild_sentence(tdata::Data, sequence::Array{BitArray{2},1}, kth::Int)
    sentence = Array{Any, 1}()
    for i=1:length(sequence)
        z = find(x->x==true, sequence[i][kth,:])
        append!(sentence, z)
    end
    ret = map(x->tdata.index_to_word[x], sentence)
    return ret[2:end-1] # because first character and the last character are padded.
end


"""Compares the whether batched sentence in the given test or not, we expect it to return empty array"""
function ihelp_sentence_batch(tdata, batches, set; test_surplus=true)
    ret = Any[]
    for sequence in batches
        l = size(sequence[1])[1]
        sentences = map(x->ibuild_sentence(tdata, sequence, x), collect(1:l))
        (test_surplus) && (l < 3) && push!(ret, sentences)
        for item in sentences
            assert(item in set)
        end
    end
    return ret
end


function test_sentence_batch(testfile)
    # get ready for the sentence batch
    tdata = test_datatype(testfile)
    batchsize = 3
    test_cleanseqdict!(tdata.sequences, batchsize)
    vocabsize = length(tdata.word_to_index)
    seq5 = tdata.sequences[5]
    seq6 = tdata.sequences[6]
    seq7 = tdata.sequences[7]

    sets = icreate_sets(testfile)
    set5 = sets[1]
    set6 = sets[2]
    set7 = sets[3]
    
    batches5 = icreate_batches(seq5, batchsize, vocabsize)
    batches6 = icreate_batches(seq6, batchsize, vocabsize)
    batches7 = icreate_batches(seq7, batchsize, vocabsize)
    
    x1 = ihelp_sentence_batch(tdata, batches5, set5)
    x2 = ihelp_sentence_batch(tdata, batches6, set6)
    x3 = ihelp_sentence_batch(tdata, batches7, set7)
    return (x1, x2, x3)
end


function test_cleanseqdict!(seqdict, batchsize)
    clean_seqdict!(seqdict, batchsize)
end


function test_iteration(s::Data)
    sequences = Any[]
    vocabsize = length(s.word_to_index)
    for sequence in s
        push!(sequences, sequence)
    end

    # Those part of the code is very costly and use it just to make sure batching done correctly
    # for seq in sequences
    #     for item in seq
    #         @assert (size(item)==(s.batchsize, vocabsize)) "$(size(item))"
    #     end
    # end
    return sequences
end


function main()
    datafile = "ptb/ptb.train.txt"
    testfile = "testdata.txt"
    
    # tests the vocabulary creation correctedness
    # vocabfile = "test_vocab"
    # global V = vocab_from_file(vocabfile)
    # global x = length(V)
    # words = split(readstring(testfile))
    # global vocab = Dict()
    # for word in words; get!(vocab, word, 1+length(vocab));end
    # for k in keys(vocab); assert(k in V);end


    # test of Data type
    # global tdata = test_datatype(testfile)

    # tests the sequence property that is being hold by the Data type

    
    # test of sentenbatch
    # global surplused = test_sentence_batch(testfile)

    # test of surplus cleaning
    # test_cleanseqdict!(tdata.sequences, 3)

    # test of iterable implementation
    global tdata = test_datatype(testfile)
    
    # test the iteration
    global data = test_datatype(datafile)
    global sequences = test_iteration(data)
    
    # TODO: check that it <unk>s the unknown words if an external vocabulary is given
end
# main()

# Rough documentation:
# --
# Read the all sentence and put them into unique sets by icreate_sets
# After iterating over the Data type ibuild_sentence can build ith sentence by going over the whole sequence of sequence in Datatype
