using Knet, ArgParse, JLD
include("model2.jl")
include("preprocess.jl")
#include("generator.jl")


function test(param, state, data; perp=false)
    totloss = 0.0
    numofbatch = 0
    for sequence in data
        if length(sequence) == 2
            continue
        end
        val = loss(param, deepcopy(state), sequence) #val = loss(param, state, sequence)
        totloss += (perp? exp(val) : val)
        numofbatch += 1
    end
    return totloss / numofbatch
end


function update!(param, state, sequence; lr=1.0, gclip=0.0)
    gloss = lossgradient(param, deepcopy(state), sequence) #    gloss = lossgradient(param, state, sequence)
    gscale = lr
    if gclip > 0
        gnorm = sqrt(mapreduce(sumabs2, +, 0, gloss))
        if gnorm > gclip
            gscale *= gclip / gnorm
        end
    end

    for k=1:length(param)
        axpy!(-gscale, gloss[k], param[k])
    end

    isa(state, Vector{Any}) || error("State should not be Boxed.")
    # TODO: is that really needed?
    for i=1:length(state)
        state[i] = AutoGrad.getval(state[i])
    end
end


function train!(param, state, data, o)
    for sequence in data
        if length(sequence) == 2
            continue
        end
        # Only open for gradient check
        if o[:gcheck] > 0
            gradcheck(loss, param, copy(state), sequence; gcheck=o[:gcheck])
        end
        update!(param, state, copy(sequence); lr=o[:lr], gclip=o[:gclip]) #update!(param, state, sequence; lr=o[:lr], gclip=o[:gclip])
    end
end


function main(args=ARGS)
    s = ArgParseSettings()
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--trainfile"; default="readme_data.txt" ;help="Training file")
        ("--devfile"; help="Dev file")
        ("--testfile"; help="Test file")
        ("--loadfile"; help="Initialize model from file")
        ("--savefile"; help="Save final model to file")
        ("--vocabfile"; default=nothing; help="Vocabulary file")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--layerconfig"; arg_type=Int; nargs='+'; default=[16]; help="Sizes of the one or more LSTM layers")
        ("--embedding"; arg_type=Int; default=200; help="embedding vector")
        ("--batchsize"; arg_type=Int; default=1; help="Minibatchsize")
        ("--epochs"; arg_type=Int; default=3; help="Number of epochs for training.")
        ("--winit"; arg_type=Float64; default=0.3; help="Initial weights set to winit*randn().")
        ("--decay"; arg_type=Float64; default=0.9; help="Learning rate decay.")
        ("--lr"; arg_type=Float64; default=4.0; help="Initial learning rate.")
        ("--single_embedding"; default=false)
        ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
        ("--gclip"; arg_type=Float64; default=0.0; help="Value to clip the gradient norm at.")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:atype] = eval(parse(o[:atype]))

    for (k,v) in o
        println("$k => $v")
    end
    !o[:single_embedding] && println("Double embedding is used\n")

    # Data preperation
    tdata = Data(o[:trainfile]; batchsize=o[:batchsize], vocabfile=o[:vocabfile])
    vocabsize = length(tdata.word_to_index)

    # Devdata preperation
    dfile = (o[:devfile] != nothing ? o[:devfile] : o[:trainfile])
    ddata = Data(dfile; batchsize=o[:batchsize], word_to_index=tdata.word_to_index)


    # Model initialization
    param = initparams(o[:atype], o[:layerconfig], o[:embedding], vocabsize, o[:winit]; single_embedding=o[:single_embedding])
    state = initstates(o[:atype], o[:layerconfig], o[:batchsize])


    # inital loss value
    initial_loss = test(param, state, tdata; perp=true) #initial_loss = test(param, deepcopy(state), tdata; perp=true) 
    println("Initial loss is $initial_loss")
    devloss = test(param, state, ddata; perp=true) #devloss = test(param, deepcopy(state), ddata; perp=true)
    println("Initial dev loss is $devloss")
    devlast = devbest = devloss

    # training started
    for epoch=1:o[:epochs]
        @time train!(param, state, tdata, o) # train!(param, deepcopy(state), tdata, o)
        devloss = test(param, state, ddata; perp=true)
        println("Dev loss for epoch $epoch : $devloss")

        if (epoch % 5) == 0
            trainloss = test(param, state, tdata; perp=true)
            println("Train loss for epoch $epoch : $trainloss")
        end
        # check whether model becomes better
        if devloss < devbest
            devbest = devloss
            if o[:savefile] != nothing
                saveparam = map(p->convert(Array{Float32}, p), param)
                save(o[:savefile], "model", saveparam, "vocab", tdata.word_to_index, "config", o)
            end
        end

        if devloss > devlast
            o[:lr] *= o[:decay]
            info("New learning rate: $(o[:lr])")
        end
        devlast = devloss
        flush(STDOUT)
    end
end
!isinteractive() && main(ARGS)
