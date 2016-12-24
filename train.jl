using Knet, ArgParse
include("model2.jl")
include("preprocess.jl")


function update!(param, state, sequence; lr=1.0, gclip=0.0)
    gloss = lossgradient(param, state, sequence)
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


function train!(param, data, o)
    state = initstates(o[:atype], o[:layerconfig], o[:batchsize])

    # test purpose!
    sequence = data[1]

    initial_loss = loss(param, state, sequence)
    println("Initial loss is $initial_loss")
    if o[:gcheck] > 0
        gradcheck(loss, param, copy(state), sequence; gcheck=o[:gcheck])
    end
    update!(param, state, sequence)

    # test purpose!
    next_loss = loss(param, state, sequence)
    println("Next loss is $next_loss")
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
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:atype] = eval(parse(o[:atype]))

    for (k,v) in o
        println("$k => $v")
    end

    # Data preperation
    tdata = Data(o[:trainfile]; batchsize=o[:batchsize])
    vocabsize = length(tdata.word_to_index)
    sequences = Any[];
    for item in tdata
        push!(sequences, item)
    end

    param = initparams(o[:atype], o[:layerconfig], o[:embedding], vocabsize, o[:winit]; single_embedding=o[:single_embedding])
    train!(param, sequences, o)
end
main(ARGS)
