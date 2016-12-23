using Knet, ArgParse
include("model2.jl")
include("preprocess.jl")

function main(args=ARGS)
    s = ArgParseSettings()
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
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
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:atype] = eval(parse(o[:atype]))

    for (k,v) in o
        println("$k => $v")
    end

    # Data preperation
    sdata = Data("readme_data.txt"; batchsize=o[:batchsize])
    vocabsize = length(sdata.word_to_index)
    sequences = Any[];
    for item in sdata
        push!(sequences, item)
    end
    sequence = sequences[1]

    param = initparams(o[:atype], o[:layerconfig], o[:embedding], vocabsize, o[:winit]; single_embedding=o[:single_embedding])
    state = initstates(o[:atype], o[:layerconfig], o[:batchsize])
    initial_loss = loss(param, state, sequence)
    println("Initial loss is $initial_loss")
end
main(ARGS)
