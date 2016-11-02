# Sequential preprocessor for large sized text files

# Rough TODO:
# Flexibility of vocab file and create the vocabulary while reading the file
# Gather the sequences of the same length s.t. when it is called it brings batches of similar sequences with the help of iterables

const SOS = "<s>"
const EOS = "</s>"
const UNK = "<unk>"
const LLIMIT = 2





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
