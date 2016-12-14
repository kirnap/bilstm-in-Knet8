# BLSTM model implementation in [Knet](https://github.com/denizyuret/Knet.jl)
---
## Task Definition
TODO: put the task definition here
## Preprocessing
BLSTM model accepts sequence as an input, and in this task our sequences are input sentences. Thus, we need to arrange the senteces such that train method of the model should take the whole sentence and calculate the loss per token.

Suppose that you have a data file such that in eachline of that file there exists one single sentence of length n. There you can use `Data type` as follows:
```Julia
include("preprocess.jl")
datafile = "readme_data.txt"
batchsize = 3
test_data = Data(datafile; batchsize=batchsize)
for sequence in test_data
	# Feed your blstm model with that sequence
end 
```
To understand what ```sequence``` is let's go through with an example. Suppose you have 3 test sentences of length 6 and ```batchsize``` is 3:
- The dog ran out of memory
- Doga is a name in Turkish
- Life is short let us code

When you construct a data out of that test sentences the sequence will be an array of length 8. You may ask why not 6 but 8, well that is because `Data` padds the special start and end of sentence tokens as follows: `<s> "sentence" </s>` and here is how sequence looks like:
```Julia
[[<s>, <s>, <s>],
[The, Doga, Life],
[dog, is, is],
[ran, a short],
[out, in, let], 
[memory, Turkish, code], 
[</s>, </s>, </s>]]
```
However, one more trick here is applied by `Data` is that each word is not represented as regular string but instead they are encoded as one-hots where 1 represented as `true` and 0s represented as `false` for the sake of memory usage. Finally, each element of the sequence is a bitarray of size `batchsize, vocabulary`. The very final version of sequence:

```Julia
julia> sequence
8-element Array{BitArray{2},1}:
 Bool[true false … false false; true false … false false; true false … false false]
 Bool[false false … false false; false false … false false; false false … false false]
 Bool[false false … false false; false false … false false; false false … false false]
 Bool[false false … false false; false false … false false; false false … false false]
 Bool[false false … false false; false false … false false; false false … false false]
 Bool[false false … false false; false false … false false; false false … true false]
 Bool[false false … false false; false false … false false; false false … false true]
 Bool[false true … false false; false true … false false; false true … false false]
 

julia> sequence[1]
3×20 BitArray{2}:
 true  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false
 true  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false
 true  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false
 

julia> sequence[4]
3×20 BitArray{2}:
 false  false  false  false  false   true  false  false  false  false  false  false  false  false  false  false  false  false  false  false
 false  false  false  false  false  false  false  false  false  false  false   true  false  false  false  false  false  false  false  false
 false  false  false  false  false  false  false  false  false  false  false  false  false  false  false  false   true  false  false  false
```

TODO:put additional futures of ```Data```

## Model
TODO: put the model definition here

Ps: Since it is written for my own understanding and testing purpose, I am not recommending to read preproptest.jl file, unless you want to help me to create more sophisticated test coverage.


## Loss function implementation details
---
Here let **y <sub>pred</sub>**  be the prediction vector which has batchsize many rows and vocabulary size many columns:

![equation](http://www.sciweavers.org/tex2img.php?eq=y_%7Bpred%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%0A%20y_1%26y_2%20%26..%26y_C%20%0A%5Cend%7Bbmatrix%7D_%7Bbatchsize%20%5Chspace%7B1mm%7D%20x%20%20%5Chspace%7B1mm%7D%20vocabsize%7D&bc=White&fc=Black&im=gif&fs=12&ff=arev&edit=0)

The traditional softmax output and loss for a single instance would be

![equation](http://www.sciweavers.org/tex2img.php?eq=%5Chat%7Bp_i%7D%3D%20%5Cfrac%7B%5Cexp%20y_i%7D%7B%5Csum_%7Bc%3D1%7D%5EC%20%5Cexp%20y_c%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)   and   ![equation](http://www.sciweavers.org/tex2img.php?eq=%5Cell%20%3D%20%5Csum_%7Bc%3D1%7D%5EC%20p_c%20%5Clog%20%5Chat%7Bp%7D_c&bc=White&fc=Black&im=gif&fs=12&ff=arev&edit=0)

Let's play with the logarithmic term in the loss function:
![equation](http://www.sciweavers.org/tex2img.php?eq=%20%5Clog%20%5Chat%7Bp%7D_i%20%3D%20log%28e%5E%7By_i%7D%29%20-%20log%28%5Csum_k%5EC%7Be%5E%7Bx_k%7D%7D%29%20%3D%20y_i%20-%20log%28%5Csum_k%5EC%7Be%5E%7Bx_k%7D%7D%29&bc=White&fc=Black&im=gif&fs=12&ff=arev&edit=0)

Finally here is the how knet makes that trick:
```Julia
ynorm = logp(ypred,2) # ypred .- log(sum(exp(ypred),2))
```

Thanks to Professor [Deniz Yuret](http://www.denizyuret.com/) for his genuine help in understanding the concepts.


