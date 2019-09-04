
# NNET

This is my Jupyter notebook with notes / implementation of a neural network in Julia. All materials are based on deeplearning.ai's Deep Learning Specialization. I refer to specific videos and other materials where applicable.

I created this notebook to review some of the concepts I learned in the first three courses of the deep learning specialization. Its purpose is primarily for me to process the information and produce the implementation from my own understanding rather than simply copying it from another place. As a result, you may find that the implementation notes are somewhat long-winded or tangential.

If you find a mistake or think this code can be improved, it would be great if you could fork the repository and make a pull request. You could also leave a comment in the issues section.

### This notebook contains

- Loading & pre-processing the MNIST dataset
- forward and backward propagation using a n-layer neural network
- ReLU / sigmoid / 'safe' softmax activation functions
- Inverted dropout implementation
- Xavier / He weight initialization 
- Adam
- Learning rate decay
- mini-batch gradient descent

### References

- I refer to the deep learning specialization course materials where applicable
- I translated some of the numpy code from Jonathan Weisberg's "Building a Neural Network from Scratch" [part I](https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/) and [part II](https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%202/) to Julia

## Loading the MNIST dataset

The MNIST dataset contains $60.000$ images of digits $1-10$. We use the MLDatasets module to retrieve and load the data. Normally, we would [normalize](https://www.coursera.org/learn/deep-neural-network/lecture/lXv6U/normalizing-inputs) the input data such that it has mean $0$ and standard deviation $1$ (also known as a standard normal distribution). However, this has already been done for this version of the MNIST dataset.


```julia
using MLDatasets

# Train & test data
train_x, train_y = MNIST.traindata();
test_x, test_y = MNIST.testdata();

# Save original data
Xtr = train_x
Xtest = test_x;

# Note that the data have been normalized already
# See: https://juliaml.github.io/MLDatasets.jl/latest/datasets/MNIST/
```

    ┌ Info: Recompiling stale cache file /home/jasper/.julia/compiled/v1.0/MLDatasets/9CUQK.ji for MLDatasets [eb30cadb-4394-5ae3-aed4-317e484a6458]
    └ @ Base loading.jl:1190



```julia
function plot_image(X, label)
    
    #= 
    Plot an image
    =#
    
    println("Label: ", label)
    
    # Plot
    img = colorview(Gray, transpose(X));
    p1 = plot(X, legend=false)
    p2 = plot(img)
    plot(p1,p2)
    
    end;
```


```julia
i = 107
plot_image(Xtr[:,:,i], train_y[i])
```

    Label: 6



    UndefVarError: colorview not defined

    

    Stacktrace:

     [1] plot_image(::Array{FixedPointNumbers.Normed{UInt8,8},2}, ::Int64) at ./In[2]:10

     [2] top-level scope at In[3]:2


The shapes of the data are as follows


```julia
## Get shapes
println(size(train_x))
println(size(train_y))
trainsize = size(train_x)[3];
```

    (28, 28, 60000)
    (60000,)


I find it easier to think about the data if the examples are in the row dimension. We will reshape the data such that $X$ is a tensor of shape $(60.000, 784)$. 


```julia
# Reshape x
train_x = transpose(reshape(train_x, 28 * 28, size(train_x)[3]));
test_x = transpose(reshape(test_x, 28 * 28, size(test_x)[3]));

# Reshape Y
train_y = reshape(train_y, size(train_y)[1], 1);
test_y = reshape(test_y, size(test_y)[1], 1);
```

$Y$ has $10$ unique labels ranging from $0-9$. However, this is actually a recoding of the data.


```julia
size(train_y)
```




    (60000, 1)




```julia
unique(train_y)
```




    10-element Array{Int64,1}:
     5
     0
     4
     1
     9
     2
     3
     6
     7
     8



### The data until now

- $X$ is a tensor with dimension $(60.000, 784)$
- $Y$ is a tensor with dimension $(60.000, 1)$

<img src="img/XandY2.png" width = "500px">

### Creating a one-hot encoding for $Y$

In order to train the neural network, we need to one-hot-encode <link> the outcome variable $Y$.

The trick to doing this is as follows:

- Create a $k \times k$ identity matrix. In our case, we have $k=10$, so:

$$
I = 
\begin{bmatrix}
1 & 0 & \dots & 0 \\
0 & 1 & \dots & 0 \\
\vdots & \vdots & \ddots & 0 \\
0 & 0 & \dots & 1
\end{bmatrix}
$$

- For each example in $Y$, *index the column of the value of the label*. For example, if the the label equals '5', we index the $5^{th}$ column, which has a $1$ on the $5^{th}$ element and zeroes elsewhere

$$
\begin{bmatrix}
0 \\
0 \\
0 \\
0 \\
1 \\
0 \\
0 \\
0 \\
0 \\
0 
\end{bmatrix}
$$

- Because we do this $n$ times (the number of training examples), we will end up with a tensor of size $(10, 60.000)$

$$
\begin{bmatrix}
0 & 0 & 0 & \dots & 1 \\
1 & 1 & 0 & \dots & 0 \\
0 & 0 & 0 & \dots & 0 \\
0 & 0 & 0 & \dots & 0 \\
0 & 0 & 0 & \dots & 0 \\
0 & 0 & 0 & \dots & 0 \\
0 & 0 & 1 & \dots & 0 \\
0 & 0 & 0 & \dots & 0 \\
0 & 0 & 0 & \dots & 0 \\
0 & 0 & 0 & \dots & 0
\end{bmatrix}
$$

- Again, I like to think in terms of (examples, dimensions) so we will transpose the one-hot-encoded matrix such that it has the dimensions $(60.000, 10)$

<img src="img/y_one_hot.png" width = "300px">


```julia
# One-hot encoding for y
# We need a shape for outcome variable of size [num_labels, num_examples]

# Number of distinct classes
k = length(unique(train_y));
# Number of training examples
n = size(train_y)[1];
```


```julia
using LinearAlgebra

function one_hot_encode(y_labels)
    
    #=
    From multiclass labels to one-hot encoding
    
    :param y_labels: column vector of shape (number of examples, 1) containing k > 2 classes
    
    :return: one-hot encoded labels
    =#
    
    # Number of distinct classes
    k = length(unique(y_labels));
    # Number of rows
    n = size(y_labels)[1];
    
    # 1. Create k x k identity matrix 

    # Arrange the identity matrix such that, for each class, we get a one-hot encoded vector.
    # This means that the rows are of length k (the number of distinct classes)
    # The columns are of length n (the number of examples).
    y_ohe = Matrix{Float64}(I, k, k)

    # 2. Organize such that we get a k x m matrix. We do this by letting the label index
    #     the column value. Since we have m labels, we index the columns m times.
    #     So for m = 1.000 where m_1000 = 3, we index the kth column [0,0,1,0,...,k]
    
    # We have to add +1 to the classes (because Julia indexes from 1)
    # Unlike e.g. numpy we need to explicitly call 'broadcast' to match shapes between
    # two elements that we're adding
    y_ohe = y_ohe[:, broadcast(+, y_labels, 1)][:,:];
    
    # Return
    return(y_ohe)
    
    end;

# One-hot encoding
y_train_ohe = transpose(one_hot_encode(train_y));
y_test_ohe = transpose(one_hot_encode(test_y));

println(size(y_train_ohe))
println(size(y_test_ohe))
```

    (60000, 10)
    (10000, 10)


just to make sure that the one-hot encoding was done properly, we can randomly sample a number of values and check if the one-hot encoded examples line up with the original labels.


```julia
using Random

function sanity(y_ohe, train_y; n_sample = 5)
    
    #=
    Ensure that one-hot encoded labels are encoded properly by transforming them back into
     labels.
    
    :param y_ohe: one-hot encoded training labels
    :param train_y: array of size (examples, 1) containing the labels
    :param n_sample: number of examples to sample randomly
    
    :return: This function does not return a value
    =#
    
    # Shapes
    n, k = size(y_ohe)
    
    # Pick a random example
    ind_shuffled = shuffle(1:n)
    
    # Subset
    ind = ind_shuffled[1:n_sample]
    
    # For each, print OHE + convert back to class
    for i in ind
        
        # Find position of 1
        pos = findall(x -> x==1, y_ohe[i, :])
        
        # Subtract 1
        pos = pos[1] - 1
        
        # Print
        println("Example ", i, ":\n", 
                "\t[OHE position: ", pos, "] ==> [label: ", train_y[i,1], "]")
        
    end
    
end;
```


```julia
sanity(y_train_ohe, train_y)
```

    Example 37974:
    	[OHE position: 6] ==> [label: 6]
    Example 52430:
    	[OHE position: 0] ==> [label: 0]
    Example 58525:
    	[OHE position: 7] ==> [label: 7]
    Example 27837:
    	[OHE position: 1] ==> [label: 1]
    Example 52157:
    	[OHE position: 6] ==> [label: 6]


### NNet functions

Plan: build a neural net with one hidden layer and $n_h$ hidden units

The following have not (yet) been implemented

- Matrix calculus (see https://atmos.washington.edu/~dennis/MatrixCalculus.pdf)
- Images of the neural network structure
- Backprop computation graph image + derivatives
    * Also when using Tanh
- 'safe' softmax function

### Cross validation


```julia
function cross_validation(X, Y, split_prop = 0.05)
    
    #=
    Create train/test split
    
    :param X: Input data X
    :param Y: output data Y
    :param split_prop: percentage of data to use as validation set
    
    :return: dictionary with train & validation data
    :seealso: https://www.coursera.org/learn/deep-neural-network/lecture/cxG1s/train-dev-test-sets
    =#
    
    # Number of training examples
    trainsize = size(X)[1]
    
    # Number of validation examples
    ndev = Integer(floor(trainsize * split_prop))
    
    # Shuffle indices
    ind = shuffle(1:trainsize);
    # Rearrange train x and y
    X = X[ind, :];
    Y = Y[ind, :];

    # Validation split
    dev_x = X[1:ndev, :];
    dev_y = Y[1:ndev, :];

    # Remove from train
    train_x = X[ndev+1:end, :];
    train_y = Y[ndev+1:end, :];
    
    # To dict
    split = Dict(
        "X_train" => train_x,
        "Y_train" => train_y,
        "X_dev" => dev_x,
        "Y_dev" => dev_y
    )
    
    # Return
    return(split)
    
    end;
```

### Weight initialization


```julia
function initialize_layer(n_h_current, n_h_last, activation, layer; drop_chance = 0.2, last=false)
    
    #=
    Initialize a layer
    =#
    
    # Weights
    W = randn(n_h_current, n_h_last)
    b = zeros(n_h_current, 1)

    # Weight initialization
    if activation == "ReLU"

        W = W .* sqrt(2 / n_h_last)

    else

        W = W .* sqrt(1 / n_h_last)

        end;
    
    # Initialization for Adam / RMSprop
    vdW = zeros(n_h_current, n_h_last)
    vdb = zeros(n_h_current, 1)
    
    sdW = zeros(n_h_current, n_h_last)
    sdb = zeros(n_h_current, 1)
    
    # To dict
    current_layer = Dict(
        "layer" => layer,
        "last_layer" => last,
        "activation" => activation,
        "hidden_current" => n_h_current,
        "hidden_previous" => n_h_last,
        "W" => W,
        "b" => b,
        "vdW" => vdW,
        "vdb" => vdb,
        "sdW" => sdW,
        "sdb" => sdb
    )
    
    # If last_layer == false, add drop chance
    if last == false
        
        current_layer["keep_prob"] = 1 - drop_chance
        
        end;
    
    # Return
    return(current_layer)
    
    end;

function initialize_parameters(n_x, n_h, n_y, activations, layers, dropout)
    
    #=
    Initialize the weight matrices and bias vectors
    
    :param n_x: number of observations in the input data
    :param n_h1: number of hidden units in the first layer
    :param n_h2: number of hidden units in the second layer
    :param n_y: number of labels in the output data
    
    :return:
       - parameters: dict containing weights W and bias vectors b for each layer
       - cache:      dict containing the moving average of the gradient (first moment) and moving average of the second gradient (second moment)
    
    :seealso: 
       - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/XtFPI/random-initialization
       - https://www.coursera.org/learn/deep-neural-network/lecture/C9iQO/vanishing-exploding-gradients
       - https://www.coursera.org/learn/deep-neural-network/lecture/RwqYe/weight-initialization-for-deep-networks
       - https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
       - https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
    =#
    
    # For sigmoid / tanh / softmax --> Xavier weight initialization --> 1 / sqrt(units in previous layer)
    # For ReLU weight initialization --> 2 / sqrt(units in previous layer)
    
    # List to hold results
    layer_details = []
    
    # Add 1 to layers (last layer is the softmax layer)
    layers = layers + 1
    
    # For each layer, initialize
    for layer = 1:layers
        
        # If first layer, then pass the value of n_x
        if layer == 1
            
            # Expect a tuple if layers > 2
            current_layer = initialize_layer(ifelse(layers <= 2, 
                                                    n_h, 
                                                    n_h[1]), 
                                             n_x, 
                                             ifelse(layers <= 2, 
                                                    activations, 
                                                    activations[layer]), 
                                             layer,
                                             drop_chance = ifelse(layers <= 2, 
                                                            dropout, 
                                                            dropout[layer]))
          
        # If last layer, then make it a softmax and pass n_y
        elseif(layer == layers)
            
            current_layer = initialize_layer(n_y, 
                                             ifelse(layers <= 2,
                                                    n_h,
                                                    n_h[layer-1]), 
                                             "softmax",
                                             layer, 
                                             last=true)
        
        # Else, we have the hidden layers in between input X and output Y
        else
            
            current_layer = initialize_layer(n_h[layer], n_h[layer-1], activations[layer], layer,
                                             drop_chance = dropout[layer])
            
            end;
        
        # Push layer
        push!(layer_details, current_layer)
        
        end;
    
    # Return
    return(layer_details)
    
    end;
```


```julia
# Multiple layers
layers_multiple = initialize_parameters(784, (256, 128), 10, ("ReLU", "ReLU"), 2, (0.2, 0.4));
# Single layer
layers_single = initialize_parameters(784, 256, 10, "ReLU", 1, 0.5);
```


```julia
layers_multiple[2]
```




    Dict{String,Any} with 12 entries:
      "W"               => [0.037904 0.0099701 … 0.060304 0.0483584; 0.134146 -0.08…
      "b"               => [0.0; 0.0; … ; 0.0; 0.0]
      "keep_prob"       => 0.6
      "vdW"             => [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0…
      "sdb"             => [0.0; 0.0; … ; 0.0; 0.0]
      "hidden_current"  => 128
      "layer"           => 2
      "activation"      => "ReLU"
      "sdW"             => [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0…
      "hidden_previous" => 256
      "last_layer"      => false
      "vdb"             => [0.0; 0.0; … ; 0.0; 0.0]




```julia
layers_single[1]
```




    Dict{String,Any} with 12 entries:
      "W"               => [-0.0520756 0.0434059 … 0.0186072 0.0272161; 0.0802813 -…
      "b"               => [0.0; 0.0; … ; 0.0; 0.0]
      "keep_prob"       => 0.5
      "vdW"             => [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0…
      "sdb"             => [0.0; 0.0; … ; 0.0; 0.0]
      "hidden_current"  => 256
      "layer"           => 1
      "activation"      => "ReLU"
      "sdW"             => [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0…
      "hidden_previous" => 784
      "last_layer"      => false
      "vdb"             => [0.0; 0.0; … ; 0.0; 0.0]



### Mini-batch gradient descent

An algorithm for splitting the data into batches is given below. Here:

- $n$ is the number of training examples
- $k$ is the batch size
- $j$ is the number of total batches that can be created **using the batch size $k$**

That is, $j$ is the integer part of the division of $\frac{n}{k}$.

<img src="img/minibatch.png" width = "500px">

This toy example is given as an algorithm below


```julia
# Number of examples
n = 101;
# Batch size
k = 10;
# Number of batches of size k (discounting any remainders)
j = floor(n/k);

# For each minibatch
for i = 1:j
    
    # Start at 1 if minibatch is 1
    if i == 1
        index_begin = 1
    else
        # Else start at the end of the previous minibatch plus one
        index_begin = Integer(((i - 1) * k) + 1)
        end;
    
    # End 
    index_end = Integer((i * k))
    
    println("Minibatch ", Integer(i), "\n\t[Begin ", index_begin, "] ==> [end ", index_end, "]")
    
    end;

# If there is a remainder
if n - (j * k) > 0
    
    index_begin = Integer((j * k) + 1)
    index_end = n
    
    println("Minibatch ", Integer(j+1), "\n\t[Begin ", index_begin, "] ==> [end ", index_end, "]")
    
    end;
```

    Minibatch 1
    	[Begin 1] ==> [end 10]
    Minibatch 2
    	[Begin 11] ==> [end 20]
    Minibatch 3
    	[Begin 21] ==> [end 30]
    Minibatch 4
    	[Begin 31] ==> [end 40]
    Minibatch 5
    	[Begin 41] ==> [end 50]
    Minibatch 6
    	[Begin 51] ==> [end 60]
    Minibatch 7
    	[Begin 61] ==> [end 70]
    Minibatch 8
    	[Begin 71] ==> [end 80]
    Minibatch 9
    	[Begin 81] ==> [end 90]
    Minibatch 10
    	[Begin 91] ==> [end 100]
    Minibatch 11
    	[Begin 101] ==> [end 101]


The code below implements mini-batches for real data


```julia
function create_batches(X, Y, batch_size = 128)
    
    #=
    Creates b batches of size batch_size
    
    :param X:          input data X
    :param Y:          output data Y
    :param batch_size: size of each batch
    
    :return: list containing m batches of length batch_size and possibly 1 batch of size batch_size_remainder < batch_size
    :seealso: 
      - https://www.coursera.org/learn/deep-neural-network/lecture/qcogH/mini-batch-gradient-descent
      - https://www.coursera.org/learn/deep-neural-network/lecture/lBXu8/understanding-mini-batch-gradient-descent
    =#
    
    # Shuffle training examples randomly
    n = size(X)[1]
    ind = shuffle(1:n)
    
    # Rearrange data in X and Y
    X = X[ind, :]
    Y = Y[ind, :]
    
    # List to store minibatches
    mbatches = []
    
    # Number of complete training examples. 
    #  This means: n / 128 leaves a remainder (most likely)
    #  Therefore, there will be (b - 1) batches with size batch_size
    #  and 1 batch with size last_batch_size < batch_size
    b_first = Integer(floor(n / batch_size))
    
    # First, loop through the (b_first - 1) examples
    #  We need to loop through b_first - 1 because we construct 
    #    @ index_begin => i * batch_size (e.g. 9 * 128 = 1152)
    #    @ index_end => (i + 1) * batch_size (e.g. 10 * 128 = 1280)
    #  index_end needs (i + 1). If i == k where k is the last possible index, we cannot subset index_end
    #  because it would require (k + 1) indices.
    for i = 1:b_first
        
        # Beginning and end indices
        if i == 1
            index_begin = 1
        else
            index_begin = Integer(((i - 1) * batch_size) + 1)
            end;
        
        index_end = Integer(i * (batch_size))
        
        X_current_batch = X[index_begin:index_end, :]
        Y_current_batch = Y[index_begin:index_end, :]

        # Add to array of minibatches
        push!(mbatches, [X_current_batch, Y_current_batch])
        
        end;
    
    # Then, if necessary, make a batch for the remainder
    b_rem = n - (b_first * batch_size)
    if b_rem != 0
        
        # Subset X & Y
        index_begin = Integer(((b_first) * batch_size) + 1) # i+1 is from the for-loop above (i.e. the last 'full' minibatch of size batch_size)
        index_end = n 
        
        X_current_batch = X[index_begin:index_end, :]
        Y_current_batch = Y[index_begin:index_end, :]
        
        # Append
        push!(mbatches, [X_current_batch, Y_current_batch])
        
        end;
    
    # Return mini batches
    return(mbatches)
    
    end;
```

### Activation functions and Loss function


```julia
function softmax(z)
    
    #=
    Softmax function
    
    :param z: result of the linear transformation of the final layer
    :return: activations for z
    
    :seealso:
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/4dDC1/activation-functions
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/OASKH/why-do-you-need-non-linear-activation-functions
      - https://www.coursera.org/learn/deep-neural-network/lecture/HRy7y/softmax-regression
      - https://www.coursera.org/learn/deep-neural-network/lecture/LCsCH/training-a-softmax-classifier
    =#
    
    # Subtract the max of z to avoid exp(z) getting too large
    z = z .- maximum(z)
    
    # Softmax & return
    return(exp.(z) ./ sum(exp.(z),dims=1))
    
    end;

function sigmoid(z)
    
    #=
    Sigmoid function
    
    :param z: result of the linear transformation for layer l
    :return: activations for z
    
    :seealso:
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/4dDC1/activation-functions
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/OASKH/why-do-you-need-non-linear-activation-functions
    =#
    
    1 ./ (1 .+ exp.(-z))
    
    end;

function ReLU(z)
    
    #=
    ReLU implementation
    
    :param z: result of the linear transformation for layer l
    :return: activations for z
    
    :seealso:
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/4dDC1/activation-functions
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/OASKH/why-do-you-need-non-linear-activation-functions
    =#
    
    ((z .> 0) * 1) .* z
    
    end;

function crossentropy_cost(y, yhat)
    
    #=
    Crossentropy cost function for m classes
    
    :param y: actual (one-hot encoded) values for y
    :param yhat: predicted (one-hot encoded) values for yhat
    
    :return: loss value
    
    :seealso: https://www.coursera.org/learn/neural-networks-deep-learning/lecture/yWaRd/logistic-regression-cost-function
    =#
    
    loss = sum(transpose(y) .* log.(yhat))
    n = size(y)[1]
    
    return(-(1/n) * loss)
    
    end;
```

### Dropout regularization


```julia
# Vectorized dropout
function dropout(X, keep_prob = 0.8)
    
    #=
    Dropout implementation. Randomly sets the weights of some hidden units to 0
    
    :param X: input data
    :param keep_prob: probability of keeping a hidden unit
    
    :return: dropout matrix containing 1 for each unit that should be kept and 0 for each unit that should be 
              dropped. also returns X in which some units are dropped and the remaining units are scaled by 
              drop_chance.
    
    :seealso:
      - https://www.coursera.org/learn/deep-neural-network/lecture/eM33A/dropout-regularization
      - https://www.coursera.org/learn/deep-neural-network/lecture/YaGbR/understanding-dropout
    =#
    
    n = size(X)[1]
    k = size(X)[2]
    
    # Uniformly distributed probabilities
    a = rand(n, k)
    
    # Evaluate against dropout probability
    #  run the mask and turn boolean into integer
    D = (a .< keep_prob) * 1

    # Element-wise multiplication
    X = D .* X

    # Scale
    X = X ./ keep_prob
    
    # Return
    return((D, X))
    
    end;
```

### Forward propagation

<img src="img/dims.png" width = "300px">

<img src="img/forwardprop.png" width = "800px">


```julia
function forward_prop(layers, X)
    
    #=
    Forward propagation
    
    :param parameters:  dict containing weights and bias vectors for each layer
    :param cache:       dict containing computations, dropout vectors and exponentially weighted gradients
    :param X:           input data
    :param keep_prob:   probability of keeping a hidden unit
    
    :return: updated cache with dropout vectors and the computations for each forward pass for the current 
              minibatch
    
    :seealso:
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/4WdOY/computation-graph
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/tyAGh/computing-a-neural-networks-output
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/MijzH/forward-propagation-in-a-deep-network
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/Rz47X/getting-your-matrix-dimensions-right
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/znwiG/forward-and-backward-propagation
    =#
    
    # For each layer, compute forward propagation step
    for layer in layers
        
        # Layer number
        layer_number = layer["layer"]
        # Last layer?
        last_layer = layer["last_layer"]
        # Activation function
        activation = layer["activation"]
        
        ## Compute the linear transformation Z
        
        # If first layer, then use X (also known as A_0)
        # Else: use A_{layer_number - 1}
        
        if layer_number == 1
                        
            # Multiply weights times X
            layer["Z"] = broadcast(+, layer["W"] * transpose(X), layer["b"], 1)
            
        else
            
            # Multiply weights times the activation of the previous layer
            layer["Z"] = broadcast(+, layer["W"] * layers[layer_number - 1]["A"], layer["b"], 1)
            
            end;
        
        ## Activation functions over Z ==> raise error if unknown
        
        if activation == "ReLU"
            
            A = ReLU(layer["Z"])
            
        elseif activation == "sigmoid"
            
            A = sigmoid(layer["Z"])
            
        elseif activation == "tanh"
            
            A = tanh.(layer["Z"])
            
        elseif activation == "softmax"
            
            A = softmax(layer["Z"])
            
        else
            
            # Raise error
            error(string("Don't know how to handle activation function '", activation, "'"))
        
            end;
        
        ## If last layer, then simply store the activations

        if last_layer
            
            layer["A"] = A

        ## Else, we apply dropout to the layer
        else
            
            D, A = dropout(A, layer["keep_prob"])
            
            # Store the data in the layer
            layer["D"] = D
            layer["A"] = A
            
            end;
        
        end;
    
    # Return layers
    return(layers)
    
    end;
```


```julia
# One step of forward propagation
X_tst = train_x[1:1000,:]
layers = forward_prop(layers_multiple, X_tst);
```


```julia
layers[3]
```




    Dict{String,Any} with 13 entries:
      "Z"               => [6.14783 1.56987 … 3.4709 3.45912; 3.41114 2.90086 … 2.9…
      "W"               => [0.0498594 -0.0533071 … -0.0170837 0.0772881; -0.0812939…
      "b"               => [0.0; 0.0; … ; 0.0; 0.0]
      "vdW"             => [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0…
      "sdb"             => [0.0; 0.0; … ; 0.0; 0.0]
      "hidden_current"  => 10
      "layer"           => 3
      "activation"      => "softmax"
      "sdW"             => [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0…
      "A"               => [0.835874 0.0263893 … 0.255762 0.0433476; 0.0541511 0.09…
      "hidden_previous" => 128
      "last_layer"      => true
      "vdb"             => [0.0; 0.0; … ; 0.0; 0.0]



### Backpropagation


```julia
# Derivatives of activation functions
function dSigmoid(z)
    
    #=
    Derivative of the sigmoid function
    
    :param z: result of the linear transformation computed during forward propagation. Generally:
                   Z = W_lA_{l-1} + b_l
    :return: derivative of the activation function
    
    :seealso: 
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/qcG1j/derivatives-of-activation-functions
      - https://deepnotes.io/softmax-crossentropy
      - https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    =#
    
    sigmoid(z) .* (1 .- sigmoid(z))
    
    end;

function dReLU(z)
    
    #=
    Derivative of ReLU function
    
    :param z: result of the linear transformation computed during forward propagation. Generally:
                   Z = W_lA_{l-1} + b_l
    :return: derivative of the activation function
    
    :seealso: https://www.coursera.org/learn/neural-networks-deep-learning/lecture/qcG1j/derivatives-of-activation-functions
    =#
    
    (z .> 0) * 1
    
    end;

function dTanh(z)
    
    #=
    Derivative of tanh function
    
    :param z: result of the linear transformation computed during forward propagation. Generally:
                   Z = W_lA_{l-1} + b_l
    :return: derivative of the activation function
    
    :seealso: https://www.coursera.org/learn/neural-networks-deep-learning/lecture/qcG1j/derivatives-of-activation-functions
    =# 
    
    1 .- (tanh.(z) .* tanh.(z))
    
    end;
```


```julia
function backward_prop(layers, X, Y)
    
    #=
    Backward propagation
    
    :param parameters:  dict containing weights and bias vectors for each layer
    :param cache:       dict containing computations, dropout vectors and exponentially weighted gradients
    :param X:           input data
    :param drop_chance: probability of keeping a hidden unit
    
    :return: updated cache with new gradients for the current minibatch
    
    :seealso:
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/0ULGt/derivatives
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/oEcPT/more-derivative-examples
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/4WdOY/computation-graph
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/0VSHe/derivatives-with-a-computation-graph
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/6dDj7/backpropagation-intuition-optional
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/znwiG/forward-and-backward-propagation
      Docs about dims argument for sum()
      - https://docs.julialang.org/en/v0.6.1/stdlib/collections/#Base.sum
    =#
    
    # Dims
    n = size(Y)[1]
    
    # For layers (reverse)
    for layer_number = length(layers):-1:1
        
        # Last layer?
        last_layer = layers[layer_number]["last_layer"]
        # Activation function
        activation = layers[layer_number]["activation"]
        
        ## Last layer derivatives (softmax)
        
        if last_layer
            
            # Softmax derivative
            layers[layer_number]["dZ"] = layers[layer_number]["A"] .- transpose(Y)
            layers[layer_number]["dW"] = (1/n) .* (layers[layer_number]["dZ"] * transpose(layers[layer_number - 1]["A"]))
            layers[layer_number]["db"] = (1/n) .* sum(layers[layer_number]["dZ"], dims=2)
            
        ## First layer derivatives
            
        elseif layer_number == 1
            
            # Activations for layer 1
            dA = transpose(layers[layer_number + 1]["W"]) * layers[layer_number + 1]["dZ"]
            # Apply dropout
            dA = (layers[layer_number]["D"] .* dA) ./ layers[layer_number]["keep_prob"] 
            
            # Derivative of activation function
            if activation == "ReLU"

                layers[layer_number]["dZ"] = dA .* dReLU(layers[layer_number]["Z"])

            # Sigmoid derivative
            elseif activation == "sigmoid"

                layers[layer_number]["dZ"] = dA .* dSigmoid(layers[layer_number]["Z"])

            elseif activation == "tanh"

                layers[layer_number]["dZ"] = dA .* dTanh(layers[layer_number]["Z"])

                end;  
        
            layers[layer_number]["dA"] = dA
            # Linear combination derivative
            layers[layer_number]["dW"] = (1/n) .* (layers[layer_number]["dZ"] * X)
            layers[layer_number]["db"] = (1/n) .* sum(layers[layer_number]["dZ"], dims=2)            
            
        ## Intermediate layer derivatives
            
        else
            
            dA = transpose(layers[layer_number + 1]["W"]) * layers[layer_number + 1]["dZ"]
            # Apply dropout
            dA = (layers[layer_number]["D"] .* dA) ./ layers[layer_number]["keep_prob"]
            
            # Derivative of activation function
            if activation == "ReLU"

                layers[layer_number]["dZ"] = dA .* dReLU(layers[layer_number]["Z"])

            # Sigmoid derivative
            elseif activation == "sigmoid"

                layers[layer_number]["dZ"] = dA .* dSigmoid(layers[layer_number]["Z"])

            elseif activation == "tanh"

                layers[layer_number]["dZ"] = dA .* dTanh(layers[layer_number]["Z"])

                end;
        
            layers[layer_number]["dA"] = dA
            # Linear combination derivative
            layers[layer_number]["dW"] = (1/n) .* (layers[layer_number]["dZ"] * transpose(layers[layer_number-1]["A"]))
            layers[layer_number]["db"] = (1/n) .* sum(layers[layer_number]["dZ"], dims=2)
            
            end;
        
        end;

    return(layers)

    end;
```


```julia
layers = backward_prop(layers, X_tst, y_train_ohe[1:1000, :]);
```


```julia
layers[2]
```




    Dict{String,Any} with 19 entries:
      "Z"               => [2.40873 0.902895 … 2.44057 1.46958; -1.42964 -3.08879 ……
      "W"               => [0.037904 0.0099701 … 0.060304 0.0483584; 0.134146 -0.08…
      "dZ"              => [0.195609 -0.0 … 0.0 0.0706079; -0.0 0.0 … -0.182764 0.0…
      "dW"              => [0.0488287 0.0408837 … 0.0247196 0.0213366; 0.000983771 …
      "b"               => [0.0; 0.0; … ; 0.0; 0.0]
      "keep_prob"       => 0.6
      "vdW"             => [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0…
      "sdb"             => [0.0; 0.0; … ; 0.0; 0.0]
      "hidden_current"  => 128
      "layer"           => 2
      "activation"      => "ReLU"
      "db"              => [0.0309754; 0.00034264; … ; -0.0188653; 0.00783569]
      "sdW"             => [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0…
      "A"               => [4.01455 0.0 … 0.0 2.4493; -0.0 -0.0 … 1.02054 -0.0; … ;…
      "hidden_previous" => 256
      "D"               => [1 0 … 0 1; 1 1 … 1 0; … ; 0 0 … 0 0; 1 1 … 0 0]
      "dA"              => [0.195609 -0.0 … 0.0 0.0706079; -0.211335 0.311394 … -0.…
      "last_layer"      => false
      "vdb"             => [0.0; 0.0; … ; 0.0; 0.0]



### Updating weights: Adam & RMSprop


```julia
function Adam(layers; alpha = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
    
    #=
    Update parameters using Adam optimization algorithm
    
    :param parameters:  dict containing weights and bias vectors for each layer
    :param cache:       dict containing computations, dropout vectors and exponentially weighted gradients
    :param alpha:       learning rate
    :param beta:        exponential weighting value. Usually set at 0.9 <MORE>
    :param epsilon:     small value to prevent divide by zero
    
    :return: updates weights in cache for current minibatch
    
    :seealso:
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/A0tBd/gradient-descent
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/udiAq/gradient-descent-on-m-examples
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/Wh8NI/gradient-descent-for-neural-networks
      - https://www.coursera.org/learn/deep-neural-network/lecture/duStO/exponentially-weighted-averages
      - https://www.coursera.org/learn/deep-neural-network/lecture/Ud7t0/understanding-exponentially-weighted-averages
      - https://www.coursera.org/learn/deep-neural-network/lecture/XjuhD/bias-correction-in-exponentially-weighted-averages
      - https://www.coursera.org/learn/deep-neural-network/lecture/y0m1f/gradient-descent-with-momentum
      - https://www.coursera.org/learn/deep-neural-network/lecture/BhJlm/rmsprop
      - https://www.coursera.org/learn/deep-neural-network/lecture/w9VCZ/adam-optimization-algorithm
      - https://www.coursera.org/learn/deep-neural-network/notebook/eNLYh/optimization
    =#
    
    # For each layer, update parameters
    for layer in layers
        
        # Momentum
        layer["vdW"] = (beta1 .* layer["vdW"] .+ (1 - beta1) .* layer["dW"]) ./ (1 / beta1^2)
        layer["vdb"] = (beta1 .* layer["vdb"] .+ (1 - beta1) .* layer["db"]) ./ (1 / beta1^2)
        
        # RMSprop
        layer["sdW"] = (beta2 .* layer["sdW"] .+ (1 - beta2) .* layer["dW"].^2) ./ (1 / beta2^2)
        layer["sdb"] = (beta2 .* layer["sdb"] .+ (1 - beta2) .* layer["db"].^2) ./ (1 / beta2^2)
        
        # Update parameters
        layer["W"] = layer["W"] .- (alpha .* (layer["vdW"] ./ (sqrt.(layer["sdW"] .+ epsilon))))
        layer["b"] = layer["b"] .- (alpha .* (layer["vdb"] ./ (sqrt.(layer["sdb"] .+ epsilon))))
        
        end;
    
    # Return layers
    return(layers)
    
    end;
```


```julia
function RMSprop(layers; alpha = 0.001, beta = 0.9, epsilon = 1e-8)
    
    #=
    Update parameters using RMSProp optimization algorithm
    
    :param parameters:  dict containing weights and bias vectors for each layer
    :param cache:       dict containing computations, dropout vectors and exponentially weighted gradients
    :param alpha:       learning rate
    :param beta:        exponential weighting value. Usually set at 0.9 <MORE>
    :param epsilon:     small value to prevent divide by zero
    
    :return: updates weights in cache for current minibatch
    
    :seealso:
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/A0tBd/gradient-descent
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/udiAq/gradient-descent-on-m-examples
      - https://www.coursera.org/learn/neural-networks-deep-learning/lecture/Wh8NI/gradient-descent-for-neural-networks
      - https://www.coursera.org/learn/deep-neural-network/lecture/duStO/exponentially-weighted-averages
      - https://www.coursera.org/learn/deep-neural-network/lecture/Ud7t0/understanding-exponentially-weighted-averages
      - https://www.coursera.org/learn/deep-neural-network/lecture/XjuhD/bias-correction-in-exponentially-weighted-averages
      - https://www.coursera.org/learn/deep-neural-network/lecture/y0m1f/gradient-descent-with-momentum
      - https://www.coursera.org/learn/deep-neural-network/lecture/BhJlm/rmsprop
      - https://www.coursera.org/learn/deep-neural-network/lecture/w9VCZ/adam-optimization-algorithm
      - https://www.coursera.org/learn/deep-neural-network/notebook/eNLYh/optimization
    =#
    
    # For each layer, update parameters
    for layer in layers
        
        # RMSprop
        layer["sdW"] = (beta .* layer["sdW"] .+ (1 - beta) .* layer["dW"].^2) ./ (1 / beta^2)
        layer["sdb"] = (beta .* layer["sdb"] .+ (1 - beta) .* layer["db"].^2) ./ (1 / beta^2)
        
        # Update parameters
        layer["W"] = layer["W"] .- (alpha .* (layer["dW"] ./ (sqrt.(layer["sdW"] .+ epsilon))))
        layer["b"] = layer["b"] .- (alpha .* (layer["db"] ./ (sqrt.(layer["sdb"] .+ epsilon))))
        
        end;
    
    # Return layers
    return(layers)
    
    end;
```


```julia
layers = Adam(layers);
```


```julia
layers[2]
```




    Dict{String,Any} with 19 entries:
      "Z"               => [2.40873 0.902895 … 2.44057 1.46958; -1.42964 -3.08879 ……
      "W"               => [0.0353454 0.00741374 … 0.0577608 0.0458222; 0.133385 -0…
      "dZ"              => [0.195609 -0.0 … 0.0 0.0706079; -0.0 0.0 … -0.182764 0.0…
      "dW"              => [0.0488287 0.0408837 … 0.0247196 0.0213366; 0.000983771 …
      "b"               => [-0.00255072; -0.000275927; … ; 0.00252866; -0.00237735]
      "keep_prob"       => 0.6
      "vdW"             => [0.00395512 0.00331158 … 0.00200229 0.00172827; 7.96854e…
      "sdb"             => [9.57555e-7; 1.17168e-10; … ; 3.55189e-7; 6.12752e-8]
      "hidden_current"  => 128
      "layer"           => 2
      "activation"      => "ReLU"
      "db"              => [0.0309754; 0.00034264; … ; -0.0188653; 0.00783569]
      "sdW"             => [2.37947e-6 1.66813e-6 … 6.09837e-7 4.54341e-7; 9.6587e-…
      "A"               => [4.01455 0.0 … 0.0 2.4493; -0.0 -0.0 … 1.02054 -0.0; … ;…
      "hidden_previous" => 256
      "D"               => [1 0 … 0 1; 1 1 … 1 0; … ; 0 0 … 0 0; 1 1 … 0 0]
      "dA"              => [0.195609 -0.0 … 0.0 0.0706079; -0.211335 0.311394 … -0.…
      "last_layer"      => false
      "vdb"             => [0.002509; 2.77538e-5; … ; -0.00152809; 0.000634691]



### Learning rate decay


```julia
function learning_rate_decay(alpha, epoch, decay = 1)
    
    #= 
    Decays the value of alpha (learning rate) as number of epochs increases
    
    :param alpha: learning rate 
    :param epoch: current epoch number
    :param decay: decay value
    
    :return: updated learning rate
    
    :seealso: https://www.coursera.org/learn/deep-neural-network/lecture/hjgIA/learning-rate-decay
    =#
    
    alpha_new = 1 / (1 + decay * epoch) * alpha
    
    # Return
    return(alpha_new)
    
    end;

```

### Evaluation functions


```julia
function predict(X, layers)
    
    #=
    Predict Y given X and the parameters
    
    :param X:          input data
    :param parameters: parameters for each layer
    
    :return: predicted values given X and parameters
    =#
    
    # Define scope for A_previous and A
    local A_prev
    local A
    
    # For each layer, compute forward propagation step
    for layer in layers
        
        # Layer number
        layer_number = layer["layer"]
        # Last layer?
        last_layer = layer["last_layer"]
        # Activation function
        activation = layer["activation"]
        
        # If first layer
        if layer_number == 1
            
            # Multiply weights times X
            Z = broadcast(+, layer["W"] * transpose(X), layer["b"], 1)
            
        else
            
            # Multiply weights times the activation of the previous layer
            Z = broadcast(+, layer["W"] * A_prev, layer["b"], 1)
            
            end;
        
        # Activation function
        if activation == "ReLU"
            
            A = ReLU(Z)
            A_prev = A
            
        elseif activation == "sigmoid"
            
            A = sigmoid(Z)
            A_prev = A
            
        elseif activation == "tanh"
            
            A = tanh.(Z)
            A_prev = A
            
        elseif activation == "softmax"
            
            A = softmax(Z)
            
            end;
                
        end;
    
    # Return
    return(A)
    
    end;

function accuracy(Y, Yhat)
    
    #=
    Calculates accuracy
    
    :param Y:    actual outcome values (as one-hot encoded) 
    :param Yhat: predicted outcome values (as one-hot encoded)
    
    :return: accuracy of the predictions as a float between 0 and 1
    =# 
    
    # Predictions --> labels
    vals, inds = findmax(Yhat, dims = 1);
    Yhat_labels = map(x -> x[1] - 1, inds)

    # Y ohe --> labels
    vals, inds = findmax(transpose(Y), dims=1)
    Y_labels = map(x -> x[1] - 1, inds)
    
    # Accuracy
    return(sum(Y_labels .== Yhat_labels) / length(Y_labels))
    
    end;

function evaluate_model(X, Y, layers; return_accuracy = false)
    
    #= 
    Evaluate the model on the development & train sets
    
    :param X:               input data 
    :param Y:               output data
    :param parameters:      dict containing parameters for each layer
    :param return_accuracy: boolean. If true, returns loss and accuracy. If false, only returns loss
    
    :return: either loss and accuracy or only loss value
    =#
    
    # Predict
    Yhat = predict(X, layers)
    
    # Loss
    loss = crossentropy_cost(Y, Yhat)
    
    # Accuracy
    if return_accuracy
        acc = accuracy(Y, Yhat)
        return((loss, acc))
        
        end;
    
    # Return
    return(loss)
    
    end;

using Plots

function plot_history(history)
    
    #=
    Plot loss across epochs
    
    :param history: nn history returned by nnet function
    
    :return: this function plots the history but does not return a value 
    
    :seealso: https://docs.juliaplots.org/latest/tutorial/#plot-recipes-and-recipe-libraries
    =#
    
    # Retrieve data
    epoch = [x[1] for x in history]
    train_loss = [x[2] for x in history]
    dev_loss = [x[3] for x in history]
    # Train and dev loss in one array (2 columns)
    loss = hcat(train_loss, dev_loss);
    
    # Plot epochs versus loss
    plot(epoch, loss, seriestype=:line, xlabel = "epoch", ylabel = "cost", lab = ["Train" "Dev"],
         lw = 2)
    
    end;
```

    ┌ Info: Recompiling stale cache file /home/jasper/.julia/compiled/v1.0/Plots/ld3vC.ji for Plots [91a5bcdd-55d7-5caf-9e0b-520d859cae80]
    └ @ Base loading.jl:1190


### Testing the functions


```julia
# Initialize a single layer with relu activation and no dropout
layers = initialize_parameters(784, 256, 10, "ReLU", 1, 0);
```


```julia
# For clarity
X = train_x
Y = y_train_ohe

# Train the model for 5 steps to see if the cost goes down
for i = 1:5
    
    layers = forward_prop(layers, X[1:1000,:])
    layers = backward_prop(layers, X[1:1000,:], Y[1:1000,:])
    layers = RMSprop(layers)
    cost, acc = evaluate_model(X[1:1000,:], Y[1:1000,:], layers, return_accuracy=true)
    
    # Print
    println("[Cost: ", cost, "] ==> [Accuracy: ", acc, "]")
    
    end;

```

    [Cost: 2.1672560037581707] ==> [Accuracy: 0.199]
    [Cost: 1.6499699252559397] ==> [Accuracy: 0.52]
    [Cost: 1.3718493237236906] ==> [Accuracy: 0.64]
    [Cost: 1.2124436308872937] ==> [Accuracy: 0.689]
    [Cost: 1.136326180769171] ==> [Accuracy: 0.706]


### Putting it all together: the nnet() function


```julia
function nnet(X, Y; n_h = 128, activations = "tanh", validation_split = 0.1, optimizer = "Adam", 
              alpha = 0.001, beta1=0.9, beta2 = 0.999, epsilon=10e-8, epochs = 10, batch_size=128, 
              lr_decay=1, dropout_prop=0.3)
    
    #= 
    Implements a two-layer neural network
    
    :param X:                   input data
    :param Y:                   actual outcome values (as one-hot encoded) 
    :param n_h1:                hidden units for layer 1
    :param n_h2:                hidden units for layer 2
    :param validation_split:    decimal indicating what percentage of the data should be reserved for 
                                 validation split
    :param alpha:               learning rate 
    :param beta:                exponential weighting value. Usually set at 0.9 <MORE>
    :param epsilon:             small value to prevent divide by zero in RMSprop 
    :param epochs:              number of passes (epochs) to train the network
    :param batch_size:          size of the minibatches
    :param lr_decay:            decay value for learning rate (alpha)
    :param dropout_prop:        probability of dropping a hidden unit when performing dropout regularization. 
                                 A value of 0 keeps all hidden units
    
    :return: final parameters for the trained model and training history
    =#
    
    ## Checks
    
    # hidden units, activations and dropout must all be tuple
    if all((isa(n_h, Tuple), isa(activations, Tuple), isa(dropout_prop, Tuple)))
        
        @assert length(n_h) == length(activations) "Number of hidden units must equal number of activations"
        @assert length(n_h) == length(dropout_prop) "Number of hidden units must equal the number of dropout probabilities"
        
        hidden_layers = length(n_h)
        
    else
        
        hidden_layers = 1
        
        end;
    
    # Set dimensions
    n = size(X)[1]
    n_x = size(X)[2]
    n_y = size(Y)[2]

    # Save alpha value
    alpha_initial = alpha

    # Initialize parameters
    layers = initialize_parameters(n_x, n_h, n_y, activations, hidden_layers, dropout_prop)
    
    # Open list for history
    history = []
    
    # Validation and train set
    data = cross_validation(X, Y, validation_split)

    # Unroll
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_dev = data["X_dev"]
    Y_dev = data["Y_dev"]

    # Loop through epochs
    for i = 1:epochs

        # Set up number of batches
        batches = create_batches(X_train, Y_train, batch_size)
        
        ## Mini-batches (inner loop)
        for batch in batches

            # Unroll X and y
            X = batch[1]
            Y = batch[2]
            
            # Forward prop
            layers = forward_prop(layers, X)
            
            # Backward propagation
            layers = backward_prop(layers, X, Y)
            
            # Update params
            if optimizer == "Adam"
                
                layers = Adam(layers, alpha = alpha, beta1 = beta1, beta2 = beta2, epsilon = epsilon)
                
            else
                
                layers = RMSprop(layers, alpha = alpha, beta = beta1, epsilon = epsilon)
                
                end;
            
            end;
        
        # Learning rate decay
        alpha = learning_rate_decay(alpha_initial, i, lr_decay)

        # Evaluate
        eval_train = evaluate_model(X_train, Y_train, layers)
        eval_dev, eval_acc = evaluate_model(X_dev, Y_dev, layers, return_accuracy=true)
        
        # Round
        eval_train = round(eval_train, digits=3)
        eval_dev = round(eval_dev, digits=3)

        # Print
        println("Epoch ", i, ":\n",
                "\t[train cost: ", eval_train, 
                "] ==> [validation cost: ", eval_dev, "]",
                "\n\tvalidation accuracy: ", round(eval_acc, digits=3))
        
        # Add to history
        push!(history, [i, eval_train, eval_dev])

        end;
    
    # Return the final parameters
    return((layers, history))
    
    end;
```

## Running the network


```julia
# Split test into dev (70%) test (30%)
Random.seed!(9726)
split = cross_validation(test_x, y_test_ohe, 0.3)
X_dev = split["X_train"]
Y_dev = split["Y_train"]
X_test = split["X_dev"]
Y_test = split["Y_dev"];
```

### Baseline model

The baseline model is basically a logistic regression model. It has a single hidden unit and a sigmoid activation function.


```julia
layers, history = nnet(X, Y, n_h = 1, activations = "sigmoid",
                       dropout_prop=0, batch_size = 512, alpha = 0.008, epochs=30,
                       lr_decay=0, validation_split = 0.1);
```

    Epoch 1:
    	[train cost: 1.955] ==> [validation cost: 1.962]
    	validation accuracy: 0.197
    Epoch 2:
    	[train cost: 1.894] ==> [validation cost: 1.902]
    	validation accuracy: 0.212
    Epoch 3:
    	[train cost: 1.866] ==> [validation cost: 1.873]
    	validation accuracy: 0.214
    Epoch 4:
    	[train cost: 1.85] ==> [validation cost: 1.859]
    	validation accuracy: 0.219
    Epoch 5:
    	[train cost: 1.837] ==> [validation cost: 1.846]
    	validation accuracy: 0.224
    Epoch 6:
    	[train cost: 1.828] ==> [validation cost: 1.839]
    	validation accuracy: 0.227
    Epoch 7:
    	[train cost: 1.819] ==> [validation cost: 1.828]
    	validation accuracy: 0.229
    Epoch 8:
    	[train cost: 1.81] ==> [validation cost: 1.823]
    	validation accuracy: 0.235
    Epoch 9:
    	[train cost: 1.8] ==> [validation cost: 1.812]
    	validation accuracy: 0.259
    Epoch 10:
    	[train cost: 1.791] ==> [validation cost: 1.803]
    	validation accuracy: 0.26
    Epoch 11:
    	[train cost: 1.775] ==> [validation cost: 1.789]
    	validation accuracy: 0.261
    Epoch 12:
    	[train cost: 1.761] ==> [validation cost: 1.773]
    	validation accuracy: 0.281
    Epoch 13:
    	[train cost: 1.745] ==> [validation cost: 1.757]
    	validation accuracy: 0.288
    Epoch 14:
    	[train cost: 1.738] ==> [validation cost: 1.751]
    	validation accuracy: 0.327
    Epoch 15:
    	[train cost: 1.719] ==> [validation cost: 1.73]
    	validation accuracy: 0.335
    Epoch 16:
    	[train cost: 1.708] ==> [validation cost: 1.719]
    	validation accuracy: 0.365
    Epoch 17:
    	[train cost: 1.696] ==> [validation cost: 1.707]
    	validation accuracy: 0.37
    Epoch 18:
    	[train cost: 1.686] ==> [validation cost: 1.698]
    	validation accuracy: 0.373
    Epoch 19:
    	[train cost: 1.679] ==> [validation cost: 1.691]
    	validation accuracy: 0.353
    Epoch 20:
    	[train cost: 1.67] ==> [validation cost: 1.683]
    	validation accuracy: 0.366
    Epoch 21:
    	[train cost: 1.662] ==> [validation cost: 1.674]
    	validation accuracy: 0.368
    Epoch 22:
    	[train cost: 1.654] ==> [validation cost: 1.665]
    	validation accuracy: 0.377
    Epoch 23:
    	[train cost: 1.646] ==> [validation cost: 1.659]
    	validation accuracy: 0.387
    Epoch 24:
    	[train cost: 1.64] ==> [validation cost: 1.653]
    	validation accuracy: 0.373
    Epoch 25:
    	[train cost: 1.637] ==> [validation cost: 1.651]
    	validation accuracy: 0.374
    Epoch 26:
    	[train cost: 1.629] ==> [validation cost: 1.642]
    	validation accuracy: 0.377
    Epoch 27:
    	[train cost: 1.621] ==> [validation cost: 1.635]
    	validation accuracy: 0.38
    Epoch 28:
    	[train cost: 1.614] ==> [validation cost: 1.628]
    	validation accuracy: 0.374
    Epoch 29:
    	[train cost: 1.609] ==> [validation cost: 1.621]
    	validation accuracy: 0.379
    Epoch 30:
    	[train cost: 1.606] ==> [validation cost: 1.62]
    	validation accuracy: 0.392



```julia
# Plot history
plot_history(history)
```




![svg](output_62_0.svg)




```julia
# Train accuracy
Yhat = predict(X, layers);
acc = accuracy(Y, Yhat)
```




    0.39995




```julia
# Dev accuracy
Yhat_dev = predict(X_dev, layers);
# Accuracy
acc_dev = accuracy(Y_dev, Yhat_dev)
```




    0.4005714285714286



THe baseline model achieves an accuracy of $40\%$ after $30$ epochs. Not great, but much better than a random guess.

### Single layer model with ReLU activation, low dropout probability and batch size 128

We can probably improve on the above model by using a different activation function and by changing the batch size.


```julia
layers_simple, history_simple = nnet(X, Y, n_h = 64, activations = "ReLU", batch_size = 128, dropout_prop=0.2,
                                     alpha = 0.002, epochs=12, lr_decay=0, validation_split = 0.1);
```

    Epoch 1:
    	[train cost: 0.225] ==> [validation cost: 0.229]
    	validation accuracy: 0.931
    Epoch 2:
    	[train cost: 0.172] ==> [validation cost: 0.182]
    	validation accuracy: 0.943
    Epoch 3:
    	[train cost: 0.136] ==> [validation cost: 0.15]
    	validation accuracy: 0.955
    Epoch 4:
    	[train cost: 0.115] ==> [validation cost: 0.134]
    	validation accuracy: 0.958
    Epoch 5:
    	[train cost: 0.1] ==> [validation cost: 0.121]
    	validation accuracy: 0.963
    Epoch 6:
    	[train cost: 0.088] ==> [validation cost: 0.113]
    	validation accuracy: 0.964
    Epoch 7:
    	[train cost: 0.08] ==> [validation cost: 0.105]
    	validation accuracy: 0.968
    Epoch 8:
    	[train cost: 0.072] ==> [validation cost: 0.101]
    	validation accuracy: 0.968
    Epoch 9:
    	[train cost: 0.066] ==> [validation cost: 0.098]
    	validation accuracy: 0.97
    Epoch 10:
    	[train cost: 0.061] ==> [validation cost: 0.094]
    	validation accuracy: 0.97
    Epoch 11:
    	[train cost: 0.059] ==> [validation cost: 0.096]
    	validation accuracy: 0.97
    Epoch 12:
    	[train cost: 0.052] ==> [validation cost: 0.088]
    	validation accuracy: 0.973



```julia
# Plot history
plot_history(history_simple)
```




![svg](output_68_0.svg)




```julia
# Train accuracy
Yhat = predict(X, layers_simple);
acc = accuracy(Y, Yhat)
```




    0.9835166666666667




```julia
# Dev accuracy
Yhat_dev = predict(X_dev, layers_simple);
# Accuracy
acc_dev = accuracy(Y_dev, Yhat_dev)
```




    0.9738571428571429



This model achieves $97.4\%$ accuracy on the development set. Towards the end of the training epochs it looks like it will start to overfit soon. Still, it is good performance for a small network with only 1 hidden layer.

### Multiple layers

The overfitting problem will become more pressing as we build deeper models. 


```julia
layers, history = nnet(X, Y, n_h = (128, 128), activations = ("ReLU", "ReLU"), 
                        optimizer = "Adam", dropout_prop=(0.3, 0.3), batch_size = 128, alpha = 0.003, 
                        epochs=12, lr_decay=0.1, validation_split = 0.1);
```

    Epoch 1:
    	[train cost: 0.155] ==> [validation cost: 0.179]
    	validation accuracy: 0.946
    Epoch 2:
    	[train cost: 0.109] ==> [validation cost: 0.138]
    	validation accuracy: 0.958
    Epoch 3:
    	[train cost: 0.09] ==> [validation cost: 0.122]
    	validation accuracy: 0.962
    Epoch 4:
    	[train cost: 0.071] ==> [validation cost: 0.104]
    	validation accuracy: 0.968
    Epoch 5:
    	[train cost: 0.064] ==> [validation cost: 0.104]
    	validation accuracy: 0.969
    Epoch 6:
    	[train cost: 0.056] ==> [validation cost: 0.096]
    	validation accuracy: 0.972
    Epoch 7:
    	[train cost: 0.048] ==> [validation cost: 0.089]
    	validation accuracy: 0.972
    Epoch 8:
    	[train cost: 0.043] ==> [validation cost: 0.089]
    	validation accuracy: 0.974
    Epoch 9:
    	[train cost: 0.039] ==> [validation cost: 0.086]
    	validation accuracy: 0.974
    Epoch 10:
    	[train cost: 0.037] ==> [validation cost: 0.087]
    	validation accuracy: 0.974
    Epoch 11:
    	[train cost: 0.032] ==> [validation cost: 0.08]
    	validation accuracy: 0.976
    Epoch 12:
    	[train cost: 0.032] ==> [validation cost: 0.084]
    	validation accuracy: 0.975



```julia
# Plot history
plot_history(history)
```




![svg](output_74_0.svg)




```julia
# Train accuracy
Yhat = predict(X, layers);
acc = accuracy(Y, Yhat)
```




    0.9888833333333333




```julia
# Dev accuracy
Yhat_dev = predict(X_dev, layers);
# Accuracy
acc_dev = accuracy(Y_dev, Yhat_dev)
```




    0.9765714285714285



$97.65$%, but clearly starting to overfit on the data and we can't really justify the extra cost in terms of running time when compared to the $0.15$% increase on the single layer model.

### Predict on test data


```julia
# Test accuracy
Yhat_tst = predict(X_test, layers_simple);
# Accuracy
acc_tst = accuracy(Y_test, Yhat_tst)
```




    0.9756666666666667


