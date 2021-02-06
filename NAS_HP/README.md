
# Running the NAS and HP

## Pre requisistes


* [NOMAD](https://www.gerad.ca/nomad/).
* Python > 3.6
* [PyTorch](https://pytorch.org/)
* GCC > 9.0

## The blackbox

The blackbox does the following:

* Read a series of values as input that define the architecture, batch size and training HPs.
* Creates a scaled version of a baseline network (ResNet18 for now).
* Trains the network on 3 subsets of CIFAR-10. Each subset represents 10% of the dataset
* Returns the mean best accuracy  as the performance measure.


### Run an example with the blackbox

You can run this command: 


```
    python blackbox.py depth_mult width_mult resolution batch_size optmizer lr arg1 arg2 weight_decay
``` 
Note that the interpretation of arg1 and arg2 depend on which optimizer is used. Here is a numerical example:

```
    python blackbox.py 1.25 0.34 1 128 2 0.1 0.9 0.99 0.0001 
``` 
Here it is the Adam optimizer and 0.9, 0,99 are the values of beta1 and beta2. 


## Launching a NOMAD optimization

### Parameter file

To launch the NOMAD optimization we need to define the search space through the parameter file. In this case it looks like this: 

```

DIMENSION               9                               # NOMAD optimizes 3 hyperparameters
BB_EXE                  "$python ./nomad_linker.py"    # The script that links NOMAD to this blackbox

BB_OUTPUT_TYPE          OBJ  -  - -                     # The blackbox returns 4 outputs: 
                                                        # OBJ : Mean best validation accuracy
                                                        #  - : std of best validation accuracy
                                                        #  - : Mean best training accuracy
                                                        #  - : std of best training accuracy
                                                        
BB_INPUT_TYPE           ( R  R  R I I R R R R )         # depth_mult width_mult resolution and Real hyperparameters
                                                        # batch size + Optimizer are Integer

X0                      ( 1  1  1 128 2 0.1 0.9 0.99 0.0001)    # NOMAD needs an initial starting point       

LOWER_BOUND             ( 0.5 0.125 0.125 10 1 0 0 0 0 )                 # Lower bound on d, w and r
UPPER_BOUND             ( 2.5  2.5  2.5  512 2 1 1 1 1)               # Upper bound on d, w and r

MAX_BB_EVAL             100                             # Number of blackbox evaluations
DISPLAY_DEGREE          3                               # Display all the logs of NOMAD
```

### Commad to run the optimization

To run the NOMAD optimization, we need to execute: 

```
$ nomad parameter_file.txt
```

