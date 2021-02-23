*****
# Implement HP tuning using Nomad
*****

## Goal

Find the best batch_size, learning_rate, weight_decay and optimizer.
Provide a parameter file to use Nomad to tune these parameters.

### Key points

Can run several seeds for the random sampling of the subset of images.
Each evaluation for a given seed is conducted in parallel.
Need to specify the GPU indices available in the nomad_linker_parallel.py

The design space for the HP is provided in the param.txt file.
A mapping is done in the blackbox_single.py between the variables handled by Nomad (integers) and the choice for HPs. For example, the first variable is the batch size; a value of 3 in Nomad is mapped to a batch size of 256.

To launch an optimization
$NOMAD_HOME/bin/nomad param.txt



 
