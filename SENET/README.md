# Scaling the SeNET network

## NAS phase

We use d, w, and r to scale the depth, width and resolution of the baseline network SeNET. 

The tests are done on an Nvidia GeForce RTX 2060. Every network trains during 1000 seconds 
instead of the required 600 seconds on an Nvidia P100.


| Network | d | w | r | Best validation score
| :---: | :---: | :---: | :---: |  :---: |
| SeNET | 1 | 1 | 1 | 84% | 
| Best solution | 0.83 | 0.54 | 1.03 | 86.9%

### Run a training with a given (d, w, r) 


```
python blackbox2.py d w r

```
### Run the NAS


```
nomad parameter_file.txt 

```
