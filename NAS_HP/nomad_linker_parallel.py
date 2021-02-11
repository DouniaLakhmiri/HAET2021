import os
import sys
import multiprocessing
from joblib import Parallel, delayed
from statistics import mean, stdev

# Update the list with available gpu devices!
gpuIndexList = [1,2,3]

# The number of cifar10 subsets defined in blackbox_single
numberOfSubsets = 3

if len(sys.argv) != 2:
    print('Usage of nomad_linker.py: X.txt')
    exit()

fin = open(sys.argv[1], 'r')
Lin = fin.readlines()
Xin = Lin[0].split()
fin.close()

def run_single_blackbox(indexOfSubset):

    #
    # Create system command
    #
    syst_cmd = 'CUDA_VISIBLE_DEVICES=' +str(gpuIndexList[indexOfSubset]) + ' python3 blackbox_single.py '

    # Add the cifar10 subset index
    syst_cmd += str(indexOfSubset) + ' '
    
    for i in range(len(Xin)):
        syst_cmd += str(Xin[i]) + ' '

    outputFileName = 'out'+str(indexOfSubset)+'.txt'
    
    # Need to put the outputs in separate files
    syst_cmd += '> '+outputFileName+' 2>&1'
    
#    print(syst_cmd)
    
    # Launch the execution
     os.system(syst_cmd)

    # Read the output
    fout = open(outputFileName, 'r')
    Lout = fout.readlines()
    for line in Lout:

        if 'Best valid acc' in line:
            tmp = line.split()
            val_acc = '-'+str(tmp[-1])

        if 'Best train acc' in line:
            tmp = line.split()
            train_acc = '-'+str(tmp[-1])

        fout.close()

# To test uncomment this and comment the system command and read output
#    valid_acc= 10+indexOfSubset
#    train_acc= 10+indexOfSubset**2

    return valid_acc,train_acc

if __name__ == "__main__":

    if len(gpuIndexList)!=numberOfSubsets:
        print('Number of GPU ',len(gpuIndexList),' smaller than number of subsets',numberOfSubsets)
        exit()

    processed_list = Parallel(n_jobs=numberOfSubsets)(delayed(run_single_blackbox)(i) for i in range(numberOfSubsets))

    # Collect the outputs for validation and training, get mean and std dev for both and output
    best_train_acc = []
    best_valid_acc = []
    for element in processed_list:
        best_valid_acc.append(element[0])
        best_train_acc.append(element[1])

    print(mean(best_valid_acc),stdev(best_valid_acc), mean(best_train_acc), stdev(best_train_acc))
