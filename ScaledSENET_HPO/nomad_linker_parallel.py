import os
import sys
#import multiprocessing
from joblib import Parallel, delayed
from statistics import mean, stdev

# Update the list with available gpu devices!
gpuIndexList = [2,3,4]
# gpuIndexList = [2,3]

# The number of cifar10 subsets defined in blackbox_single
numberOfSeeds = 3

logAllFile = "logAllOutputs.txt"

if len(sys.argv) != 2:
    print('Usage of nomad_linker.py: X.txt')
    exit()

fin = open(sys.argv[1], 'r')
Lin = fin.readlines()
Xin = Lin[0].split()
fin.close()

def run_single_blackbox(indexOfSeed):

    val_acc = 0
    train_acc = 0
    #
    # Create system command
    #
    syst_cmd = 'CUDA_VISIBLE_DEVICES=' +str(gpuIndexList[indexOfSeed]) + ' python3 blackbox_single.py '

    # Add the cifar10 subset index
    syst_cmd += str(indexOfSeed) + ' '
    
    for i in range(len(Xin)):
        syst_cmd += str(Xin[i]) + ' '

    outputFileName = 'out'+str(indexOfSeed)+'.txt'
    
    # Need to put the outputs in separate files
    syst_cmd += '> '+outputFileName+' 2>&1'
    # print(syst_cmd)
    os.system(syst_cmd)

    # append file into log file
    append_cmd =   'echo ================================================================= >> ' + logAllFile + ' ; '
    append_cmd +=  'cat ' + outputFileName + ' >> ' + logAllFile
    
    # Launch the execution
    # print(append_cmd)
    os.system(append_cmd)

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

    return val_acc,train_acc


if __name__ == "__main__":

    if len(gpuIndexList)!=numberOfSeeds:
        print('Number of GPU ',len(gpuIndexList),' smaller than number of subsets',numberOfSeeds)
        exit()

    processed_list = Parallel(n_jobs=-1)(delayed(run_single_blackbox)(i) for i in range(numberOfSeeds))
#    best_valid_acc, best_train_acc = run_single_blackbox(0)


    # Collect the outputs for validation and training, get mean and std dev for both and output
    best_train_acc = []
    best_valid_acc = []
    for element in processed_list:
        best_valid_acc.append(float(element[0]))
        best_train_acc.append(float(element[1]))
        
    # print(best_valid_acc,best_train_acc)

    print(mean(best_valid_acc),stdev(best_valid_acc), mean(best_train_acc), stdev(best_train_acc))
