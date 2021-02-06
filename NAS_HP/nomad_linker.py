import os
import sys

if len(sys.argv) != 2:
    print('Usage of nomad_linker.py: X.txt')
    exit()

fin = open(sys.argv[1], 'r')
Lin = fin.readlines()
Xin = Lin[0].split()
fin.close()

syst_cmd = 'python3 blackbox.py '

for i in range(len(Xin)):
    syst_cmd += str(Xin[i]) + ' '

syst_cmd += '> out.txt 2>&1'
os.system(syst_cmd)

fout = open('out.txt', 'r')
Lout = fout.readlines()
for line in Lout:

    if 'Mean best valid acc' in line:
        tmp = line.split()
        val_acc = '-'+str(tmp[-1])

    if "Std best valid acc" in line:
        tmp = line.split()
        std_val = str(tmp[-1])

    if "Mean best train acc" in line:
        tmp = line.split()
        train_acc = '-'+str(tmp[-1])

    if "Std best train acc" in line:
        tmp = line.split()
        std_train = str(tmp[-1])

        fout.close()

        print(val_acc, std_val, train_acc, std_train)
        exit()

print('Inf', 'Inf', 'Inf', 'Inf')
fout.close()
