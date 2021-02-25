import os
import sys

if len(sys.argv) != 3:
    print ('Usage of pytorch_bb.py: DATABASE_NAME X.txt')
    exit()

if 'HYPERNOMAD_HOME' not in os.environ:
    print('The environment variable $HYPERNOMAD_HOME is not set')
    exit()

# print(os.environ.get('HYPERNOMAD_HOME'))

fin = open(sys.argv[2], 'r')
Lin = fin.readlines()
Xin = Lin[0].split()
fin.close()

syst_cmd = ' python3 bb_hps_scaled_senet.py ' + sys.argv[1] + ' '


for i in range(len(Xin)):
    syst_cmd += str(Xin[i]) + ' '

syst_cmd += '> out.txt 2>&1'

print(syst_cmd)

os.system(syst_cmd)

fout = open('out.txt', 'r')
Lout = fout.readlines()
for line in Lout:
    if "Best valid acc" in line:
        tmp = line.split()
        print('-' + str(tmp[3]))
        fout.close()
        exit()

print('Inf')
fout.close()
