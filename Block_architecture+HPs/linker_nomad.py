# ------------------------------------------------------------------------------
#  HyperNOMAD - Hyper-parameter optimization of deep neural networks with
#  NOMAD.
#
#
#
#  This program is free software: you can redistribute it and/or modify it
#  under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
#  for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#  You can find information on the NOMAD software at www.gerad.ca/nomad
# ------------------------------------------------------------------------------


import os
import sys

fin = open(sys.argv[1], 'r')
Lin = fin.readlines()
Xin = Lin[0].split()
fin.close()

syst_cmd = 'python3 ' + 'blackbox.py' + ' ' + sys.argv[1] + ' '

# syst_cmd = 'OMP_NUM_THREADS=3 python3 blackbox.py CIFAR10 '
out_file = 'logs_training_'

for i in range(len(Xin)):
    syst_cmd += str(Xin[i]) + ' '
    out_file += str(Xin[i]) + '_'

out_file += '.txt'
syst_cmd += '> ' + out_file
print(syst_cmd)
os.system(syst_cmd)

fout = open(out_file, 'r')
Lout = fout.readlines()
score = None

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
