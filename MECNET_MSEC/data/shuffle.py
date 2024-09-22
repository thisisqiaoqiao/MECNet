import os
import random
lines=[]
with open("/home/ubuntu/home/qgf/ECLNet-main/ECLNet_MSEC/data/train_1.txt", 'r') as infile:
     for line in infile:
         lines.append(line)
random.shuffle(lines)
out = open("/home/ubuntu/home/qgf/ECLNet-main/ECLNet_MSEC/data/train_1.txt",'w')
for line in lines:
    out.write(line)

infile.close()
out.close()

