##################################################
# exec on linux command: python3 bin2ten.py 1101
##################################################

import os
import sys

B = sys.argv[1]
I = 0
while B:
	I = I * 2 + (ord(B[0]) - ord('0'))
	print(I)
	B = B[1:]
	print(B)

print("The result= " + str(I) + "\n")

