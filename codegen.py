# file: codegen.py
# author: 
# date: 2017.10.19
import numpy as np
from os import system
from sys import argv
from subprocess import Popen, call, PIPE

if __name__ == "__main__":

	TX = 16
	TY = 16

	opt_TX = 0
	opt_TY = 0
	opt_BM = 0
	opt_BK = 0
	opt_BN = 0
	opt_time = 1e10

	result = open('config.csv', 'w')
	result.write("TX, TY, BM, BK, BN, time(ms)\n")
	result.flush()	

	M = int(argv[1])
	K = int(argv[2])
	N = int(argv[3])

	AX = 32
	BX = 8
	AY = TX * TY / AX
	BY = TX * TY / BX
	for M_factor in range(2, 5):
		for K_factor in range(1, 5):
			for N_factor in range(2, 5):
					
				BM = max(AX, TX) * M_factor
				BK = max(AY, BX) * K_factor
				BN = max(TY, BY) * N_factor
				
				smem_size = (BK * BN + BM * BK) * 4 / 1024.0
				if smem_size > 48:
					continue

				cmd = '''sed "s/THREAD_BLOCK_X/%d/g;
					s/THREAD_BLOCK_Y/%d/g;
					s/ROW_BLOCK_A/%d/g;
					s/ROW_BLOCK_B/%d/g;
					s/COL_BLOCK_C/%d/g;
					s/DIM_XA/%d/g;
					s/DIM_XB/%d/g" mysgemm_template.cu > mysgemm.cu'''%(TX, TY, BM, BK, BN, AX, BX)
			
				print(cmd)
				call(cmd, shell = True)
				Popen("make", shell = True, stdout = PIPE, stderr = PIPE).wait()
				
				ret = Popen("./mysgemm %d %d %d"%(M, K, N), shell = True, stdout = PIPE, bufsize = 1).stdout.readlines()
				timeinfo = ret[0]
				validinfo = ret[1]
				timeinfo = str(timeinfo)
				validinfo = str(validinfo)	
				time = float(timeinfo[timeinfo.find("=") + 2:timeinfo.find("ms") - 1])
				valid = validinfo[validinfo.find("=") + 2:validinfo.find("\n") - 2]
				print(valid)
				if valid == "PASS":
					result.write("%d,%d,%d,%d,%d,%.2f\n"%(TX, TY, BM, BK, BN, time))
					result.flush()
					if time < opt_time:
						opt_TX = TX
						opt_TY = TY
						opt_BM = BM
						opt_BK = BK
						opt_BN = BN
						opt_time = time
		
	 
	cmd = '''sed "s/THREAD_BLOCK_X/%d/g;
		s/THREAD_BLOCK_Y/%d/g;
		s/ROW_BLOCK_A/%d/g;
		s/ROW_BLOCK_B/%d/g;
		s/COL_BLOCK_C/%d/g;
		s://#define VERBOSITY:#define VERBOSITY:g;
		s/DIM_XA/%d/g;
		s/DIM_XB/%d/g" mysgemm_template.cu > mysgemm.cu'''%(opt_TX, opt_TY, opt_BM, opt_BK, opt_BN, AX, BX)
	
	call(cmd, shell = True)
	Popen("make", shell = True, stdout = PIPE, stderr = PIPE).wait()

