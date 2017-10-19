# file: codegen.py
# author: Zhang Xiao
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
	print >> result, "TX, TY, BM, BK, BN, time(ms)"
	result.flush()	

	M = int(argv[1])
	K = int(argv[2])
	N = int(argv[3])

	for M_factor in xrange(2, 15):
		for K_factor in xrange(2, 15):
			for N_factor in xrange(2, 15):
					
				BM = TY * M_factor
				BK = TY * K_factor
				BN = TX * N_factor
				
				smem_size = BK * BN * 4 / 1024.0
				if smem_size > 48:
					continue

				cmd = '''sed "s/THREAD_BLOCK_X/%d/g;
					s/THREAD_BLOCK_Y/%d/;
					s/ROW_BLOCK_A/%d/;
					s/ROW_BLOCK_B/%d/;
					s/COL_BLOCK_C/%d/" mygemm_template.cu > mygemm.cu'''%(TX, TY, BM, BK, BN)
			
				print cmd
				call(cmd, shell = True)
				Popen("make", shell = True, stdout = PIPE, stderr = PIPE).wait()
				
				ret = Popen("./mysgemm %d %d %d"%(M, K, N), shell = True, stdout = PIPE, bufsize = 1).stdout.readlines()
				timeinfo = ret[0]
				validinfo = ret[1]
				time = float(timeinfo[timeinfo.find("=") + 2:timeinfo.find("ms") - 1])
				valid = validinfo[validinfo.find("=") + 2:-1]
				print valid
				if valid == "PASS":
					print >> result, "%d,%d,%d,%d,%d,%.2f"%(TX, TY, BM, BK, BN, time)
					result.flush()
					if time < opt_time:
						opt_TX = TX
						opt_TY = TY
						opt_BM = BM
						opt_BK = BK
						opt_BN = BN
		
	 
	
