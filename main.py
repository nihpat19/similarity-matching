import subprocess
import argparse

parser = argparse.ArgumentParser(description="Similarity Matching")
parser.add_argument('--net','-n','Which neural network type to use',metavar='net')
parser.add_argument('--testOnly','-t',action='store_true', help='Whether or not to test the network (Leave blank to train)',metavar='test')
parser.add_argument('--lr', default=0.1, type=float,help='learning rate for optimizer', metavar='lr')

args = parser.parse_args()

assert(args.net=="student" or args.net=="robust")
if args.net=="student":
	if args.test:
		subprocess.call(['python3','main_sim_final.py','-t'])
	else:
		reg = "start_string"
		while not reg in ['y','n']:
			reg = str(input("Regularize Training? (Y/N): ")).lower().strip()
		if reg=='y':
			lmb = 1
			while not lmb > 1:
				lmb = float(input("Enter Lambda Regularization Constraint (Must be greater than 1): ").strip())
			if lmb > 1:
				eta = 0
				while not 0<eta<1:
					eta = float(input("Enter Eta Regularization Constant (Must be greater than 0 and less than 1): ").strip())
				if 0<eta<1:
					if args.lr != 0.1:
						subprocess.call(['python3','main_sim_final.py','--sim'])
					else:
						subprocess.call(['python3','main_sim_final.py','--sim'])
		else:
			
			
	


