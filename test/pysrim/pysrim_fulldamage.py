import os

import numpy as np
import time
from srim import Ion, Layer, Target, TRIM

start_time = time.time()

for j in ['He','C','F']: # Elements
	#for k in [1,3,6,10,12,14,16,18,20,22,24,26,28,30,35,40,45,50,55,60,100]: # k=(ion energies in keV)
	for k in [1,3]: # k=(ion energies in keV)
		t1 = time.time() #start time of new energy sample
		for i in range(2): # number of runs
		# Construct the ion
			ion = Ion(j, energy=k*1000) #energy=(energy in eV)
		# Adjust target width according to ion type and energy
			if j == 'H':
				wdth=157*(pow(k,0.57))
			if j == 'He':
				wdth=57*(pow(k,0.72))
			if j == 'C': 
				wdth=18.2*(pow(k,0.80))
			if j == 'F': 
				wdth=13.2*(pow(k,0.81))
			print ('The layer width is '+ str(wdth)+ ' microns.')
		# Construct the layer of He:CF4 gas
			layer = Layer({
				'He': {
 				'stoich': 3.0,
 				'E_d': 5.0,
 				'lattice': 1.0,
  				'surface': 2.0},
 				'C': {
 				'stoich': 2.0,
 				'E_d': 28.0,
 				'lattice': 3.0,
 				'surface': 7.41},
				'F': {
 				'stoich': 8.0,
 				'E_d': 25.0,
 				'lattice': 2.0,
 				'surface': 3.0}
 				}, density=0.001526, width=wdth*10000, phase=1)

 		# Construct a target of a single layer
			target = Target([layer])

		#Define the plot window x_min, x_max (in microns)
			x_min=0
			x_max=wdth*10000 #first window
#			while x_max <= wdth*10000: #remember the indentation if using this loop
				
			print ('Running event from ' + j + ' of ion ' + str(i) + ' for the energy of ' + str(k) + ' keV.')

		# Initialize a TRIM calculation with given target and ion for 1 ion, quick calculation (1), full cascade (2), monolayer (3)
			trim = TRIM(target, ion, number_ions=1, calculation=2, exyz=130, plot_xmin=x_min, plot_xmax=x_max, collisions=True, bragg_correction=0.9586, ranges=0)

		# Specify the directory of SRIM.exe
		# For windows users the path will include C://...
			srim_executable_directory = '/usr/local/srim/'
			outdirectory = '/tmp/pysrim_output/'

		# takes about 10 seconds on my laptop
			results = trim.run(srim_executable_directory)
		# If all went successfull you should have seen a TRIM window popup
		# results is `srim.output.Results` and contains all output files parsed
	                 
			from srim import TRIM

			os.makedirs(outdirectory, exist_ok=True)
			TRIM.copy_output_files(srim_executable_directory,outdirectory)
			#os.rename (outdirectory+'/LATERAL.txt', outdirectory+'/LATERAL' + j + 'Energy' + str(k) + 'keV'+ 'run' + str(i)+'.txt')
			#os.rename (outdirectory+'/Ioniz-3D.txt', outdirectory+'/Ioniz-3D_'+ j + '_' + str(k) + 'keV_'+ 'run' + str(i) + '.txt')
			#os.rename (outdirectory+'/BACKSCAT.txt', outdirectory+'/BACKSCAT'+ j + 'Energy' + str(k) + 'keV'+ 'run' + str(i)+'.txt')
			os.rename (outdirectory+'/IONIZ.txt', outdirectory+'/IONIZ_'+ j + '_' + str(k) + 'keV_'+ 'run' + str(i) + '.txt')
			#os.rename (outdirectory+'/E2RECOIL.txt', outdirectory+'/E2RECOIL'+ j + 'Energy' + str(k) + 'keV'+ 'run' + str(i)+'.txt')
			#os.rename (outdirectory+'/NOVAC.txt', outdirectory+'/NOVAC'+ j + 'Energy' + str(k) + 'keV'+ 'run' + str(i)+'.txt')
			#os.rename (outdirectory+'/PHONON.txt', outdirectory+'/PHONON_'+ j + '_' + str(k) + 'keV_'+ 'run' + str(i)+'.txt')
			#os.rename (outdirectory+'/RANGE.txt', outdirectory+'/RANGE_'+ j + '_' + str(k) + 'keV_'+ 'run' + str(i)+'.txt')
			os.rename (outdirectory+'/TDATA.txt', outdirectory+'/TDATA_'+ j + '_' + str(k) + 'keV_'+ 'run' + str(i) + '.txt')
			#os.rename (outdirectory+'/VACANCY.txt', outdirectory+'/VACANCY'+ j + 'Energy' + str(k) + 'keV'+ 'run' + str(i)+'.txt')
			#os.rename (outdirectory+'/RANGE_3D.txt', outdirectory+'/RANGE_3D_'+ j + '_' + str(k) + 'keV_'+ 'run' + str(i) +'.txt')
			os.rename (outdirectory+'/COLLISON.txt', outdirectory+'/COLLISON_'+ j + '_' + str(k) + 'keV_'+ 'run' + str(i) +'.txt')
			os.rename (outdirectory+'/EXYZ.txt', outdirectory+'/EXYZ_'+ j + '_' + str(k) + 'keV_'+ 'run' + str(i) +'.txt')

		# Increase plot window for next iteration
#				x_min = x_max
#				if (x_max + 100000) < wdth*10000: 
#					x_max = x_max + 100000 #window becomes the next 100 microns
#				elif x_max == wdth*10000: 
#					break #it should break the innermost loop
#				else: 
#					x_max = wdth*10000
		eltime = time.time() - t1
		print("Time elapsed for {} at {} keV : {} seconds".format(j,k,eltime))

print("Time elapsed in total : %s seconds" % (time.time() - start_time))
