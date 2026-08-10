#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pathlib
this_dir = pathlib.Path(__file__).parent.resolve()

def main():
	### SimData, cuboid
	actions = np.arange(6)
	optimal = np.ones(6) * 3.
	eig_mean = 1000.0*np.array([0.02211848626, 0.01113804446, 0.007341087954, 0.004440149013, 0.003777719951, 0.003437718454])
	eig_ci95 = 1000.0*np.array([0.01206864927, 0.0048534782, 0.003112919725, 0.00155789221, 0.0006074147008, 0.0004107652626])
	random_mean = 1000.0 * np.array([0.0255883203, 0.01405082205, 0.01091017591, 0.009689241161, 0.006771326673, 0.005451947241])
	random_ci95 = 1000.0 * np.array([0.007122974013, 0.006770614906, 0.005518353172, 0.006258749049, 0.002377443961, 0.001375572562])
	# Corner Image
	with matplotlib.cbook.get_sample_data(os.path.join(this_dir, "sim_cube.png")) as file:
		sim_image = plt.imread(file, format='png')

	### SimData, polytope
	actions_poly = np.arange(9)
	eig_mean_poly = 1000.0*np.array([0.01434366309, 0.01136475317, 0.00724415925, 0.006414542538, 0.00685647615, 0.006147293602, 0.005716909037, 0.00558817285, 0.005707930354])
	eig_ci95_poly = 1000.0*np.array([0.002993411842, 0.004215687271, 0.002430216704, 0.001094652409, 0.001767235913, 0.001352082702, 0.001016471458, 0.001100403206, 0.00123767093])
	random_mean_poly = 1000.0 * np.array([0.01165882248, 0.01090799279, 0.01180387061, 0.01113218038, 0.009530420322, 0.007214829917, 0.006216667116, 0.005927627679, 0.005705503392])
	random_ci95_poly = 1000.0 * np.array([0.004349024313, 0.004963664643, 0.005733467932, 0.005055751477, 0.005087510401, 0.002069269684, 0.001304699034, 0.0005515629653, 0.0008513721328])
	random_outlier_poly = 1000.0 * np.array([0.01848405257, 0.02110649858, 0.0545742038, 0.0817396564, 0.01876991879, 0.008710569905, 0.02163157877, 0.01014240999, 0.01023394891])
	with matplotlib.cbook.get_sample_data(os.path.join(this_dir, "sim_poly.png")) as file:
		poly_image = plt.imread(file, format='png')

	### RealData
	#eig_mean_real = 1000.0*np.array([0.0290949488, 0.02069897397, 0.01487995172, 0.01383119741, 0.01266366022, 0.01218087022])
	#eig_ci95_real = 1000.0*np.array([0.001727716425, 0.007436204734, 0.004116229527, 0.003138741172, 0.00314116972, 0.0019007818])
	#random_mean_real = 1000.0 * np.array([0.02620941767, 0.01986856948, 0.01770433005, 0.01673022005, 0.01652878857, 0.01689979114])
	#random_ci95_real = 1000.0 * np.array([0.006526382667, 0.003574438573, 0.004508994351, 0.003542714023, 0.002937445078, 0.001754346216])

	### Plotting
	matplotlib.rcParams.update({'font.size': 30, 'font.family':'serif'})
	fig, ax = plt.subplots(nrows=2)
	ax[0].set_ylabel("← bCH (mm)")
	ax[0].plot(actions, random_mean, color='black', linewidth=3, label="Random")
	ax[0].fill_between(actions, random_mean-random_ci95, random_mean+random_ci95, linewidth=0., color=(0.0, 0.0, 0.0, 0.3), label="Random 95% CI")
	ax[0].plot(actions, eig_mean, color='red', linewidth=4, label="EIG (Ours)")
	ax[0].fill_between(actions, eig_mean-eig_ci95, eig_mean+eig_ci95, linewidth=0., color=(0.9, 0.0, 0.0, 0.4), label="EIG 95% CI")
	ax[0].set_xlim([0., 5.])
	ax[0].set_ylim([0., 20.])
	ax[0].set_title("Simulation, Cuboid and Convex Polytope")
	ax[0].grid(True, 'major', 'y', color="black")
	# Draw image
	axin = ax[0].inset_axes([0.7, 0.6, 0.4, 0.4])    # create new inset axes in data coordinates
	axin.imshow(sim_image)
	axin.axis('off')

	ax[1].set_ylabel("← bCH (mm)")
	ax[1].set_xlabel("Actions Since First Contact (1s ea)")
	ax[1].plot(actions_poly, random_mean_poly, color='black', linewidth=3, label="Random")
	ax[1].fill_between(actions_poly, random_mean_poly-random_ci95_poly, random_mean_poly+random_ci95_poly, linewidth=0., color=(0.0, 0.0, 0.0, 0.3), label="Random 95% CI")
	ax[1].plot(actions_poly, np.minimum(random_outlier_poly, np.ones(9)*22.), color='black', marker='x', markersize='20', linestyle='None')
	ax[1].plot(actions_poly, eig_mean_poly, color='red', linewidth=4, label="EIG (Ours)")
	ax[1].fill_between(actions_poly, eig_mean_poly-eig_ci95_poly, eig_mean_poly+eig_ci95_poly, linewidth=0., color=(0.9, 0.0, 0.0, 0.4), label="EIG 95% CI")
	ax[1].set_xlim([0., 8.])
	ax[1].set_ylim([0., 22.])
	ax[1].grid(True, 'major', 'y', color="black")
	ax[1].annotate("54mm", xytext=(1.5, 20.), xy=(2., 22.,),
            arrowprops=dict(arrowstyle="->"))
	ax[1].annotate("81mm", xytext=(2.5, 20.), xy=(3., 22.,),
            arrowprops=dict(arrowstyle="->"))
	# Draw image
	axin = ax[1].inset_axes([0.7, 0.6, 0.4, 0.4])    # create new inset axes in data coordinates
	axin.imshow(poly_image)
	axin.axis('off')
	ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)
	plt.show()

if __name__ == '__main__':
	main()
