#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pathlib
this_dir = pathlib.Path(__file__).parent.resolve()

def main():

	actions = np.arange(6)
	### RealData
	eig_mean_real = 1000.0*np.array([0.0290949488, 0.02069897397, 0.01487995172, 0.01383119741, 0.01266366022, 0.01218087022])
	eig_ci95_real = 1000.0*np.array([0.001727716425, 0.007436204734, 0.004116229527, 0.003138741172, 0.00314116972, 0.0019007818])
	random_mean_real = 1000.0 * np.array([0.02620941767, 0.01986856948, 0.01770433005, 0.01673022005, 0.01652878857, 0.01689979114])
	random_ci95_real = 1000.0 * np.array([0.006526382667, 0.003574438573, 0.004508994351, 0.003542714023, 0.002937445078, 0.001754346216])
	# Corner Image
	with matplotlib.cbook.get_sample_data(os.path.join(this_dir, "real_cube.png")) as file:
		real_image = plt.imread(file, format='png')
	with matplotlib.cbook.get_sample_data(os.path.join(this_dir, "real_estimate.png")) as file:
		real_estimate = plt.imread(file, format='png')

	### Plotting
	matplotlib.rcParams.update({'font.size': 36, 'font.family':'serif'})
	fig, ax = plt.subplots(nrows=1)
	ax.set_ylabel("← bCH (mm)")
	ax.set_xlabel("Actions Since First Contact (1.5s ea)")
	ax.plot(actions, random_mean_real, color='black', linewidth=3, label="Random")
	ax.fill_between(actions, random_mean_real-random_ci95_real, random_mean_real+random_ci95_real, linewidth=0., color=(0.0, 0.0, 0.0, 0.3), label="Random 95% CI")
	ax.plot(actions, eig_mean_real, color='red', linewidth=4, label="EIG (Ours)")
	ax.fill_between(actions, eig_mean_real-eig_ci95_real, eig_mean_real+eig_ci95_real, linewidth=0., color=(0.9, 0.0, 0.0, 0.4), label="EIG 95% CI")
	ax.set_xlim([0., 5.])
	ax.set_ylim([5., 30.])
	ax.set_title("Real Robot, Cuboid Parameterization")
	ax.grid(True, 'major', 'y', color="black")
	ax.legend(loc='lower left', bbox_to_anchor=(0., 0.),
          fancybox=True, shadow=True, ncol=1)
	# Draw image
	axin = ax.inset_axes([0.75, 0.7, 0.3, 0.3])    # create new inset axes in data coordinates
	axin.imshow(real_image)
	axin.axis('off')
	axin2 = ax.inset_axes([0.55, 0.7, 0.3, 0.3])    # create new inset axes in data coordinates
	axin2.imshow(real_estimate)
	axin2.axis('off')

	plt.show()

if __name__ == '__main__':
	main()
