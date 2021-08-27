
"""

Error model:
assume for the kinetic energy
E(t) = E0 + E1*sin(w*t)
E_bar(t) = 1/t * int_0^t E(tau) dtau = 1/t * (E0*t + E1/w*(1 - cos(w*t))) = E0 + E1/(w*t)*(1 - cos(w*t))

Relative error:
eps(t) = (E_bar(t) - E0)/E0 = E1/(E0*w*t)*(1 - cos(w*t))
eps_bar(t) = E1/(E0*w*t)

Averaging time based on oscillating frequency and error tolerance:
t_avg = E1/(E0*2*pi*f*tol)

Required tolerance to identify thermal expansion coeff.:

alpha = 1/a*da/dT

da/a = alpha*dT

example:
alpha \approx 10^-5 / K
dT = 100 K
da/a = 10^-3

=> minimum of U has to be identified with rel. accuracy of 10^-3

assume quadratic model with noise

U(a) = U0 + U1*(a - a0)^2 + N

minimum is at U' = 0

2*U1*(a - a0) = |N'|

so the error is roughly
(a - a0)/a = |N'|/(U''*a) = E1/(w*t*U''*a)

assumin
|N'| \approx N/a

so the integration time is roughly

t = E1/(alpha*dT*f*U''*a^2) 

t_avg = E1/(E0*2*pi*f*tol)


E1: oscillation amplitude of Ekin or Epot
f: oscillation frequency of Ekin or Epot
dT: temperature range for computing thermal expansion coeff. (by finite differences)
a: nominal lattice constant
alpha: expected thermal expansion coefficient
U'': second derivative of Epot w.r.t. a at minimum


E1 = 1eV
U'' = 4eV/A^2
alpha = 1e-5/K
dT = 100K
f = 20/ps
a = 3 A

t = 1ps

"""

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import heamd
import numpy as np
import sys
import os

hm = heamd.HM()
hm.load_xml("project.xml")

for T in [300, 600]:

	a_arr = np.linspace(3.2, 3.7, 10)
	a_arr = np.linspace(2.980, 2.991, 10)
	m = 5
	lattice = "bcc"
	v = 10
	rtol = 0.01

	csv_filename = "U_vs_a_%s_%d_v%d.csv" % (lattice, T, v)
	print("CSV:", csv_filename)

	if not os.path.exists(csv_filename):

		U_arr = []
		T_arr = []

		for a in a_arr:

			result_filename = "results_%s_%d_%d_%d_v%d.xml" % (lattice, int(a*1000), m, T, v)
			print("result:", result_filename)

			if not os.path.exists(result_filename):
				hm.set("result_filename", result_filename)
				hm.set("variables.L..value", a)
				hm.set("variables.m..value", m)
				hm.set("dt", 0.001*300/T)
				hm.set("steps.step[0]..T", T)
				hm.set("steps.step[0]..t", -2*300/T)
				hm.set("steps.step[1]..t", -1*300/T)
				# t_avg = E1/(E0*2*pi*f*tol)
				t_avg = 40*300/T
				t_avg = 1*300/T
				hm.set("steps.step[3]..t", t_avg)
				hm.set("initial_position", lattice)
				#print(hm.get_xml())
				#sys.exit(0)
				hm.run()


			r = ET.parse(result_filename).getroot()
			timesteps = r.find("timesteps")

			mean_U = 0
			mean_T = 0
			sum_dt = 0
			for ts in timesteps:
				t = float(ts.attrib["t"])
				if t < 0:
					continue

				stats = ts.find("stats")
				dt = float(stats.find("dt").text)
				mean_U += float(stats.find("Epot").text)*dt
				mean_T += float(stats.find("T").text)*dt
				sum_dt += dt

			mean_U /= sum_dt
			mean_T /= sum_dt

			U_arr.append(mean_U)
			T_arr.append(mean_T)

			print(a, mean_U)

		U_arr = np.array(U_arr)
		T_arr = np.array(T_arr)
		np.savetxt(csv_filename, [a_arr, U_arr, T_arr])
	else:
		x = np.loadtxt(csv_filename)
		a_arr = x[0,:]
		U_arr = x[1,:]
		T_arr = x[2,:]



	fig, ax = plt.subplots()
	n = 4*m**3
	U_arr /= n
	ax.plot(a_arr, U_arr, marker='x', markersize=3)

	x = np.linspace(a_arr[0], a_arr[-1], 400)
	pc = np.polyfit(a_arr, U_arr, 3)
	p = np.poly1d(pc)
	print(p)
	ax.plot(x, p(x))


	imin = np.argmin(U_arr)
	print("a = ", a_arr[imin])
	print("a_fit = ", p.deriv().r)
	ax.plot([a_arr[imin]], [U_arr[imin]], marker='o', markersize=3, color="red")



	ax.set_xlim([a_arr[0], a_arr[-1]])
	ax.set(xlabel='a (Å)', ylabel='U (eV/atom)', title='')
	ax.grid()


	fig, ax = plt.subplots()
	dU = np.diff(U_arr)/np.diff(a_arr)
	ax.plot(a_arr[0:-1], dU)

	ddU = (dU[-1] - dU[0])/(a_arr[-1] - a_arr[0])
	print("U'' =", ddU)


	fig.savefig("U_vs_a_%s_%d_v%d.png" % (lattice, T, v), dpi=300)


plt.show()

