
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import heamd
import numpy as np
import sys
import os

hm = heamd.HM()
hm.load_xml("project.xml")

a_arr = np.linspace(3.2, 3.7, 10)
a_arr = np.linspace(2, 4, 100)
m = 5
lattice = "bcc"

#print(a_arr)
#sys.exit(0)
U_arr = []


csv_filename = "U_vs_a_%s.csv" % (lattice,)

if not os.path.exists(csv_filename):

	for a in a_arr:

		result_filename = "results_%s_%d_%d.xml" % (lattice, int(a*1000), m)

		if not os.path.exists(result_filename):
			hm.set("result_filename", result_filename)
			hm.set("variables.L..value", a)
			hm.set("variables.m..value", m)
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

		U_arr.append(mean_U)

		print(a, mean_U)

	U_arr = np.array(U_arr)
	np.savetxt(csv_filename, [a_arr, U_arr])
else:
	x = np.loadtxt(csv_filename)
	a_arr = x[:,0]
	U_arr = x[:,1]
	np.savetxt("x", [a_arr, U_arr])


fig, ax = plt.subplots()
n = 4*m**3
U_arr /= n
ax.plot(a_arr, U_arr)

imin = np.argmin(U_arr)
print("a = ", a_arr[imin])
ax.plot([a_arr[imin]], [U_arr[imin]], marker='o', markersize=3, color="red")

ax.set_xlim([a_arr[0], a_arr[-1]])
ax.set(xlabel='a (Å)', ylabel='U (eV/atom)', title='')
ax.grid()

fig.savefig("U_vs_a_%s.png" % (lattice,), dpi=300)
plt.show()


