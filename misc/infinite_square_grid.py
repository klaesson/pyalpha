"""
TM-band structure for infinite 2D grid.
Square lattice of metallic rods.
Settings as in: "Dispersion and Reflection Properties of Artificial
Media Formed By Regular Lattices of Ideally Conducting Wires" by
P.A. Belov , S.A. Tretyakov & A.J. Viitanen

See: https://doi.org/10.1163/156939302X00688
"""
import sys
sys.path.insert(0, "../simulations/")
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
from scipy import constants as con
import funcs as fu


def setup_simulation(**kwargs):

    geometry = mp.Cylinder(radius=0.5*kwargs["wire_size"],
                           material=kwargs["wire_material"])
    s_1 = mp.Source(mp.GaussianSource(frequency=kwargs["source_frequency"],
                                      fwidth=kwargs["source_frequency_width"]),
                    component=mp.Ez,
                    center=-8.4*kwargs["wire_size"]*mp.Vector3(-1.1, 1.306, 0),
                    size=mp.Vector3(0, 0, 0))

    sim = mp.Simulation(cell_size=kwargs["cell_size"], geometry=[geometry],
                        sources=[s_1],
                        resolution=kwargs["resolution"])

    return sim


P = {
    "meep_unit": 1,
    "resolution": 5,
    "wire_shape": "round",
    "wire_spacing_x": 1,
    "wire_spacing_y": 1,
    "until_after_sources": 300,
    }
P["wire_size"] = np.sqrt((0.001*(P["wire_spacing_x"]**2))/np.pi)
P["wire_material"] = mp.metal
P["source_frequency"] = 0.7
P["source_frequency_width"] = 1.2
P["cell_size"] = P["wire_spacing_x"]*mp.Vector3(1., 1., 0)
P["directory"] = "out"
P["filename"] = ("infinite_square_grid")

sim = setup_simulation(**P)

f = plt.figure(dpi=150)
sim.plot2D()
plt.legend()
plt.show()

# 1st Brillouin zone for 2D square lattice
# (should be in cartesian coordinates)
Gamma = mp.Vector3()
X_ = mp.Vector3(0.5/P["wire_spacing_x"])
M_ = 0.5/P["wire_spacing_x"]*mp.Vector3(1, 1)
k_points = [M_, Gamma, X_, M_]

# Get the general band structure
k_interp = 40
k_points_list = mp.interpolate(k_interp, k_points)
freqs = sim.run_k_points(P["until_after_sources"],
                         k_points_list)

plt.title("Dispersion stuff for TM bands")
for k in range(len(freqs)):
    for f in freqs[k]:
        plt.scatter(k, f*P["wire_spacing_x"], color="b", marker=".")

plt.scatter(0., (fu.frequency_SI_to_MP(fu.plasma_frequency_pavels(**P), **P)
            * P["wire_spacing_x"]), color="g", marker="o", label="plasma freq")
points_in_between = (len(freqs) - 4) / 3
tick_locs = [i*points_in_between+i for i in range(4)]
tick_labs = ['M', 'Γ', 'X', 'M']
plt.xticks(tick_locs, tick_labs)
plt.ylabel(r"$\frac{ka}{2\pi}$")
plt.ylim(0., 1.2)
plt.legend()
plt.savefig(P["directory"]+"/"+P["filename"]+"_res"+str(P["resolution"])
            + ".pdf")
plt.show()

freq_Gamma = [0, 0]
freq_Gamma_set = []
freq_M = [0, 0]
freq_M_set = []

for k in range(len(freqs)):
    for f in freqs[k]:
        if k_points_list[k] == Gamma:
            freq_Gamma_set.append((np.real(f), np.imag(f)))
            if freq_Gamma[0] == 0 or np.real(f) < freq_Gamma[0]:
                freq_Gamma = [np.real(f), np.imag(f)]

        if k_points_list[k] == M_:
            freq_M_set.append((np.real(f), np.imag(f)))
            if freq_M[0] == 0 or np.real(f) < freq_M[0]:
                freq_M = [np.real(f), np.imag(f)]

print("#######################")
print("Check for high error modes, freq_Gamma_set=", freq_Gamma_set)
print("#######################")
print("#######################")
print("Check for high error modes, freq_M_set=", freq_M_set)
print("#######################")
