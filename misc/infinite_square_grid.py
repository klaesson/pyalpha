"""
TM-band structure for infinite 2D grid.
Square lattice of metallic rods.
Settings as in: "Dispersion and Reflection Properties of Artificial
Media Formed By Regular Lattices of Ideally Conducting Wires" by
P.A. Belov , S.A. Tretyakov & A.J. Viitanen

See: https://doi.org/10.1163/156939302X00688
"""

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
from scipy import constants as con


def setup_simulation(**kwargs):

    geometry = mp.Cylinder(radius=0.5*kwargs["wire_size"],
                           material=kwargs["wire_material"])

    # Seems to be good to have atleast two sources?
    s_1 = mp.Source(mp.GaussianSource(frequency=kwargs["source_frequency"],
                                      width=kwargs["source_frequency_width"]),
                    component=mp.Ez,
                    center=3.5*kwargs["wire_size"]*mp.Vector3(1.05, 1.27, 0),
                    size=mp.Vector3(0, 0, 0))
    s_2 = mp.Source(mp.GaussianSource(frequency=kwargs["source_frequency"],
                                      fwidth=kwargs["source_frequency_width"]),
                    component=mp.Ez,
                    center=-8.4*kwargs["wire_size"]*mp.Vector3(-1.1, 1.306, 0),
                    size=mp.Vector3(0, 0, 0))

    sim = mp.Simulation(cell_size=kwargs["cell_size"], geometry=[geometry],
                        sources=[s_1, s_2],
                        resolution=kwargs["resolution"])

    return sim


def get_field_patterns_for_k_point(k_point, name, **kwargs):

    sim = setup_simulation(**kwargs)
    sim.k_point = k_point
    h_ = mp.Harminv(mp.Ez, kwargs["harminv_sampling_mp4"],
                    fcen=kwargs["source_frequency"],
                    df=kwargs["source_frequency_width"])
    sim.run(mp.at_beginning(mp.output_epsilon),
            mp.after_sources(h_),
            until_after_sources=kwargs["until_after_sources"])
    # sim.run(mp.at_every(1/kwargs["source_frequency"]/20, mp.output_efield_z),
            # until=1/kwargs["source_frequency"])
    animate = mp.Animate2D(sim, fields=mp.Ez, f=plt.figure(dpi=150),
                           realtime=False, normalize=False)
    frequency = h_.modes[0].freq
    sim.run(mp.at_every(1/frequency/20, animate), until=1/frequency)
    plt.close()
    fps = 5
    animate.to_mp4(fps, kwargs["directory"]+"/"+kwargs["filename"]
                   + "_at_"+name+"_point.mp4")

    return sim, h_.modes


def plasma_frequency_pavels(**kwargs):
    """Input Meep units, output in SI units. From Belov et al"""

    a_ = kwargs["wire_spacing"]*kwargs["meep_unit"]

    if "wire_size" in kwargs:
        r_ = 0.5*kwargs["wire_size"]*kwargs["meep_unit"]

    else:
        r_ = kwargs["wire_radius"]*kwargs["meep_unit"]

    k_0 = np.sqrt((2*np.pi/(a_**2))/(np.log(a_/(2*np.pi*r_))+0.5275))

    return (k_0*con.c)/(2*np.pi)


def frequency_SI_to_MP(x, **kwargs):

    return (x*kwargs["meep_unit"])/con.c


def frequency_MP_to_SI(x, **kwargs):

    return (x*con.c)/kwargs["meep_unit"]


P = {
    "meep_unit": 1,
    "resolution": 5,
    "wire_shape": "round",
    "wire_spacing": 1,
    "until_after_sources": 300,
    }
P["wire_size"] = np.sqrt((0.001*(P["wire_spacing"]**2))/np.pi)
P["wire_material"] = mp.metal
P["source_frequency"] = 0.7
P["source_frequency_width"] = 1.2
P["harminv_sampling_mp4"] = 4.5*P["wire_size"]*mp.Vector3(1.321, -1.07, 0)
P["cell_size"] = P["wire_spacing"]*mp.Vector3(1., 1., 0)
P["directory"] = "out"
P["filename"] = ("infinite_square_grid")

sim = setup_simulation(**P)

f = plt.figure(dpi=150)
sim.plot2D()
plt.scatter(P["harminv_sampling_mp4"][0], P["harminv_sampling_mp4"][1],
            color="b", marker="o", label="Harminv sampel point for animate")
plt.legend()
plt.show()

# 1st Brillouin zone for 2D square lattice
# (should be in cartesian coordinates)
Gamma = mp.Vector3()
X_ = mp.Vector3(0.5/P["wire_spacing"])
M_ = mp.Vector3(0.5/P["wire_spacing"], 0.5/P["wire_spacing"])
k_points = [M_, Gamma, X_, M_]

# Get the general band structure
k_interp = 40
k_points_list = mp.interpolate(k_interp, k_points)
freqs = sim.run_k_points(P["until_after_sources"],
                         k_points_list)

plt.title("Dispersion stuff for TM bands")
for k in range(len(freqs)):
    for f in freqs[k]:
        plt.scatter(k, f*P["wire_spacing"], color="b", marker=".")

plt.scatter(0., (frequency_SI_to_MP(plasma_frequency_pavels(**P), **P)
            * P["wire_spacing"]), color="g", marker="o",
            label="plasma freq, Pavels")
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

sim.reset_meep()

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

P_Gamma = P.copy()
P_Gamma["source_frequency"] = freq_Gamma[0]
P_Gamma["source_frequency_width"] = 1.9*(1.41-freq_Gamma[0])
print("For Gamma with Re(f)=", freq_Gamma[0],
      ". Im(f)=", freq_Gamma[1])
sim, mode = get_field_patterns_for_k_point(Gamma, "Gamma", **P_Gamma)
mode_Gamma = mode
print("#######################")
print("Mode Gamma", mode_Gamma)
print("#######################")

sim.reset_meep()

P_M = P.copy()
P_M["source_frequency"] = freq_M[0]
P_M["source_frequency_width"] = 1.9*(0.918-freq_M[0])
print("For M with Re(f)=", freq_M[0],
      ". Im(f)=", freq_M[1])
sim, mode = get_field_patterns_for_k_point(M_, "M", **P_M)
mode_M = mode
print("#######################")
print("Mode M", mode_M)
print("#######################")
sim.reset_meep()



