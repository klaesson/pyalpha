"""This is a script for tuning the TM_110 frequency by moving rows of wires.

In a row of wires, all wires have the same x-coordinates. The tuning is done by
moving every other row of wires in either the x or y-direction. You can choose
to move either odd or even-numbered rows by changing the value of the key
"offset_row" to either odd or even.

The resulting frequencies are stored in directory out. Other output specific
to the i:te geometry, e.g. such as field plots, is stored in outi.

When the rows are moved, the point where Harminv collects data is adjusted
automatically s.t. it always lies between two wires in an un-tuned row.

If P["use_previous_frequency_output"] is True, the source frequency of 
parameter n, will be used as source for parameter n+1, (for n > 0).

"""

import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import meep as mp
import time

import funcs as fu

mp.verbosity(1)

start = time.time()

P = {
    "meep_unit": 1e-3,
    "resolution": 5,
    "wire_shape": "round",
    "wire_size": 2.,
    "wire_number_x": 6,
    "wire_number_y": 6,
    "wire_spacing_x": 10.,
    "wire_spacing_y": 10.,
    "wire_offset_x": 0,
    "wire_offset_y": 0.,
    "boundary_condition": "PEC",
    "solver": "TD_harminv",
    "until_after_sources": 100,
    "harminv_max_bands": 10
    }

P["wire_material"] = mp.metal
P["walls_to_wires_x"] = 0.5*P["wire_spacing_x"]
P["walls_to_wires_y"] = 0.5*P["wire_spacing_y"]
P["source_frequency"] = fu.frequency_SI_to_MP(11.67e+9, **P)
P["source_frequency_width"] = (1.9*(fu.TM_nm0_WM_2D(1, 2, **P)
                               - fu.TM_nm0_WM_2D(**P)))

P["walls_width"] = 0
P["walls_material"] = mp.air
P["absorber_width"] = 0
P["absorber_layers"] = [mp.Absorber(P["absorber_width"])]

P["cell_size"] = fu.cell_size_2D_wire_grid(**P)
P["directory"] = "out"
P["filename"] = "lattice_2D_tuning_wires"

# -----------------------------------------------------------------------------
# Control tuning:

# choose which set of wires to move and along which axis
P["offset_row"] = "even"  # or "odd"
P["offset_axis"] = "x"  # or "y"

# Set the parameter name to offset and choose in how many increments to tune
P["parameter_name"] = "offset"
P["offset_steps"] = 5

# Get the parameter list to loop over
P = fu.get_offset_set_tuning_wires(**P)

# With below set to True, the source frequency of parameter n, will be used as
# source for parameter n+1, (for n > 0).
P["use_previous_frequency_output"] = True

# -----------------------------------------------------------------------------
# controls defects

P["get_defects"] = False
P["defects"] = fu.get_defects(**P)

# -----------------------------------------------------------------------------
# controls simulation output

P["get_fields"] = True
P["get_max_field_slice"] = True
P["get_field_slices"] = False
P["get_phase_plot"] = True
P["get_error_estimates"] = True

P["get_steady_state_fields"] = False
P["steady_state_fields_sample_periods_min"] = 2
P["steady_state_fields_sample_periods_max"] = 10
P["steady_state_fields_overlap_stop_condition"] = 1e-4

P["plot_subcell"] = False
if P["get_steady_state_fields"] is True:
    P["plot_subcell"] = False
P["plot_sampling"] = True
P["plot_accentuated_wires"] = False

# -----------------------------------------------------------------------------
# Plot the geometry before running sims.

for i in range(len(P["parameter_set"])):
    if P["offset_axis"] == "x":
        P["wire_offset_x"] = P["parameter_set"][i]
    else:
        P["wire_offset_y"] = P["parameter_set"][i]
    P["directory"] = "out"+str(i)
    fu.get_plot_geometry_and_harminv_sampling(**P)
P["wire_offset_x"], P["wire_offset_y"] = 0, 0
P["directory"] = "out"

# -----------------------------------------------------------------------------
# launch simulation
freq_set, mode_set, fields_set = fu.get_freqs_2D_grid_delta_parameter(**P)

# -----------------------------------------------------------------------------

fu.save_output(freq_set, mode_set, fields_set, **P)
fu.plot_mode_set(mode_set, **P)
fu.plot_fields_set(fields_set, **P)

end = time.time()
print("elapsed time =", (end-start)/60., "minutes")
