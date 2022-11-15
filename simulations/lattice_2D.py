import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import meep as mp
import time

import funcs as fu

start = time.time()


P = {
    "meep_unit": 1e-3,
    "resolution": 5,
    "wire_shape": "round",
    "wire_size": 2.,
    "wire_spacing_x": 10.,
    "wire_spacing_y": 10.,
    "wire_number_x": 5,
    "wire_number_y": 5,
    "wire_offset_x": 0,
    "wire_offset_y": 0.,
    "offset_row": "even",
    "boundary_condition": "PEC",
    "solver": "TD_harminv",
    "until_after_sources": 100
    }

P["wire_material"] = mp.metal
P["walls_to_wires_x"] = 0.5*P["wire_spacing_x"]
P["walls_to_wires_y"] = 0.5*P["wire_spacing_y"]
P["source_frequency"] = fu.frequency_SI_to_MP(11.86e9, **P)
P["source_frequency_width"] = 1.9*((fu.TM_nm0_WM_2D(1, 2, **P)
                                    - fu.TM_nm0_WM_2D(**P)))
P["walls_width"] = 0.
P["walls_material"] = mp.air
P["absorber_width"] = 0.
P["absorber_layers"] = [mp.Absorber(P["absorber_width"])]

P["cell_size"] = fu.cell_size_2D_wire_grid(**P)
P["directory"] = "out"

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

P["plot_subcell"] = True
if P["get_steady_state_fields"] is True:
    P["plot_subcell"] = False
P["plot_sampling"] = True
P["plot_accentuated_wires"] = False

# -----------------------------------------------------------------------------
# launch simulation

freq_set, mode_set, fields_set, sim = fu.get_freqs_for_2D_grid(**P)

# -----------------------------------------------------------------------------

fu.save_output(freq_set, mode_set, fields_set, **P)

end = time.time()
elapsed_time = end - start
print("elapsed time =", (end-start)/60., "minutes")
