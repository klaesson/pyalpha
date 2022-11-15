import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
import itertools as it
from scipy import constants as con
import time


mp.verbosity(1)


# -----------------------------------------------------------------------------
# materials


def get_simple_metall(epsilon_r=1.00001, E_conductivity=5.8e+7, **kwargs):
    """Sets resistive material."""
    medium = mp.Medium(epsilon=epsilon_r, D_conductivity=((kwargs["meep_unit"]
                       * E_conductivity)/(con.c*epsilon_r*con.epsilon_0)))

    return medium


# -----------------------------------------------------------------------------
# defects


def get_defects(**kwargs):

    defects = np.zeros(shape=(kwargs["wire_number_x"],
                              kwargs["wire_number_y"], 2))

    if kwargs["get_defects"] is True:

        if kwargs["defects_type"] == "single_wire":
            i = kwargs["defects_position"][0]
            j = kwargs["defects_position"][1]

            if kwargs["defects_axis"] == "x":
                defects[i][j][0] = kwargs["defects_size"]

            elif kwargs["defects_axis"] == "y":
                defects[i][j][1] = kwargs["defects_size"]

            elif kwargs["defects_axis"] == "xy":
                defects[i][j][0] = kwargs["defects_size"]
                defects[i][j][1] = kwargs["defects_size"]

        elif kwargs["defects_type"] == "single_plane":
            index = kwargs["defects_position"]

            if kwargs["defects_axis"] == "x":
                defects[:, index][:, 0] = kwargs["defects_size"]

            elif kwargs["defects_axis"] == "y":
                defects[:, index][:, 1] = kwargs["defects_size"]

            elif kwargs["defects_axis"] == "xy":
                defects[:, index][:, 0] = kwargs["defects_size"]
                defects[:, index][:, 1] = kwargs["defects_size"]

        elif kwargs["defects_type"] == "all_random":
            defects = np.load("/cfs/home/tokl6780/defects/"
                              + str(kwargs["wire_number_x"])
                              + "by"+str(kwargs["wire_number_y"])
                              + "/defects_"+str(kwargs["defects_number"])
                              + ".npy")

        elif kwargs["defects_type"] == "misaligned_planes":
            defects = get_plane_misalignment_defects(**kwargs)

        elif kwargs["defects_type"] == "all_random_and_misaligned_planes":
            defects = (np.load("/cfs/home/tokl6780/defects/"
                               + str(kwargs["wire_number_x"])
                               + "by"+str(kwargs["wire_number_x"])
                               + "/defects_0.npy")
                       + get_plane_misalignment_defects(**kwargs))
        else:
            raise ValueError("Unsupported type of defect")

    return defects


def get_plane_misalignment_defects(**kwargs):
    """ Rotates a set of even or odd wires around the center of the grid."""

    kwargs["defects"] = np.zeros(shape=(kwargs["wire_number_x"],
                                 kwargs["wire_number_y"], 2))

    if check_if_offset_geometry(**kwargs):
        w_grid = geometry_2D_wire_grid_offset([], **kwargs)
    else:
        w_grid = geometry_2D_wire_grid([], **kwargs)

    w_grid = np.asarray(w_grid).reshape(kwargs["wire_number_x"],
                                        kwargs["wire_number_y"])

    for i in range(kwargs["wire_number_x"]):
        for j in range(kwargs["wire_number_y"]):
            if ((kwargs["misaligned_planes"] == "even" and i % 2 == 0) or
               (kwargs["misaligned_planes"] == "odd" and i % 2 != 0)):
                x_tilt, y_tilt = get_clockwise_rotation(w_grid[i][j].center[0],
                                                        w_grid[i][j].center[1],
                                                        kwargs["defects_size"])
                kwargs["defects"][i][j][0] = x_tilt
                kwargs["defects"][i][j][1] = y_tilt
            else:
                kwargs["defects"][i][j][0] = 0
                kwargs["defects"][i][j][1] = 0

    return kwargs["defects"]


def get_clockwise_rotation(x_, y_, theta):
    """ Theta in radians. """

    return (x_*np.cos(theta)-y_*np.sin(theta)-x_,
            x_*np.sin(theta)+y_*np.cos(theta)-y_)


def get_defects_positions_type_A(x_n=[["-", "-"], [0, 0]], **kwargs):
    """ Used for single type defects in a n x n grid.

    Returns a list with an element representing the un-altered grid and a
    triangle of positions in the lower left quadrant. I.e. for n=3 it returns
    ["-", "-"] and the following wire indices marked by x:
    . . .
    . x .
    x x .
    """

    if x_n[-1][1] == np.ceil(0.5*kwargs["wire_number_x"])-1:
        return x_n
    else:
        x_m = x_n+[[i, x_n[-1][1]+1] for i in range(x_n[-1][0]+2)]
        return get_defects_positions_type_A(x_m, **kwargs)


def get_defects_size_set(**kwargs):

    return list(np.linspace(0, kwargs["defects_size"],
                            kwargs["defects_steps"]))


# -----------------------------------------------------------------------------
# setup of the geometry

def cell_size_2D_wire_grid(**kwargs):

    sx = (kwargs["wire_spacing_x"]*(kwargs["wire_number_x"]-1)
          + 2*(kwargs["walls_to_wires_x"]+kwargs["walls_width"]
               + kwargs["absorber_width"]))
    sy = (kwargs["wire_spacing_y"]*(kwargs["wire_number_y"]-1)
          + 2*(kwargs["walls_to_wires_y"]+kwargs["walls_width"]
               + kwargs["absorber_width"]))

    return mp.Vector3(sx, sy, 0.)


def cell_size_2D_empty_cavity(cavity_size, **kwargs):

    sx = cavity_size+2*(kwargs["walls_width"]+kwargs["absorber_width"])

    return mp.Vector3(sx, sx, 0.)


def geometry_2D_walls(geometry, **kwargs):

    block_size_x = kwargs["cell_size"][0]-2*kwargs["absorber_width"]
    cavity_size_x = block_size_x-2*kwargs["walls_width"]
    block_size_y = kwargs["cell_size"][1]-2*kwargs["absorber_width"]
    cavity_size_y = block_size_y-2*kwargs["walls_width"]

    geometry.append(mp.Block(mp.Vector3(block_size_x, block_size_y, 0),
                             center=mp.Vector3(),
                             material=kwargs["walls_material"]))
    geometry.append(mp.Block(mp.Vector3(cavity_size_x,
                                        cavity_size_y,
                                        0.),
                             center=mp.Vector3(), material=mp.air))

    return geometry


def geometry_2D_wire_grid(geometry, **kwargs):

    def place_x(integer, **kwargs):

        cell_size = kwargs["cell_size"][0]
        position = (-cell_size*0.5 + kwargs["walls_to_wires_x"]
                    + kwargs["walls_width"]+kwargs["absorber_width"]
                    + integer*kwargs["wire_spacing_x"])
        return position

    def place_y(integer, **kwargs):

        cell_size = kwargs["cell_size"][1]
        position = (-cell_size*0.5 + kwargs["walls_to_wires_y"]
                    + kwargs["walls_width"]+kwargs["absorber_width"]
                    + integer*kwargs["wire_spacing_y"])
        return position

    for i in range(kwargs["wire_number_x"]):
        for j in range(kwargs["wire_number_y"]):
            defect_x = kwargs["defects"][i][j][0]
            defect_y = kwargs["defects"][i][j][1]
            x_ = place_x(i, **kwargs)
            y_ = place_y(j, **kwargs)

            center = mp.Vector3(x_+defect_x, y_+defect_y, 0.)

            if kwargs["wire_shape"] == "square":
                size = mp.Vector3(x=kwargs["wire_size"], y=kwargs["wire_size"])
                geometry.append(mp.Block(material=kwargs["wire_material"],
                                         size=size,
                                         center=center))
            else:
                geometry.append(mp.Cylinder(material=kwargs["wire_material"],
                                            radius=0.5*kwargs["wire_size"],
                                            center=center))

    return geometry


def geometry_2D_wire_grid_offset(geometry, **kwargs):

    def place(integer, column, axis, **kwargs):

        if axis == "x_axis":
            offset = kwargs["wire_offset_x"]
            cell_size = kwargs["cell_size"][0]
            walls_to_wires = kwargs["walls_to_wires_x"]
            wire_spacing = kwargs["wire_spacing_x"]

        else:
            offset = kwargs["wire_offset_y"]
            cell_size = kwargs["cell_size"][1]
            walls_to_wires = kwargs["walls_to_wires_y"]
            wire_spacing = kwargs["wire_spacing_y"]

        if (kwargs["offset_row"] == "even" and column % 2 == 0):
            position = (-cell_size*0.5+walls_to_wires
                        + kwargs["walls_width"]+kwargs["absorber_width"]
                        + integer*wire_spacing+offset)

        elif kwargs["offset_row"] == "odd" and column % 2 != 0:
            position = (-cell_size*0.5+walls_to_wires
                        + kwargs["walls_width"]+kwargs["absorber_width"]
                        + integer*wire_spacing+offset)

        else:
            position = (-cell_size*0.5+walls_to_wires
                        + kwargs["walls_width"]+kwargs["absorber_width"]
                        + integer*wire_spacing)

        return position

    for i in range(kwargs["wire_number_x"]):
        for j in range(kwargs["wire_number_y"]):
            defect_x = kwargs["defects"][i][j][0]
            defect_y = kwargs["defects"][i][j][1]
            x_ = place(i, i, "x_axis", **kwargs)
            y_ = place(j, i, "y_axis", **kwargs)
            center = mp.Vector3(x_+defect_x, y_+defect_y, 0.)

            if kwargs["wire_shape"] == "square":
                size = mp.Vector3(x=kwargs["wire_size"], y=kwargs["wire_size"])
                geometry.append(mp.Block(material=kwargs["wire_material"],
                                         size=size,
                                         center=center))
            else:
                geometry.append(mp.Cylinder(material=kwargs["wire_material"],
                                            radius=0.5*kwargs["wire_size"],
                                            center=center))

    return geometry


def sample_point_standard_2D_grid(**kwargs):

    point = mp.Vector3((0.5*kwargs["wire_spacing_x"]
                        * (kwargs["wire_number_x"] % 2))
                       + .4*kwargs["wire_size"],
                       + .4*kwargs["wire_size"],
                       0.)
    return point


def sample_point_offset_2D_grid(**kwargs):

    def get_sample_column(**kwargs):

        a_ = kwargs["wire_spacing_x"]
        n_ = kwargs["wire_number_x"]

        if "offset_row" not in kwargs:
            r_ = -1
        elif kwargs["offset_row"] == "even":
            r_ = -1
        else:
            r_ = 1

        if n_ % 2 != 0:
            return 0.5*a_+(-1)**((0.5*n_)+(r_*0.5))*0.5*a_

        else:
            return (-1)**(0.5*n_)*((r_*0.5)*a_)

    point = mp.Vector3(get_sample_column(**kwargs),
                       (0.5*kwargs["wire_spacing_y"]
                        * (kwargs["wire_number_y"] % 2)),
                       0.)
    return point


def get_offset_set_tuning_wires(**kwargs):

    delta_steps = kwargs["offset_steps"]
    kwargs["parameter_set"] = []

    if kwargs["offset_axis"] == "x":
        max_delta = kwargs["wire_spacing_x"]-kwargs["wire_size"]

    elif kwargs["offset_axis"] == "y":
        max_delta = 0.5*kwargs["wire_spacing_y"]-0.5*kwargs["wire_size"]

    delta_set = list(np.linspace(0, max_delta, delta_steps, endpoint=True))
    kwargs["parameter_set"] = delta_set

    return kwargs


# -----------------------------------------------------------------------------
# controlling simulations

def get_geometry(**kwargs):

    geometry = []

    if kwargs["walls_width"] > 0:
        geometry = geometry_2D_walls(geometry, **kwargs)

    if kwargs["wire_number_x"] > 0:

        if check_if_offset_geometry(**kwargs):
            geometry = geometry_2D_wire_grid_offset(geometry, **kwargs)
        else:
            geometry = geometry_2D_wire_grid(geometry, **kwargs)

    return geometry


def setup_simulation_2D_grid(**kwargs):

    geometry = get_geometry(**kwargs)
    symmetries = get_symmetries(**kwargs)

    if kwargs["solver"] == "TD_harminv":
        s_type = mp.GaussianSource(frequency=kwargs["source_frequency"],
                                   fwidth=kwargs["source_frequency_width"])

    elif kwargs["solver"] == "FD_eigensolver":
        s_type = mp.ContinuousSource(frequency=kwargs["source_frequency"])

    else:
        ValueError("Unsupported frequency solver")

    size_x = (kwargs["cell_size"][0]
              - 2.*(kwargs["walls_width"]+kwargs["absorber_width"]))
    size_y = (kwargs["cell_size"][1]
              - 2.*(kwargs["walls_width"]+kwargs["absorber_width"]))
    source = mp.Source(s_type, component=mp.Ez, center=mp.Vector3(0., 0., 0.),
                       size=mp.Vector3(size_x, size_y, 0))

    force_complex_fields = check_if_force_complex_fields(**kwargs)
    sim = mp.Simulation(cell_size=kwargs["cell_size"], geometry=geometry,
                        sources=[source], resolution=kwargs["resolution"],
                        filename_prefix="",
                        symmetries=symmetries,
                        force_complex_fields=force_complex_fields)

    if check_if("boundary_condition", **kwargs) == "open":
        sim.boundary_layers = kwargs["absorber_layers"]

    sim, directory = set_output_directory(sim, **kwargs)
    sim.get_filename_prefix()

    return sim


def get_symmetries(**kwargs):

    if kwargs["get_defects"] is False:
        if check_if_offset_geometry(**kwargs):

            if np.abs(kwargs["wire_offset_y"]) > 0:
                symmetries = []

            else:
                symmetries = [mp.Mirror(mp.Y, phase=1)]

        else:
            if ("parameter_name" in kwargs
            and kwargs["parameter_name"] == "wire_number_x"
            and kwargs["wire_number_x"] != kwargs["wire_number_y"]):
                symmetries = [mp.Mirror(mp.Y, phase=1)]

            else:
                symmetries = [mp.Mirror(mp.X, phase=1),
                              mp.Mirror(mp.Y, phase=1)]
    else:
        symmetries = []

    return symmetries


def get_symmetry_factor(sim, **kwargs):

    if check_if("plot_subcell", **kwargs):
        if len(sim.symmetries) == 2:
            return 4

        elif len(sim.symmetries) == 1:
            return 2

    else:
        return 1


def run_harminv_2D_grid(sim, **kwargs):

    sample_point = sample_point_offset_2D_grid(**kwargs)
    h_ = mp.Harminv(mp.Ez, sample_point, kwargs["source_frequency"],
                    kwargs["source_frequency_width"])

    if check_if("harminv_max_bands", **kwargs) is not False:
        h_.mxbands = kwargs["harminv_max_bands"]

    # h_.Q_thresh = 1.

    sim.run(mp.at_beginning(mp.output_epsilon), mp.after_sources(h_),
            until_after_sources=kwargs["until_after_sources"])

    if len(h_.modes) == 0:
        for i in [1, 2, 3]:
            sim.run(h_, until=50)
            if len(h_.modes) > 0:
                return h_, sim

    return h_, sim


def get_freqs_for_2D_grid(**kwargs):

    sim = setup_simulation_2D_grid(**kwargs)
    print(f"Est memory usage={sim.get_estimated_memory_usage()*9.31e-10} Gb")

    sim.plot2D()

    if check_if("plot_sampling", **kwargs) is True:
        plot_sample_point(sim, **kwargs)

    plt.savefig(kwargs["directory"]+"/mockup_plot.png", dpi=600)
    plt.close()

    if kwargs["solver"] == "FD_eigensolver":
        sim.init_sim()
        eigfreq = sim.solve_eigfreq(tol=1e-3, L=20,
                                    guessfreq=kwargs["source_frequency"])
        freqs = np.real(eigfreq)
        field = sim.get_array(mp.Ez)
        freq = str(np.round(np.real(eigfreq), decimals=3))
        plt.title("Real field for eigfreq="+freq)
        plt.imshow(np.real(field).transpose(), cmap="RdBu", origin="lower")
        get_imshow_format()
        plt.close()
        plt.title("complex field for eigfreq="+freq)
        plt.imshow(np.imag(field).transpose(), cmap="RdBu", origin="lower")
        get_imshow_format()
        plt.close()

    if kwargs["solver"] == "TD_harminv":
        h_, sim = run_harminv_2D_grid(sim, **kwargs)
        freqs = np.asarray([m.freq for m in h_.modes])

    freqs_SIG = frequency_MP_to_SI(freqs, **kwargs)/1e+9

    if len(freqs) == 1:

        if check_if("get_max_field_slice", **kwargs):
            sim = get_max_field_slice_2D(sim, freqs[0], **kwargs)

        if check_if("get_field_slices", **kwargs):
            sim = get_field_slices_2D(sim, freqs_SIG, **kwargs)

        if check_if("get_phase_plot", **kwargs):
            sim = get_phase_plot(sim, freqs_SIG, **kwargs)

    if (check_if("get_fields", **kwargs) and len(freqs) == 1):
        sim, sim_output = get_fields(sim, freqs_SIG, **kwargs)
    else:
        sim_output = get_empty_fields_set(**kwargs)

    if len(freqs) == 0:
        sim = get_max_field_slice_2D(sim, kwargs["source_frequency"],
                                     **kwargs)

    if check_if("get_steady_state_fields", **kwargs):
        sim, sim_output = get_steady_state_fields(sim, h_, sim_output,
                                                  **kwargs)

    if check_if("get_error_estimates", **kwargs):
        sim, sim_output = get_error_estimates(sim, h_, sim_output, **kwargs)

    sim_output["frequencies"] = [list(freqs)]
    sim_output["modes"] = [list(h_.modes)]

    return freqs_SIG, h_.modes, sim_output, sim


def get_freqs_2D_grid_delta_parameter(**kwargs):

    freq_set = []
    mode_set = []
    sim_output_set = {}
    sim_time_set = []
    generic_parameters = ["resolution", "until_after_sources",
                          "sample_length_dft_in_source_periods"]

    for parameter in kwargs["parameter_set"]:
        start = time.time()
        directory_number = str(kwargs["parameter_set"].index(parameter))
        kwargs["directory"] = "out"+directory_number

        if kwargs["parameter_name"] == "epsilon_r":
            kwargs["wire_material"] = get_simple_metall(epsilon_r=parameter,
                                                        **kwargs)

        elif kwargs["parameter_name"] == "E_conductivity":
            kwargs["wire_material"] = get_simple_metall(
                                      E_conductivity=parameter,
                                      **kwargs)

        elif kwargs["parameter_name"] == "source_frequency_width":
            kwargs["source_frequency_width"] = parameter

        elif kwargs["parameter_name"] == "offset":
            if kwargs["offset_axis"] == "x":
                kwargs["wire_offset_x"] = parameter

            elif kwargs["offset_axis"] == "y":
                kwargs["wire_offset_y"] = parameter

            kwargs["defects"] = get_defects(**kwargs)

            if not check_if("use_previous_frequency_output", **kwargs):
                kwargs["source_frequency"] = parameter[1]
                kwargs["source_frequency_width"] = parameter[2]

            else:
                if kwargs["parameter_set"].index(parameter) > 0:
                    kwargs["source_frequency"] = mode_set[-1][0].freq

        elif kwargs["parameter_name"] == "wire_number":
            kwargs["wire_number_x"] = parameter
            kwargs["wire_number_y"] = parameter
            kwargs["defects"] = get_defects(**kwargs)
            kwargs["cell_size"] = cell_size_2D_wire_grid(**kwargs)
            if kwargs["parameter_set"].index(parameter) > 0:
                kwargs["source_frequency"] = mode_set[-1][0].freq
                kwargs["source_frequency_width"] = (1.9*(TM_nm0_WM_2D(1, 2,
                                                                      **kwargs)
                                                    - TM_nm0_WM_2D(**kwargs)))

        elif kwargs["parameter_name"] == "source_frequency_and_its_width":
            kwargs["source_frequency"] = parameter[0]
            kwargs["source_frequency_width"] = parameter[1]*parameter[0]

        elif kwargs["parameter_name"] == "absorber_width":
            kwargs["absorber_width"] = parameter/kwargs["source_frequency"]
            kwargs["absorber_layers"] = [mp.Absorber(kwargs["absorber_width"])]
            kwargs["cell_size"] = cell_size_2D_wire_grid(**kwargs)

        elif kwargs["parameter_name"] == "absorber_to_wires_width":
            kwargs["walls_width"] = parameter
            kwargs["cell_size"] = cell_size_2D_wire_grid(**kwargs)

        elif kwargs["parameter_name"] == "defects_single_position":
            kwargs["parameter"] = parameter
            if parameter == ["-", "-"]:
                kwargs["defects"] = np.zeros(shape=(kwargs["wire_number_x"],
                                                    kwargs["wire_number_y"],
                                                    2))
            else:
                kwargs["defects_position"] = parameter
                kwargs["defects"] = get_defects(**kwargs)

        elif kwargs["parameter_name"] == "defects_single_size":
            kwargs["parameter"] = parameter
            kwargs["defects_size"] = parameter
            kwargs["defects"] = get_defects(**kwargs)
            if kwargs["parameter_set"].index(parameter) > 0:
                kwargs["source_frequency"] = mode_set[-1][0].freq

        elif kwargs["parameter_name"] == "defects_all_random":
            kwargs["parameter"] = parameter
            kwargs["defects_number"] = parameter
            kwargs["defects"] = get_defects(**kwargs)

        elif kwargs["parameter_name"] in generic_parameters:
            kwargs[kwargs["parameter_name"]] = parameter

        else:
            raise ValueError("Unsupported input parameter")

        freq, mode, sim_output, sim = get_freqs_for_2D_grid(**kwargs)
        sim_time_set.append((time.time()-start)/60.)
        sim.reset_meep()
        kwargs["directory"] = "out"

        if not list(freq):
            freq_set.append([0.])
            mode_set.append([0])
            print("Harminv could not find modes, remove 0 from data.")

        else:
            freq_set.append(freq)
            mode_set.append(mode)
            if directory_number == "0":
                sim_output_set = sim_output.copy()
            else:
                for key in sim_output_set:
                    sim_output_set[key].append(sim_output[key].pop(0))

        write_data_to_csv_file("simulation_time",
                               [kwargs["parameter_name"], "minutes"],
                               [kwargs["parameter_set"]] + [sim_time_set],
                               kwargs["directory"])
        save_output(freq_set, mode_set, sim_output_set, **kwargs)

    return freq_set, mode_set, sim_output_set


def set_output_directory(sim, **kwargs):

    if "directory" in kwargs:
        sim.use_output_directory(dname=kwargs["directory"])
        return sim, kwargs["directory"]
    else:
        sim.use_output_directory(dname="out")
        return sim, "out"


def get_subcell_center_size(sim, **kwargs):

    cell_size_x = (sim.cell_size[0]
                   - 2*(kwargs["walls_width"]+kwargs["absorber_width"]))
    cell_size_y = (sim.cell_size[1]
                   - 2*(kwargs["walls_width"]+kwargs["absorber_width"]))

    if check_if("plot_subcell", **kwargs):

        if len(sim.symmetries) == 2:
            center = 0.25*mp.Vector3(-cell_size_x, -cell_size_y)
            size = 0.5*mp.Vector3(cell_size_x, cell_size_y)

        elif len(sim.symmetries) == 1:
            center = mp.Vector3(0, -0.25*cell_size_y)
            size = mp.Vector3(cell_size_x, 0.5*cell_size_y)

        else:
            center = mp.Vector3()
            size = mp.Vector3(cell_size_x, cell_size_y)

    else:
        center = mp.Vector3()
        size = mp.Vector3(cell_size_x, cell_size_y)

    return center, size


# -----------------------------------------------------------------------------
# simulation output chosen in main script


def get_max_field_slice_2D(sim, frequency, **kwargs):

    def get_max_field_slice(sim):

        field_new = sim.get_array(component=mp.Ez, center=center,
                                  size=size, cmplx=False)
        field_new_set = [field_new, np.max(np.abs(field_new))]
        if field_old_set[1] < field_new_set[1]:
            field_old_set[0] = field_new_set[0]
            field_old_set[1] = field_new_set[1]

    until = 0.5/frequency
    slices = 10
    center, size = get_subcell_center_size(sim, **kwargs)
    field_old = sim.get_array(component=mp.Ez, center=center,
                              size=size, cmplx=False)
    field_old_set = [field_old, np.max(np.abs(field_old))]
    eps = sim.get_array(size=size,
                        center=center,
                        component=mp.Dielectric)
    sim.run(mp.at_every(until/slices, get_max_field_slice), until=until)
    max_field = field_old_set[0]
    max_field_ez_SI = E_field_MP_to_SI(max_field, **kwargs)
    plot_name = "max_field_slice_Ez"
    get_plt_image(max_field_ez_SI, eps, plot_name, "_", **kwargs)

    sim.filename_prefix = ""

    return sim


def get_field_slices_2D(sim, frequency, **kwargs):

    fields = [mp.Ez, mp.Hx, mp.Hy]
    path = "/home/i/miniconda/envs/mp_p/share/h5utils/colormaps/"
    # path = "/home/i/miniconda/envs/mp/share/h5utils/colormaps/"
    cmap = "dkbluered"
    slices = 10
    until = 0.5/((frequency[0]*kwargs["meep_unit"]*1e+9)/con.c)
    slice_specs = "-v8RZc "+path+cmap+" -C $EPS"
    # slice_specs = "-v8RZc "+cmap+" -C $EPS"
    for field in fields:
        sim.run(mp.at_every(until/slices,
                            mp.output_png(field, slice_specs)), until=until)

    return sim


def get_phase_plot(sim, frequency, **kwargs):

    center, size = get_subcell_center_size(sim, **kwargs)

    for pair in [[mp.Ez, "Ez"], [mp.Hx, "Hx"], [mp.Hy, "Hy"]]:
        filename = "phase_plot_"+pair[1]
        phase = np.angle(sim.get_array(component=pair[0], cmplx=True,
                                       center=center, size=size))

        if check_if("get_defects", **kwargs):
            get_title_defects(**kwargs)

        plt.imshow(phase.transpose(), interpolation="none", cmap="Purples",
                   origin="lower")
        get_ticks_format(phase, **kwargs)
        plt.colorbar()
        plt.savefig(kwargs["directory"]+"/"+filename+".png",
                    bbox_inches='tight', dpi=600)
        plt.close()

    return sim


def get_fields(sim, frequency, field_component=mp.Ez, **kwargs):

    center, size = get_subcell_center_size(sim, **kwargs)
    (Ez, Hx, Hy) = [sim.get_array(component=c, cmplx=True, center=center,
                                  size=size) for c in [mp.Ez, mp.Hx, mp.Hy]]
    eps = sim.get_array(component=mp.Dielectric, center=center,
                        size=size)
    (x, y, z, w) = sim.get_array_metadata(center=center, size=size)

    Ez_amp = get_Ai_amp(sim, Ez, mp.Ez, eps, **kwargs)
    Hx_amp = get_Ai_amp(sim, Hx, mp.Hx, eps, **kwargs)
    Hy_amp = get_Ai_amp(sim, Hy, mp.Hy, eps, **kwargs)

    f = {"fields_energy_stored_in_Ez": [get_energy_stored_in_Ez(sim, Ez, eps,
                                                                w, **kwargs)],
         "fields_energy_stored_in_H": [get_energy_stored_in_H(sim, w,
                                                              **kwargs)],
         "fields_overlap_Ez": [get_overlap(sim, Ez, w, **kwargs)],
         "fields_overlap_Hx": [get_overlap(sim, Hx, w, **kwargs)],
         "fields_overlap_Hy": [get_overlap(sim, Hy, w, **kwargs)],
         "fields_overlap_rephased_Ez": [get_overlap_rephased(sim, Ez_amp, w,
                                                             **kwargs)],
         "fields_overlap_rephased_Hx": [get_overlap_rephased(sim, Hx_amp, w,
                                                             **kwargs)],
         "fields_overlap_rephased_Hy": [get_overlap_rephased(sim, Hy_amp, w,
                                                             **kwargs)],
         "fields_asymmetry_Ez": [get_field_asymmetry(sim, mp.Ez, **kwargs)],
         "fields_asymmetry_Hx": [get_field_asymmetry(sim, mp.Hx, **kwargs)],
         "fields_asymmetry_Hy": [get_field_asymmetry(sim, mp.Hy, **kwargs)]
         # "field_comparison_Ez": [get_field_comparison_with_ideal(sim, Ez_amp,
                                                                 # "Ez", eps,
                                                                 # **kwargs)],
         # "field_comparison_Hx": [get_field_comparison_with_ideal(sim, Hx_amp,
                                                                 # "Hx", eps,
                                                                 # **kwargs)],
         # "field_comparison_Hy": [get_field_comparison_with_ideal(sim, Hy_amp,
                                                                 # "Hy", eps,
                                                                 # **kwargs)]
         }

    return sim, f


def get_empty_fields_set(**kwargs):

    fields = {"fields_energy_stored_in_Ez": [0],
              "fields_energy_stored_in_H": [0],
              "fields_overlap_Ez": [0],
              "fields_overlap_Hx": [0],
              "fields_overlap_Hy": [0],
              "fields_overlap_rephased_Ez": [0],
              "fields_overlap_rephased_Hx": [0],
              "fields_overlap_rephased_Hy": [0],
              "fields_asymmetry_Ez": [[0, 0]],
              "fields_asymmetry_Hx": [[0, 0]],
              "fields_asymmetry_Hy": [[0, 0]],
              # "field_comparison_Ez": [[0, 0, 0]],
              # "field_comparison_Hx": [[0, 0, 0]],
              # "field_comparison_Hy": [[0, 0, 0]]
              }
    if check_if("get_error_estimates", **kwargs):
        fields["error_estimates_std_Ez"] = [0]
        fields["error_estimates_std_Ez_rephased"] = [0]
        fields["error_estimates_mean_Ez"] = [0]
        fields["error_estimates_mean_Ez_rephased"] = [0]
        fields["error_estimates_std_phase"] = [0]

    return fields


def get_steady_state_fields(sim, h_, sim_output, **kwargs):

    def get_runtime_overlap(sim):
        Ai = sim.get_dft_array(dfts, component=mp.Ez, num_freq=0)
        Ai_amp = get_Ai_amp(sim, Ai, mp.Ez, eps, dfts, 0, **kwargs)
        overlap = get_overlap(sim, Ai, w, **kwargs)
        overlap_rephased = get_overlap_rephased(sim, Ai_amp, w, **kwargs)

        for s in [(func_name+"overlap_runtime", overlap),
                  (func_name+"overlap_rephased_runtime",
                   overlap_rephased)]:
            kwargs[s[0]].append(s[1])
            # l_ = kwargs[s[0]]
            # print(f"harminv_runtime_{s[0]}: {l_}\n")

    def get_stop_condition_overlap(sim):

        if len(kwargs[func_name+"overlap_runtime"]) > 1:
            O_new_R = np.real(kwargs[func_name+"overlap_runtime"][-1])
            O_old_R = np.real(kwargs[func_name+"overlap_runtime"][-2])
            # O_new_I = np.imag(kwargs[func_name+"overlap_runtime"][-1])
            # O_old_I = np.imag(kwargs[func_name+"overlap_runtime"][-2])
            condition_1_R = (np.abs(O_old_R-O_new_R)/O_old_R <=
                             kwargs[func_name+"overlap_stop_condition"])
            # condition_1_I = (np.abs(O_old_I-O_new_I)/O_old_I <=
            #                  kwargs[func_name+"overlap_stop_condition"])

            # condition_1 = condition_1_I and condition_1_R
            condition_1 = condition_1_R

        else:
            condition_1 = False

        max_runtime = ((2*sim.sources[0].src.cutoff*sim.sources[0].src.width)
                       + (kwargs[func_name+"sample_periods_max"]
                          * 1/kwargs["source_frequency"]))
        condition_2 = sim.meep_time() >= max_runtime

        return condition_1 or condition_2

    func_name = "steady_state_fields_"
    center, size = get_subcell_center_size(sim, **kwargs)
    (x, y, z, w) = sim.get_array_metadata(center=center, size=size)

    field_components = [mp.Ez, mp.Hx, mp.Hy]
    field_names = ["Ez", "Hx", "Hy"]
    dfts = sim.add_dft_fields(field_components,
                              np.asarray([m.freq for m in h_.modes]),
                              where=mp.Volume(center, size))
    eps = sim.get_array(center=center, size=size, component=mp.Dielectric)

    kwargs[func_name+"overlap_runtime"] = []
    kwargs[func_name+"overlap_rephased_runtime"] = []

    sim.run(until=(kwargs[func_name+"sample_periods_max"]
                   / kwargs["source_frequency"]))
    sim.run(mp.at_every(2/kwargs["source_frequency"], get_runtime_overlap),
            until=get_stop_condition_overlap)

    for i in range(len(kwargs[func_name+"overlap_runtime"])):
        plt.scatter(i*2,
                    np.real(kwargs[func_name+"overlap_runtime"][i]),
                    color="k")
    plt.savefig(kwargs["directory"]+"/"+func_name+"overlap_runtime.png",
                dpi=600)

    D_ = {"overlap_runtime": kwargs[func_name+"overlap_runtime"],
          "overlap_rephased_runtime": kwargs[func_name
                                             + "overlap_rephased_runtime"]}
    save_dict(D_, func_name+"runtime_overlaps", **kwargs)

    for i in range(len(dfts.freq)):
        for j in range(len(field_components)):
            name = func_name+"freq_"+str(i)+"_"+field_names[j]
            field_dft = sim.get_dft_array(dfts, component=field_components[j],
                                          num_freq=i)
            get_plt_image(field_dft, eps, name, **kwargs)

            if field_names[j] == "Ez":
                print("######################################################")
                print(f"{func_name}: DFT field for mode i={i}: "
                      f"Mean real Ez dft={np.mean(np.real(field_dft))} "
                      f"Overlap Ez={get_overlap(sim, field_dft, w, **kwargs)}")
                print("######################################################")

            if i == 0:
                if field_names[j] == "Ez":
                    Ez = field_dft
                    Ai_amp = get_Ai_amp(sim, field_dft, field_components[j],
                                        eps, dfts, i, **kwargs)
                    Ez_amp = Ai_amp
                elif field_names[j] == "Hx":
                    Hx = field_dft
                    # Hx_amp = Ai_amp
                else:
                    Hy = field_dft
                    # Hy_amp = Ai_amp

            if field_names[j] == "Ez" and i != 0:
                get_plt_image(np.real(Ez-field_dft), eps, func_name
                              + "_difference_plot_Ez0_minus_Ez1", "_",
                              **kwargs)

    sim_output[func_name+"overlap_Ez"] = [get_overlap(sim, Ez, w, **kwargs)]
    sim_output[func_name+"overlap_Hx"] = [get_overlap(sim, Hx, w, **kwargs)]
    sim_output[func_name+"overlap_Hy"] = [get_overlap(sim, Hy, w, **kwargs)]
    sim_output[func_name+"overlap_rephased_Ez"] = [get_overlap_rephased(sim,
                                                                        Ez_amp,
                                                                        w,
                                                                        **kwargs)]
    sim_output[func_name+"average_Ez_norm"] = [get_average_Ez_norm(sim, Ez, w,
                                                                   **kwargs)]

    field_p = mp.Vector3(0,
                         2*kwargs["wire_size"]*(kwargs["wire_number_x"] % 2))
    matrix_p = [0, field_p[1]*kwargs["resolution"]]

    E2 = Ez[int(0.5*Ez.shape[0]+matrix_p[0])][int(0.5*Ez.shape[1]+matrix_p[1])]
    E2_norm = np.sqrt(np.vdot(E2, E2))/np.max(np.abs(Ez))
    E3 = Ez_amp[int(0.5*Ez.shape[0]+matrix_p[0])][(int(0.5*Ez.shape[1]
                                                       + matrix_p[1]))]
    E3_norm = np.sqrt(np.vdot(E3, E3))/np.max(np.abs(Ez_amp))
    sim_output[func_name+"E_field_norm_antenna"] = [E2_norm]
    sim_output[func_name+"E_field_norm_antenna_amp"] = [E3_norm]

    return sim, sim_output


def get_error_estimates(sim, h_, sim_output, **kwargs):

    name = "error_estimates_"
    sample_point = sample_point_offset_2D_grid(**kwargs)
    sample_area = 0.01*mp.Vector3(sim.cell_size[0], sim.cell_size[1])
    Ez_area = sim.get_array(mp.Ez, center=sample_point, cmplx=True,
                            size=sample_area)
    phase = np.angle(sim.get_array(component=mp.Ez, cmplx=True,
                                   center=sample_point,
                                   size=mp.Vector3()))
    Ez_area_rephased = np.exp((-1j)*phase)*Ez_area

    sim_output[name+"std_Ez"] = [(np.std(np.real(Ez_area))
                                  + np.std(np.imag(Ez_area))*1j)]
    sim_output[name+"std_Ez_rephased"] = [(np.std(np.real(Ez_area_rephased))
                                           + np.std(np.imag(Ez_area_rephased))
                                           * 1j)]
    sim_output[name+"mean_Ez"] = [(np.mean(np.real(Ez_area))
                                   + np.mean(np.imag(Ez_area))*1j)]
    sim_output[name+"mean_Ez_rephased"] = [(np.mean(np.real(Ez_area_rephased))
                                            + np.mean(np.imag(Ez_area_rephased
                                                              ))
                                            * 1j)]
    sim_output[name+"std_phase"] = [np.std(np.angle(Ez_area))]
    sim_output[name+"mean_phase"] = [np.mean(np.angle(Ez_area))]

    return sim, sim_output


def get_energy_stored_in_Ez(sim, Ez, eps, w, **kwargs):

    energy_density = np.real(eps*np.conj(Ez)*Ez)
    energy = np.sum(w*energy_density)

    return get_symmetry_factor(sim, **kwargs)*energy_MP_to_SI(energy, **kwargs)


def get_energy_stored_in_H(sim, w, **kwargs):

    center, size = get_subcell_center_size(sim, **kwargs)
    (Hx, Hy) = [sim.get_array(center=center, size=size,
                              component=c, cmplx=True) for c in [mp.Hx, mp.Hy]]

    energy_density = np.real(np.conj(Hx)*Hx+np.conj(Hy)*Hy)
    energy = np.sum(w*energy_density)

    return get_symmetry_factor(sim, **kwargs)*energy_MP_to_SI(energy, **kwargs)


def get_Ai_amp(sim, Ai, component, eps, dfts=False, freq_index=False,
               **kwargs):

    wire = sim.geometry[int(len(sim.geometry)/2)].center

    if component == mp.Hx:
        field = "Hx"
        sample_point = mp.Vector3(wire[0], wire[1]+1.2*kwargs["wire_size"])
    elif component == mp.Hy:
        field = "Hy"
        sample_point = mp.Vector3(wire[0]+1.2*kwargs["wire_size"], wire[1])
    elif component == mp.Ez:
        field = "Ez"
        sample_point = sample_point_offset_2D_grid(**kwargs)

    if dfts is False:
        title = "fields_"+field
        phase = np.angle(sim.get_array(component=component,
                                       cmplx=True, center=sample_point,
                                       size=mp.Vector3()))
    else:
        n = int(sample_point[0]*sim.resolution)
        m = int(sample_point[1]*sim.resolution)
        phase = np.angle(Ai[n][m])
        title = "steady_state_fields_freq_"+str(freq_index)+"_"+field
    Ai_amp = np.exp((-1j)*phase)*Ai

    if dfts is False:
        get_plt_image(Ai_amp, eps, title+"_rephased_amp", "_", **kwargs)

        if component == mp.Ez:
            get_plt_image(np.imag(Ai)/np.abs(Ai), eps,
                          title+"_error_plot_full_field", "_", **kwargs)
            get_plt_image(np.imag(Ai_amp)/np.abs(Ai_amp), eps,
                          title+"_error_plot_rephased_amp", "_", **kwargs)

    return Ai_amp


def get_overlap(sim, Ai, w, **kwargs):

    sym_factor = get_symmetry_factor(sim, **kwargs)
    center, size = get_subcell_center_size(sim, **kwargs)
    volume = size[0]*size[1]

    return (sym_factor*np.sum(w*Ai)*np.conjugate(np.sum(w*Ai))
            / (volume*np.sum(w*(np.conj(Ai)*Ai))))


def get_overlap_rephased(sim, Ai_amp, w, **kwargs):

    sym_factor = get_symmetry_factor(sim, **kwargs)
    center, size = get_subcell_center_size(sim, **kwargs)
    volume = size[0]*size[1]

    return ((sym_factor*(np.sum(w*Ai_amp)**2))
            / (volume*np.sum(w*(np.abs(Ai_amp)**2))))


def get_field_asymmetry(sim, field_type, **kwargs):

    quacks = kwargs.copy()
    quacks["plot_subcell"] = False
    center, size = get_subcell_center_size(sim, **quacks)
    Ai_full = sim.get_array(component=field_type, cmplx=True, center=center,
                            size=size)

    def get_diff(x_, y_):

        return np.sum(np.abs((x_-y_)).flatten())

    diff = [0, 0]

    for i in range(2):
        if i == 0:
            Ai = np.real(Ai_full)/np.abs(np.real(Ai_full).max())
        else:
            Ai = np.imag(Ai_full)/np.abs(np.imag(Ai_full).max())

        ul = np.array_split(np.array_split(Ai, 2, axis=1)[0], 2, axis=0)[0]
        bl = np.array_split(np.array_split(Ai, 2, axis=1)[0], 2, axis=0)[1]
        ur = np.array_split(np.array_split(Ai, 2, axis=1)[1], 2, axis=0)[0]
        br = np.array_split(np.array_split(Ai, 2, axis=1)[1], 2, axis=0)[1]

        if field_type == mp.Ez:
            diff[i] = (get_diff(ul, np.fliplr(ur))
                       + get_diff(ul, np.flipud(bl))
                       + get_diff(ul, np.rot90(br, 2))
                       + get_diff(ur, np.flipud(br))
                       + get_diff(ur, np.rot90(bl, 2))
                       + get_diff(bl, np.fliplr(br)))

        elif field_type == mp.Hx:
            diff[i] = (get_diff(ul, -np.fliplr(ur))
                       + get_diff(ul, np.flipud(bl))
                       + get_diff(ul, -np.rot90(br, 2))
                       + get_diff(ur, np.flipud(br))
                       + get_diff(ur, -np.rot90(bl, 2))
                       + get_diff(bl, -np.fliplr(br)))

        else:
            diff[i] = (get_diff(ul, np.fliplr(ur))
                       + get_diff(ul, -np.flipud(bl))
                       + get_diff(ul, -np.rot90(br, 2))
                       + get_diff(ur, -np.flipud(br))
                       + get_diff(ur, -np.rot90(bl, 2))
                       + get_diff(bl, np.fliplr(br)))

    return (diff[0]+diff[1]*1j)


def get_field_comparison_with_ideal(sim, Ai_sim, field_name, eps, **kwargs):

    Ai_ideal = np.load("/cfs/home/tokl6780/comparison_fields/"
                       + str(kwargs["wire_number_x"])+"by"
                       + str(kwargs["wire_number_y"])+"/"+field_name
                       + "_full_amp.npy")
    if np.shape(Ai_ideal) == np.shape(Ai_sim):

        Ai_ideal_R = np.real(Ai_ideal)
        Ai_ideal_I = np.imag(Ai_ideal)
        Ai_ideal_RN = Ai_ideal_R/np.abs(Ai_ideal_R).max()
        Ai_ideal_IN = Ai_ideal_I/np.abs(Ai_ideal_I).max()

        Ai_sim_R = np.real(Ai_sim)
        Ai_sim_I = np.imag(Ai_sim)
        Ai_sim_RN = Ai_sim_R/np.abs(Ai_sim_R).max()
        Ai_sim_IN = Ai_sim_I/np.abs(Ai_sim_I).max()

        get_plt_image(Ai_ideal_RN-Ai_sim_RN, eps,
                      field_name+"_comparison_with_ideal_Real_Normalised", "_",
                      **kwargs)
        get_plt_image(Ai_ideal_IN-Ai_sim_IN, eps,
                      field_name+"_comparison_with_ideal_Imag_Normalised", "_",
                      **kwargs)

        diff = [np.sum(np.abs(Ai_ideal_RN-Ai_sim_RN)).flatten(),
                np.sum(np.abs(Ai_ideal_IN-Ai_sim_IN)).flatten()]
    else:
        diff = ["NA", "NA"]

    return diff


def get_average_Ez_norm(sim, Ez_amp, w, **kwargs):

    area = ((kwargs["wire_spacing_x"]*(kwargs["wire_number_x"]-1)
            + 2*(kwargs["walls_to_wires_x"]))
            * (kwargs["wire_spacing_y"]*(kwargs["wire_number_y"]-1)
            + 2*(kwargs["walls_to_wires_y"])))

    return (get_symmetry_factor(sim, **kwargs)
            * E_field_MP_to_SI(np.sum(w*np.abs(Ez_amp))/area, **kwargs))


def animate_Ez(sim, frequency, fields=[mp.Ez], **kwargs):

    for field in fields:
        animate = mp.Animate2D(sim, fields=field, f=plt.figure(dpi=150),
                               realtime=False, normalize=False)
        sim.run(mp.at_every(1/frequency/20, animate), until=1/frequency)
        plt.close()
        fps = 5
        frequency_SIG = frequency_MP_to_SI(frequency, **kwargs)/1e+9
        animate.to_mp4(fps, ("out/field_"+str(field)+"freq_"
                             + str(np.round(frequency_SIG, 2))+".mp4"))

    return sim



# -----------------------------------------------------------------------------
# simulation logics


def check_if_offset_geometry(**kwargs):

    if "wire_offset_x" in kwargs or "wire_offset_y" in kwargs:
        return kwargs["wire_offset_x"] != 0. or kwargs["wire_offset_y"] != 0.
    else:
        return False


def check_if(key, **kwargs):

    if key in kwargs:
        return kwargs[key]
    else:
        return False


def check_if_force_complex_fields(**kwargs):

    if "solver" in kwargs and kwargs["solver"] == "FD_eigensolver":
        return True

    elif check_if("get_phase_plot", **kwargs):
        return True

    elif check_if("get_fields", **kwargs):
        return True

    elif check_if("get_error_estimates", **kwargs):
        return True

    else:
        return False


# -----------------------------------------------------------------------------
# analytical predictions


def TM_freq_rect_2D_waveguide(n, m, **kwargs):
    """Transverse magnetic mode, empty rectangular waveguide with PEC walls."""

    if n == 0 or m == 0:
        raise ValueError("Unsupported input parameter")
    else:
        mode = 0.5*np.sqrt((n/kwargs["cell_size"][0])**2
                           + (m/kwargs["cell_size"][1])**2)
        return mode


def TE_freq_rect_2D_waveguide(n, m, **kwargs):
    """Transverse electric mode, empty rectangular waveguide with PEC walls."""

    if n == 0 and m == 0:
        raise ValueError("Unsupported input parameter")

    elif m == 0:
        return 0.5*np.sqrt((n/kwargs["cell_size"][0])**2)

    elif n == 0:
        return 0.5*np.sqrt((m/kwargs["cell_size"][1])**2)

    else:
        mode = 0.5*np.sqrt((n/kwargs["cell_size"][0])**2
                           + (m/kwargs["cell_size"][1])**2)
        return mode


def TM_nodes_freqs_rect_2D_waveguide(number_of_modes=10, **kwargs):
    """Transverse electric modes for rectangular waveguide with PEC walls."""

    cap_f = 5*(kwargs["source_frequency"]+0.5*kwargs["source_frequency_width"])

    freqs = [TM_freq_rect_2D_waveguide(nm[0], nm[1], **kwargs)
             for nm in it.product(np.arange(1, number_of_modes), repeat=2)
             if TM_freq_rect_2D_waveguide(nm[0], nm[1], **kwargs) <= cap_f]

    nodes_freqs = list(zip(list(it.product(np.arange(1, number_of_modes),
                                           repeat=2)),
                           freqs))

    return nodes_freqs


def TE_nodes_freqs_rect_2D_waveguide(number_of_modes=10, **kwargs):
    """Transverse magnetic modes for rectangular waveguide with PEC walls."""

    cap_f = 5*(kwargs["source_frequency"]+0.5*kwargs["source_frequency_width"])
    freqs, nm_set = [], []
    f_01 = TE_freq_rect_2D_waveguide(0, 1, **kwargs)
    f_10 = TE_freq_rect_2D_waveguide(1, 0, **kwargs)
    if f_01 <= cap_f:
        freqs += [f_01]
        nm_set += [(0, 1)]
    if f_10 <= cap_f:
        freqs += [f_10]
        nm_set += [(1, 0)]

    TE_freqs = list(zip(nm_set, freqs))
    TE_freqs += TM_nodes_freqs_rect_2D_waveguide(**kwargs)
    return TE_freqs


def Ez_nm0_WM(x_, y_, n=1, m=1, **kwargs):

    sx = (kwargs["wire_spacing_x"]*(kwargs["wire_number_x"]-1)
          + 2*kwargs["walls_to_wires_x"])
    sy = (kwargs["wire_spacing_y"]*(kwargs["wire_number_y"]-1)
          + 2*kwargs["walls_to_wires_y"])

    return np.sin(n*np.pi*x_/sx)*np.sin(m*np.pi*y_/sy)


def overlap_lm0(l_=1, m_=1):
    return ((8/((np.pi**2)*l_*m_))*(np.sin(0.5*l_*np.pi)**2)
            * (np.sin(0.5*m_*np.pi)**2))**2


def plot_homogenized_WM(n, m, **kwargs):

    # sim = setup_simulation_2D_grid(**kwargs)
    # sim.init_sim()
    # x_ = np.linspace(0, sim.cell_size[0],
    #                  num=int(kwargs["resolution"]*sim.cell_size[0]))
    # y_ = np.linspace(0, sim.cell_size[1],
    #                  num=int(kwargs["resolution"]*sim.cell_size[1]))

    cell_size = cell_size_2D_wire_grid(**kwargs)
    x_ = np.linspace(0, cell_size[0],
                     num=int(kwargs["resolution"]*cell_size[0]))
    y_ = np.linspace(0, cell_size[1],
                     num=int(kwargs["resolution"]*cell_size[1]))

    Ez = np.zeros((len(x_), len(y_)))
    for i in range(len(x_)):
        for j in range(len(y_)):
            Ez[i][j] = Ez_nm0_WM(x_[i], x_[j], n, m, **kwargs)
    # get_imshow_format(Ez, 0, **kwargs)
    plt.imshow(Ez.transpose(), cmap="Reds", interpolation="none",
               vmax=np.amax(Ez), vmin=np.amin(Ez), origin="lower")
    plt.savefig(kwargs["directory"]+"/"+"homogenized_WM_Ez.png", dpi=600)
    plt.close()

    # (x, y, z, w) = sim.get_array_metadata(center=mp.Vector3(),
    #                                       size=cell_size)
    # overlap = get_overlap(sim, Ez, w, **kwargs)

    return


def TM_nm0_WM_2D(n=1., m=1., **kwargs):
    """Input and output in Meep units."""

    s_x = (kwargs["wire_spacing_x"]*kwargs["wire_number_x"]
           * kwargs["meep_unit"])
    s_y = (kwargs["wire_spacing_y"]*kwargs["wire_number_y"]
           * kwargs["meep_unit"])

    f_p = plasma_frequency_pavels(**kwargs)
    k_p = (2.*np.pi*f_p)/con.c
    k_r = np.sqrt(k_p**2+((n*np.pi)/s_x)**2+((m*np.pi)/s_y)**2)
    f_r = (k_r*con.c)/(2.*np.pi)

    return frequency_SI_to_MP(f_r, **kwargs)


def TM110_to_wp(x, **kwargs):
    "In- and output in SI-units"
    dx = kwargs["wire_spacing_x"]*kwargs["wire_number_x"]
    dy = kwargs["wire_spacing_y"]*kwargs["wire_number_y"]

    return con.c*np.sqrt((x/con.c)**2-(np.pi/dx)**2
                         - (np.pi/dy)**2)


def plasma_frequency(**kwargs):
    """Input Meep units, output in SI units

    From Pendry:
    r_, a_ = 1e-6, 5e-3, gives f = 8.20e+9, condition should be >> 1.
    """
    a_ = kwargs["wire_spacing"]*kwargs["meep_unit"]
    r_ = 0.5*kwargs["wire_size"]*kwargs["meep_unit"]
    omega_p = np.sqrt((2.*np.pi*(con.c**2))/(a_**2*np.log(a_/r_)))
    condition = np.log(a_/r_)

    return omega_p/(2*np.pi), condition


def plasma_frequency_pavels(**kwargs):
    """Input Meep units, output in SI units.

    From Belov et al. radius<spacing*0.1"""

    r_ = 0.5*kwargs["wire_size"]*kwargs["meep_unit"]

    if kwargs["wire_spacing_x"] == kwargs["wire_spacing_y"]:
        a_ = kwargs["wire_spacing_x"]*kwargs["meep_unit"]
        F = 0.5275
    else:
        a_ = np.sqrt(kwargs["wire_spacing_x"]*kwargs["wire_spacing_y"]
                     * kwargs["meep_unit"]**2)
        c_ = kwargs["wire_spacing_x"]/kwargs["wire_spacing_y"]
        F = (np.pi/6.)*(c_+1./c_)-np.sqrt(np.log(c_)**2+(np.pi/3.)**2)*0.5

    # f_p = 1./(np.sqrt(2.*np.pi)*a_*np.sqrt(np.log(a_/(2.*np.pi*r_))+F))
    k_p = np.sqrt(2.*np.pi)/(a_*np.sqrt(np.log(a_/(2.*np.pi*r_))+F))
    return (k_p*con.c)/(2.*np.pi)


def frequency_MP_to_SI(x, **kwargs):

    return (x*con.c)/kwargs["meep_unit"]


def frequency_SI_to_MP(x, **kwargs):

    return (x*kwargs["meep_unit"])/con.c


def E_field_MP_to_SI(x, **kwargs):

    return x/(kwargs["meep_unit"]*con.epsilon_0*con.c)


def E_field_SI_to_MP(x, **kwargs):

    return (kwargs["meep_unit"]*con.epsilon_0*con.c)/x


def energy_MP_to_SI(x, **kwargs):

    return x/(con.epsilon_0*(con.c**2))


def energy_SI_to_MP(x, **kwargs):

    return (con.epsilon_0*(con.c**2))/x


def overlap_MP_to_SI(x, **kwargs):

    return x*kwargs["meep_unit"]*con.epsilon_0*con.c


# -----------------------------------------------------------------------------
# storing output


def write_data_to_csv_file(filename, header, data_sets, directory="out"):

    name = directory+"/"+filename+".csv"
    with open(name, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(header)
        writer.writerows(data_sets)

    return


def save_dict(dict_, name, **kwargs):

    w = csv.writer(open(kwargs["directory"]+"/"+name+".csv", "w"))
    for key, val in dict_.items():
        w.writerow([key, val])

    f = open(kwargs["directory"]+"/"+name+".pkl", "wb")
    pickle.dump(dict_, f)
    f.close()

    return


def save_output(freq_set, mode_set, sim_output_set, **kwargs):

    if "parameter_name" not in kwargs:
        kwargs["parameter_name"] = "parameter_name_none"
        kwargs["parameter_set"] = [0]

    write_data_to_csv_file("frequencies",
                           [kwargs["parameter_name"], "freqs, real part"],
                           [kwargs["parameter_set"]]+[freq_set],
                           kwargs["directory"])

    save_dict(sim_output_set, "sim_output_set", **kwargs)

    return


def save_pickle(filename, x_set, directory="out"):

    with open(directory+"/"+filename, "wb") as fp:
        pickle.dump(x_set, fp)

    return


def load_pickle(filename, directory="out"):

    with open(directory+"/"+filename, "rb") as fp:
        x_set = pickle.load(fp)

    return x_set


# -----------------------------------------------------------------------------
# plots

def get_plot_geometry_and_harminv_sampling(**kwargs):

    sim = setup_simulation_2D_grid(**kwargs)

    sim.plot2D()
    sample_point_Ez = sample_point_offset_2D_grid(**kwargs)
    plt.scatter(sample_point_Ez[0], sample_point_Ez[1],
                color="b", marker=".", label="sample point Ez Harminv")
    plt.legend()
    plt.show()

    return


def plot_sample_point(sim, **kwargs):

    wire = sim.geometry[int(len(sim.geometry)/2)].center
    sample_point_Ez = sample_point_offset_2D_grid(**kwargs)
    sample_point_Hy = mp.Vector3(wire[0]+1.2*kwargs["wire_size"], wire[1])
    sample_point_Hx = mp.Vector3(wire[0], wire[1]+1.2*kwargs["wire_size"])
    plt.scatter(sample_point_Ez[0], sample_point_Ez[1],
                color="b", marker=".", label="sample point Ez")
    plt.scatter(sample_point_Hx[0], sample_point_Hx[1],
                color="r", marker=".", label="sample point Hx")
    plt.scatter(sample_point_Hy[0], sample_point_Hy[1],
                color="g", marker=".", label="sample point Hy")

    if check_if("get_steady_state_fields", **kwargs):
        sample_point_Ez_antenna = mp.Vector3(0,
                                             (2*kwargs["wire_size"]
                                              * (kwargs["wire_number_x"] % 2)))
        plt.scatter(sample_point_Ez_antenna[0], sample_point_Ez_antenna[1],
                    color="m", marker=".", label="sample point Ez antenna")

    plt.legend()

    return


def get_xlabel_from_parameter_name(**kwargs):

    xlabels = {"wire_number": "Number of wires per row",
               "resolution": "Resolution",
               "until_after_sources": r"$\Delta t$",
               "offset": "Offset",
               "absorber_width": "Material width",
               "defects_single_position": "Single defect at [i,j]",
               "defects_single_size": "Size of single defect",
               "defects_all_random": "Setup"
               }

    if kwargs["parameter_name"] in xlabels:
        return xlabels[kwargs["parameter_name"]]

    elif (kwargs["parameter_name"] == "wire_number_x"
    and kwargs["wire_number_x"] != kwargs["wire_number_y"]):
        return "Number of wire planes in x"
    else:
        return kwargs["parameter_name"]


def plot_mode_set(modes, **kwargs):

    if "result_name" in kwargs:
        kwargs["savefig"] = kwargs["result_name"]
    else:
        kwargs["savefig"] = "out/"

    decay_set = [modes[i][0].decay for i in range(len(modes))]
    Q_set = [modes[i][0].Q for i in range(len(modes))]
    amp_set = [modes[i][0].amp for i in range(len(modes))]
    err_set = [modes[i][0].err for i in range(len(modes))]

    plot_frequencies(modes, **kwargs)
    n_set = ["Decay", "Q", "Real amplitude", "Meep error"]
    x_set = [decay_set, Q_set, amp_set, err_set]

    for i in range(len(n_set)):
        plt.xlabel(get_xlabel_from_parameter_name(**kwargs))
        plt.ylabel(n_set[i])
        if kwargs["parameter_name"] == "defects_single_position":
            j_set = [j for j in range(len(kwargs["parameter_set"]))]
            plt.scatter(j_set[:len(modes)], x_set[i],
                        marker=".", color="k", label="Simulation")
            plt.xticks(j_set, labels=[str(j) for j in kwargs["parameter_set"]])
        else:
            plt.scatter(kwargs["parameter_set"][:len(modes)], x_set[i],
                        marker=".", color="k", label="Simulation")
        plt.legend()
        plt.savefig(kwargs["savefig"]+"__"+n_set[i], dpi=600)
        plt.close()

    return


def plot_frequencies(modes, **kwargs):

    freqs = [frequency_MP_to_SI(modes[i][0].freq, **kwargs)/1e9
             for i in range(len(modes))]
    x_set = (len(kwargs["parameter_set"])
             * [frequency_MP_to_SI(TM_nm0_WM_2D(**kwargs), **kwargs)/1e+9])

    if kwargs["parameter_name"] == "defects_single_position":
        i_set = [i for i in range(len(kwargs["parameter_set"]))]
        plt.scatter(i_set[:len(freqs)], freqs,
                    label="Simulation", marker=".", color="k")
        plt.plot(i_set, x_set, label=r"Analytical", color="b")
        plt.xticks(i_set, labels=[str(i) for i in kwargs["parameter_set"]])
    else:
        plt.scatter(kwargs["parameter_set"][:len(freqs)], freqs,
                    label="Simulation", marker=".", color="k")
        plt.plot(kwargs["parameter_set"], x_set, label=r"Analytical")

    plt.xlabel(get_xlabel_from_parameter_name(**kwargs))
    plt.ylabel(r"$TM_{110}$ [Ghz]")
    plt.legend()
    plt.savefig(kwargs["savefig"]+"__frequencies_lowest.png", dpi=600)
    plt.close()

    for i in range(len(modes)):
        for f in modes[i]:
            if kwargs["parameter_name"] == "defects_single_position":
                n_set = [i for i in range(len(kwargs["parameter_set"]))]
                plt.xticks(n_set,
                           labels=[str(i) for i in kwargs["parameter_set"]])
            else:
                n_set = kwargs["parameter_set"]
            plt.scatter(n_set[i], frequency_MP_to_SI(f.freq, **kwargs)/1e9,
                        marker=".", color="k")
            plt.plot(n_set, x_set)
    plt.scatter(n_set[i], frequency_MP_to_SI(f.freq, **kwargs)/1e9,
                label="Simulation", marker=".", color="k")
    plt.plot(n_set, x_set, label=r"Analytical", color="b")
    plt.xlabel(get_xlabel_from_parameter_name(**kwargs))
    plt.ylabel(r"$TM$ [Ghz]")
    plt.legend()
    plt.savefig(kwargs["savefig"]+"__frequencies_all.png", dpi=600)
    plt.close()

    return


def plot_sim_output_set(sim_output_set, **kwargs):

    for key in sim_output_set:
        if kwargs["parameter_name"] == "defects_single_position":
            j_set = [j for j in range(len(kwargs["parameter_set"]))]
            plt.scatter(j_set[:len(sim_output_set[key])],
                        np.real(sim_output_set[key]),
                        marker=".", color="k", label="Simulation")
            labels = [str(j) for j in
                      kwargs["parameter_set"][:len(sim_output_set[key])]]
            plt.xticks(j_set, labels=labels)
        else:
            plt.scatter(kwargs["parameter_set"][:len(sim_output_set[key])],
                        np.real(sim_output_set[key]), color="k", marker=".",
                        label="Simulation")
        plt.legend()
        plt.savefig("out/"+key+".png", dpi=600)
        plt.close()

    return


def colorbar_shift_vmax_vmin(x):

    max_ = np.max(x)
    min_ = np.min(x)
    if max_ > abs(min_):
        v_max = max_
        v_min = -max_
    else:
        v_max = -min_
        v_min = min_

    return v_max, v_min


def get_title_defects(**kwargs):

    A = ["defects_single_position", "defects_single_size"]
    B = ["defects_all_random"]

    if ("parameter_name" in kwargs and kwargs["parameter_name"] in A):

        if kwargs["defects_type"] == "single_wire":
            plt.title(str(kwargs["defects_position"]))

    elif ("parameter_name" in kwargs and kwargs["parameter_name"] in B):
        plt.title(r"Setup $"+str(kwargs["parameter"])+" $")

    return


def get_imshow_format(field, eps, **kwargs):

    if check_if("plot_accentuated_wires", **kwargs):
        plt.imshow(eps.transpose(), interpolation="None", cmap="gist_gray",
                   alpha=1, origin="lower")
        alpha = 0.8
    else:
        alpha = 1.
    vmax, vmin = colorbar_shift_vmax_vmin(field)
    plt.imshow(field.transpose(), interpolation="None", origin="lower",
               cmap="seismic", alpha=alpha, vmax=vmax, vmin=vmin, )
    plt.colorbar()
    get_ticks_format(field, **kwargs)

    return


def get_ticks_format(field, **kwargs):

    if kwargs["meep_unit"] == 1e-3:
        plt.xticks(np.arange(start=0, stop=(field.shape[0]+field.shape[0]
                                            / (2*kwargs["wire_number_x"])),
                             step=field.shape[0]/(2*kwargs["wire_number_x"])),
                   labels=np.arange(start=0,
                                    stop=(kwargs["wire_number_x"]
                                          * kwargs["wire_spacing_x"] +
                                          0.5*kwargs["wire_spacing_x"]),
                                    step=0.5*kwargs["wire_spacing_x"],
                                    dtype=int))
        plt.yticks(np.arange(start=0, stop=(field.shape[1]+field.shape[1]
                                            / (2*kwargs["wire_number_y"])),
                             step=field.shape[1]/(2*kwargs["wire_number_y"])),
                   labels=np.arange(start=0,
                                    stop=(kwargs["wire_number_y"]
                                          * kwargs["wire_spacing_y"] +
                                          0.5*kwargs["wire_spacing_y"]),
                                    step=0.5*kwargs["wire_spacing_y"],
                                    dtype=int))

        plt.xlabel(r"X [mm]")
        plt.ylabel(r"Y [mm]")

    else:
        plt.xticks(color="w", fontsize=0)
        plt.yticks(color="w", fontsize=0)
        plt.xlabel(r"X")
        plt.ylabel(r"Y")

    return


def get_plt_image(field, eps, plotname, s_set=["_Real_", "_Imag_", "_Phase_"],
                  **kwargs):

    i_set = ["_"]

    if check_if("plot_accentuated_wires", **kwargs):
        i_set.append("_Ac_wires")

    for s in s_set:
        cmap = "seismic"

        if s == "Real":
            s_field = np.real(field)

        elif s == "Imag":
            s_field = np.imag(field)

        elif s == "Phase":
            s_field = np.angle(field)
            cmap = "Purples"

        else:
            s_field = np.real(field)
            s = ""

        for i in i_set:
            if (i == "_Ac_wires" and check_if("plot_accentuated_wires",
                                              **kwargs)):
                plt.imshow(eps.transpose(), interpolation="None",
                           origin="lower", cmap="gist_gray", alpha=1)
                alpha = 0.8
            else:
                alpha = 1.
                i = ""

            vmax, vmin = colorbar_shift_vmax_vmin(s_field)
            plt.imshow(s_field.transpose(), interpolation="None",
                       origin="lower", cmap=cmap, alpha=alpha, vmax=vmax,
                       vmin=vmin)
            plt.colorbar()
            get_ticks_format(s_field, **kwargs)

            if check_if("get_defects", **kwargs):
                get_title_defects(**kwargs)

            plt.savefig(kwargs["directory"]+"/"+plotname+i+s+".png",
                        dpi=600)
            plt.close()

            # if s == "_Real_" and vmax > vmin:
            #     plt.imsave(kwargs["directory"]+"/"+plotname+s
            #                + "_no_colorbar.png", arr=s_field.transpose(),
            #                cmap=cmap, vmin=vmin, vmax=vmax)

    return
