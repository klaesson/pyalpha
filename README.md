# pyalpha
Simulations for the ALPHA dark matter detector. Python scripts using Meep. These will create a 2D slice of wire material, find its resonant modes, and access various field outputs.

The module funcs will do the Meep-related setup. Some examples of what you can do with it:
 - Enclose the wire material with walls (both lossy or perfect electric conductors) or use open boundary conditions.
 - Introduce different types of defects to the wire material.
 - Tune the resonant frequency of the wire material.
 - Collect and store simulation output.

## How to use
 1. Start by taking a look at how_to_use/1.how_to_set_up_a_2D_lattice.ipynb, it explains how to construct a simulation of a 2D lattice, find its resonant modes, and access various field outputs. After reading the notebook you should know how to use the basic script simulations/lattice_2D.py for an ideal grid.

 2. Learn how to add defects to the wire material, check out how_to_use/2.how_to_add_defects.ipynb

 3. Use the simulations/lattice_2D_delta_parameter.py when you want to simulate more than one setup. It uses `get_freqs_2D_grid_delta_parameter()` to organize various outputs that you get from modifying one simulation parameter. 
For example, you can use it to test convergence for resolution, runtime, or to change the number of wires in a grid.
You can also use it to tune the grid or gradually increase the size of a defect (I've included a ready-made version for tuning: simulations/lattice_2D_tuning_wires.py) or create a new parameter by modifying the appropriate function in the funcs module.

## Misc
 - In how_to_use/how_to_use_meep_waveguide.ipynb you'll find a small intro on how to set up simulations in Meep.
 - In misc/ there's an old script for looking at the TM-band structure of an infinite 2D grid.
 - There are two .yml files with the Conda environments that I've used. mp_p is for the normal version of Meep and pmp for Meeps parallel version.
