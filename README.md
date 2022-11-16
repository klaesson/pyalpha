# pyalpha
Code for modeling a 2D slice of wire material. Python scripts using Meep. These will create a 2D slice of wire material, find its resonant modes, and access various field outputs.

The user can choose geometry and source parameters, as well as different output options, by using the scripts [lattice_2D.py](simulations/lattice_2D.py) and [lattice_2D_delta_parameter.py](simulations/lattice_2D_delta_parameter.py). These will then call the module [funcs.py](simulations/funcs.py) that does all of the Meep-related setups.
Some examples of what you can do with it:
 - Enclose the wire material with walls (both lossy or perfect electric conductors) or use open boundary conditions.
 - Introduce different types of defects to the wire material.
 - Tune the resonant frequency of the wire material.
 - Collect and store simulation output.

## How to use

 1. Start by taking a look at the notebook [1.how_to_set_up_a_2D_lattice.ipynb](how_to_use/1.how_to_set_up_a_2D_lattice.ipynb), it explains how to choose the parameters for a simulation of a 2D lattice, and the various field outputs that can be accessed through the funcs module. After reading the notebook you should know how to use the basic script [lattice_2D.py](simulations/lattice_2D.py) for an ideal grid.

 2. Learn how to add defects to the wire material, check out [2.how_to_add_defects.ipynb](how_to_use/2.how_to_add_defects.ipynb).

 3. Use the script [lattice_2D_delta_parameter.py](simulations/lattice_2D_delta_parameter.py) when you want to simulate more than one setup. It uses `get_freqs_2D_grid_delta_parameter()` to organize various outputs that you get from modifying one parameter. 
For example, you can use it to test convergence for resolution, runtime, or to change the number of wires in a grid.
You can also use it to tune the grid or gradually increase the size of a defect (I've included a ready-made version for tuning: [lattice_2D_tuning.py](simulations/lattice_2D_tuning.py) or create a new parameter by modifying the appropriate function in the funcs module.

## Additional files
 - In [how_to_use_meep_waveguide.ipynb](how_to_use/how_to_use_meep_waveguide.ipynb) you'll find a small intro on how to set up simulations directly in Meep.
 - I have iuncluded an old file that you can use to get the TM-band structure of an infinite 2D grid, [infinite_square_grid.py](misc/infinite_square_grid.py).
 - There are two .yml files with the Conda environments that I've used. [mp_p.yml]([mp_p.yml) is for the normal version of Meep and [pmp.yml]([pmp.yml) for Meeps parallel version.
