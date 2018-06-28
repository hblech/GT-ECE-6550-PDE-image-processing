## Super resolution project

This project aims to implement a fast and versatile super-resolution solution using partial differential equations.

For this project, the low resolution images are generated from a known image.
This allows us to compare more efficiently the performance of parameter sets, and to define the parameters best for the task.

The regularization selected here is an adaptive regularization: the heat equation is dominant on smooth regions, and TV regularization is dominant on edges.

An efficient implementation was made possible by the [Pythran project](https://github.com/serge-sans-paille/pythran). See notebooks for more details. To make configuration easier, it is advised to use a virtual environment. Execute the `setup.sh` script to compile the utility modules and increase greatly the execution speed.
