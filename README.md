# PyLops at EAGE HPC Milan 2021

Material for **Leveraging GPUs for matrix-free optimization with PyLOps** event,
to be presented at [Fifth EAGE Workshop on High Performance Computing for Upstream](https://eage.eventsair.com/fifth-hpc-ws/).

The material contained in this repository has been used to produce the figures in the abstract and presentation:

- ``Timing.*``: timing of PyLops core operators, convolutional modelling and phase shift operators using arrays of type float64

- ``Timing-float32.*``: same as ``Timing.*`` using arrays type float32

- ``SeismicInversion.*``: timing of PyLops post- and pre-stack inversion using arrays of type float64

- ``SeismicInversion-float32.*``: same as ``SeismicInversion.*`` using arrays type float32

- ``SeismicTransforms.*``: timing of PyLops ChirpRadon3D and PhaseShift using arrays of type float64

- ``IBEX``: figures and log files for different runs

For all cases 3 files have been provided: 

- ``.ipynb``: jupyter notebook

- ``.py``: automatically generated python script from notebook

- ``.sh``: shell script to run the python script


Additionally, the ``PyVista_visualization.ipynb`` notebook is used to create 3D visualizations of seismic data.


Notebooks are run on a personal machine with Quadro RTX 4000 GPU, whilst all other computations are run in 
the [IBEX](https://www.hpc.kaust.edu.sa/ibex) Supercomputer at KAUST.
