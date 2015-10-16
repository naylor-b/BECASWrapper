# BECASWrapper

# Overview

The `BECASWrapper` package provides a Python interface between the cross-sectional structure tool [BECAS](http://www.becas.dtu.dk) and [FUSED-Wind](http://fusedwind.org) which is built using [OpenMDAO](http://openmdao.org) as the underlying framework.

BECAS, the BEam Cross section Analysis Software, determines cross section stiffness properties using a finite element based approach. BECAS handles arbitrary cross section geometries, any number of arbitrarily oriented anisotropic materials, and correctly accounts for all geometrical and material induced couplings (e.g. bend-twist coupling).

BECAS is a licenced software, available free of charge for academic use, and available for commercial use for a small annual fee. For more information, see  http://www.becas.dtu.dk.

This Python module facilitates using BECAS for evaluating blade structure beam properties based on a geometric definition of a blade given in the format defined in FUSED-Wind, and furthermore using the tool in an aero-structural blade optimization context when interfaced with with an aeroelastic code, e.g. HAWC2 or FAST.

## Dependencies and Installation

In addition to BECAS itself the BECAS wrapper has the following requirements:

* Python 2.7.x
* numpy, scipy, sphinx
* Oct2Py (if you choose to use the Octave bridge)
* OpenMDAO 1.x
* FUSED-Wind > 0.3

You can either get hold of the module by downloading it as a zip archive or clone the repository:

    git clone http://github.com/DTUWindEnergy/BECASWrapper.git

The module is installable as a standard Python package:

    $ cd BECASWrapper
    $ python setup.py develop

or using pip:

    $ cd BECASWrapper
    $ pip install -e .

## Documentation and Tests

The documentation is written in Sphinx and can be built this way:

    $ cd docs
    $ make html

To view it open _build/html/index.html in a browser.

To ensure that you have BECAS and the BECASWrapper properly installed, run the tests in `becas_wrapper/test`.
You can find the examples accompanying the documentation in the `becas_wrapper/examples` directory.
