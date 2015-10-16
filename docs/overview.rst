
========
Overview
========

The `BECASWrapper` package provides a Python interface between the cross-sectional structure tool `BECAS <http://www.becas.dtu.dk>`_ and `FUSED-Wind <http://fusedwind.org>`_ which is built using `OpenMDAO <http://openmdao.org>`_ as the underlying framework.

BECAS, the BEam Cross section Analysis Software, determines cross section stiffness properties using a finite element based approach. BECAS handles arbitrary cross section geometries, any number of arbitrarily oriented anisotropic materials, and correctly accounts for all geometrical and material induced couplings (e.g. bend-twist coupling).

BECAS is a licenced software, available free of charge for academic use, and available for commercial use for a small annual fee. For more information, see  http://www.becas.dtu.dk.

This Python module facilitates using BECAS for evaluating blade structure beam properties based on a geometric definition of a blade given in the format defined in FUSED-Wind, and furthermore using the tool in an aerostructural blade optimization context when interfaced with with an aeroelastic code, e.g. HAWC2 of FAST.
