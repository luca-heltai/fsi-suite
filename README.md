# Continuum mechanics and fluid-structure interaction problems
## mathematical modeling and numerical approximation

Fluid-structure interaction (FSI) refers to the multiphysics coupling between
the laws that describe fluid dynamics and structural mechanics. This course
covers three main topics: i) basics of continuum mechanics, ii) mathematical
modeling of FSI problems, and iii) numerical implementations based on the
finite element method. Theoretical lectures are backed up by applied
laboratories based on the C++ language, using example codes developed with the
open source finite element library deal.II (www.dealii.org). I cover basic
principles of continuum mechanics, discuss Lagrangian, Eulerian, and Arbitrary
Lagrangian-Eulerian formulations, and cover different coupling strategies, both
from the theoretical point of view, and with the aid of computational
laboratories.

The students will learn how to develop, use, and analyze state-of-the-art
finite element approximations for the solution of continuum mechanics problems,
including non-linear mechanics, computational fluid dynamic, and
fluid-structure-interaction problems.

The laboratory part should enable a PhD student working on numerical analysis
of PDEs to implement a state-of-the-art adaptive finite element codes for FSI
problems, that run in parallel, using modern C++ libraries. The implementation
will be based on the `deal.II` library (www.dealii.org).

What you will learn:
- Advanced Finite Element theory
- How to use a modern C++ IDE, to build and debug your codes
- How to use a large FEM library to solve complex PDE problems
- How to properly document your code using Doxygen
- How to use a proper Git workflow to develop your applications
- How to leverage GitHub actions, google tests, and docker images to test and deploy your application
- How hybrid parallelisation (threads + MPI + GPU) works in real life FEM applications

Continuous Integration Status:
-----------------------------

| System | Status                                                                                                                                                                                                                                           |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | 
| Indent | [![Doxygen](https://github.com/luca-heltai/fsi-suite/actions/workflows/doxygen.yml/badge.svg)](https://github.com/luca-heltai/fsi-suite/actions/workflows/doxygen.yml)|
| Doxygen  | [![Doxygen](https://github.com/luca-heltai/fsi-suite/actions/workflows/doxygen.yml/badge.svg)](https://github.com/luca-heltai/fsi-suite/actions/workflows/doxygen.yml) |
| Continous Integration  | [![GitHub CI](https://github.com/luca-heltai/fsi-suite/actions/workflows/tests.yml/badge.svg)](https://github.com/luca-heltai/fsi-suite/actions/workflows/tests.yml)   |


## Useful links

One of my courses on theory and practice of finite elements:
- https://www.math.sissa.it/course/phd-course/theory-and-practice-finite-element-methods

## Course program

A tentative detailed program is shown below 
(this will be updated during the course to reflect the actual course content)

1. Course introduction. 
    - Motivating examples
    - Basic principles and background knowledge.

2. Recap on Finite Element Methods
    - Introduction to deal.II.

3. Basic principles of continuum mechanics 
    - conservation equations 
    - constitutive equations
    - kinematics
    - transport theorems

3. Lagrangian formulation of continuum mechanics (solids).
    - Static and compressible linear elasticity

4. Recap on mixed problems 
    - Static incompressible linear elasticity

5. Eulerian formulation of continuum mechanics
    - Stokes problem

6. Recap on time discretization schemes
    - Time dependent Stokes problem
    - Time dependent linear elasticity (wave equations)
    
7. Treating non-linearities 
    - IMEX
    - predictor corrector
    - fixed point
    - Newton
    
8. Navier-Stokes equations

9. Segregated coupling
    - Lagrangian domain deformation

10. Arbitrary Lagrangian Eulerian formulation

11. Immersed Boundary/Immersed Finite Element Method

12. Distributed Lagrange multiplier formulation

13. Students' projects.

