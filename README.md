# Continuum mechanics and fluid-structure interaction problems

**Lecture materials for the course "Continuum mechanics and fluid-structure
interaction problems: mathematical modeling and numerical approximation"**

|  <!-- --> | <!-- --> |
| -- | -- |
| **Author** | Luca Heltai <luca.heltai@sissa.it> |
| **GitHub Repository:** | https://github.com/luca-heltai/fsi-suite |
| **GitHub Pages:** | https://luca-heltai.github.io/fsi-suite/ |
| **Licence:** | see the file [LICENCE.md](./LICENCE.md) |
## Course Introduction

Fluid-structure interaction (FSI) refers to the multiphysics coupling between
the laws that describe fluid dynamics and structural mechanics. 

This page collects the material that I have used for a course given at Kaust
(https://www.kaust.edu.sa/en) during the spring semester of 2022. 

The course covers three main topics: i) basics of continuum mechanics, ii)
mathematical modeling of FSI problems, and iii) numerical implementations based
on the finite element method. Theoretical lectures are backed up by applied
laboratories based on the C++ language, using example codes developed with the
open source finite element library deal.II (www.dealii.org). I cover basic
principles of continuum mechanics, discuss Lagrangian, Eulerian, and Arbitrary
Lagrangian-Eulerian formulations, and cover different coupling strategies, both
from the theoretical point of view, and with the aid of computational
laboratories.

The students will learn how to develop, use, and analyze state-of-the-art
finite element approximations for the solution of continuum mechanics problems,
including non-linear mechanics, computational fluid dynamics, and
fluid-structure-interaction problems.

A glimpse of the PDEs that are discussed in the course is given by the following 
graph 

@dotfile serial.dot width=100%

![](./doc/dot_files/serial.svg)

The laboratory part should enable a PhD student working on numerical analysis
of PDEs to implement state-of-the-art adaptive finite element codes for FSI
problems, that run in parallel, using modern C++ libraries. The implementation
are based on the `deal.II` library (www.dealii.org).

Main topics covered by these lectures:

- Advanced Finite Element theory
- How to use a modern C++ IDE, to build and debug your codes
- How to use a large FEM library to solve complex PDE problems
- How to properly document your code using Doxygen
- How to use a proper Git workflow to develop your applications
- How to leverage GitHub actions, google tests, and docker images to test and deploy your application
- How hybrid parallelisation (threads + MPI + GPU) works in real life FEM applications

Continuous Integration Status
-----------------------------

Up to date online documentation for the codes used in the laboratories is here: 

https://luca-heltai.github.io/fsi-suite/

| System |  Status |
| ------ | ------- | 
| **Continous Integration**  | [![GitHub CI](https://github.com/luca-heltai/fsi-suite/actions/workflows/tests.yml/badge.svg)](https://github.com/luca-heltai/fsi-suite/actions/workflows/tests.yml)   |
| **Docker** |  [![github-docker](https://github.com/luca-heltai/fsi-suite/actions/workflows/docker.yml/badge.svg)](https://github.com/luca-heltai/fsi-suite/actions/workflows/docker.yml) |
| **Doxygen**  | [![Doxygen](https://github.com/luca-heltai/fsi-suite/actions/workflows/doxygen.yml/badge.svg)](https://github.com/luca-heltai/fsi-suite/actions/workflows/doxygen.yml) |
| **Indent** | [![Indent](https://github.com/luca-heltai/fsi-suite/actions/workflows/indentation.yml/badge.svg)](https://github.com/luca-heltai/fsi-suite/actions/workflows/indentation.yml) |


## Useful links

One of my courses on theory and practice of finite elements:
- https://www.math.sissa.it/course/phd-course/theory-and-practice-finite-element-methods

Exceptional video lectures by Prof. Wolfgang Bangerth, that cover all the
things you will ever need for finite element programming.

- https://www.math.colostate.edu/~bangerth/videos.html

## Quick start

If you have Docker installed, you can run any of the programs available within
this repository by executing the following commands:

    wget https://raw.githubusercontent.com/luca-heltai/fsi-suite/main/fsi-suite.sh
    chmod +x ./fsi-suite.sh
    ./fsi-suite.sh

you should see the following output:

    Usage: ./fsi-suite.sh [-np N] program-name program-options

    Will run program-name with program-options, possibly via mpirun, passing -np N to mpirun.
    Here is a list of programs you can run:
    linear_elasticity.g	 mpi_stokes.g
    distributed_lagrange	mesh_handler		 poisson
    distributed_lagrange.g	mesh_handler.g		 poisson.g
    dof_plotter		mpi_linear_elasticity	 reduced_lagrange
    dof_plotter.g		mpi_linear_elasticity.g  reduced_lagrange.g
    fsi_test		mpi_poisson		 stokes
    fsi_test.g		mpi_poisson.g		 stokes.g
    linear_elasticity	mpi_stokes

    Programs ending with .g are compiled with debug symbols. To see help on how to run
    any of the programs, add a -h flag at the end.

The above command will download the latest docker image from the `main` branch
of this repository (which is built and uploaded to
https://hub.docker.com/r/heltai/fsi-suite at every commit to master), mount the
current directory, in a directory with the same path inside the container, and
the program you selected. For example, 

    ./fsi-suite.sh -np 4 mpi_poisson.g

will run the (debug version of the) executable based on the PDEs::MPI::Poisson
problem in parallel, using 4 processors. You can change grid, boundary
conditions, forcing terms, finite element spaces, etc. by editing the
configuration file which is created the first time you run the program, and then
passing it as an argument to the program itself, i.e., in the example above, the
parameter file that would be generated is `used_mpi_poisson.g_2d.prm`, which you
can edit and then pass as input to the program itself:

    ./fsi-suite.sh -np 4 mpi_poisson.g used_mpi_poisson.g_2d.prm

If you want to read some documentation of what each parameter does and how it
works, you can call the program passing a non-existing parameter file:

    ./fsi-suite.sh -np 4 mpi_poisson.g my_test.prm

    ----------------------------------------------------
    Exception on processing: 

    --------------------------------------------------------
    An error occurred in line <77> of file <../source/base/parameter_acceptor.cc> in function
        static void dealii::ParameterAcceptor::initialize(const string&, const string&, dealii::ParameterHandler::OutputStyle, dealii::ParameterHandler&, dealii::ParameterHandler::OutputStyle)
    The violated condition was: 
        false
    Additional information: 
        You specified <my_test.prm> as input parameter file, but it does not
        exist. We created it for you.

The created parameter file will also contain documentation for each parameter.
Running again the same command will now use the parameter file that was just
created: 

    ./fsi-suite.sh -np 4 mpi_poisson.g my_test.prm

    Number of cores         : 8
    Number of threads       : 2
    Number of MPI processes : 4
    MPI rank of this process: 0
    Cycle 0
    System setup
    Number of dofs 4
    Number of degrees of freedom: 4 (4)
    cells dofs  u_L2_norm   u_H1_norm  
        1    4 0.000e+00 - 0.000e+00 - 


    +---------------------------------------------+------------+------------+
    | Total CPU time elapsed since start          |      0.07s |            |
    |                                             |            |            |
    | Section                         | no. calls |  CPU time  | % of total |
    +---------------------------------+-----------+------------+------------+
    | assemble_system                 |         1 |  0.000425s |      0.61% |
    | estimate                        |         1 |  0.000434s |      0.62% |
    | output_results                  |         1 |   0.00591s |       8.4% |
    | setup_system                    |         1 |    0.0124s |        18% |
    | solve                           |         1 |   0.00547s |       7.8% |
    +---------------------------------+-----------+------------+------------+
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
