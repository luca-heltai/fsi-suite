// Gmsh project created on Mon Feb 12 10:59:16 2024

theta = 2*Pi/4;
R = 1.0;
lc = .2;

// Intermediate angles
N = 2.0;
theta1 = (2.0*Pi-theta)/N*1+theta;
// theta2 = (2*Pi-theta)/N*2+theta;

// center
Point(1) = {0, 0, 0, lc};

// start point
Point(2) = {R*Cos(theta), R*Sin(theta), 0, lc};
Point(3) = {R*Cos(theta1), R*Sin(theta1), 0, lc};
// Point(4) = {R*Cos(theta2), R*Sin(theta2), 0, lc};

// end point
Point(5) = {R, 0, 0, lc};
// straight lines
Line(1) = {1, 2};
Line(2) = {1, 5};

// circle arcs
Circle(3) = {2, 1, 3};
// Circle(4) = {3, 1, 4};
// Circle(5) = {4, 1, 5};
Circle(4) = {3, 1, 5};

// plane surface
// Curve Loop(1) = {3, 4, 5, -2, 1};
Curve Loop(1) = {3, 4, -2, 1};
Plane Surface(1) = {1};

// define ids
Physical Curve("BoundaryId: 1", 1) = {2, 1};
Physical Curve("BoundaryId: 2, ManifoldId: 1", 2) = {3, 4};
Physical Surface("Material Id: 0", 1) = {1};

// Mesh.Algorithm = 8; 
// // Quad grids
// Transfinite Surface {1};
// Recombine Surface {1};