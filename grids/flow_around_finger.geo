// Point density: large number = let gmsh decide?
cl = 1e22;

// Cell density. Used to control how many points to insert per unit
cd = 1;

// Dimensions of flow domain
LL = 10;
H1 = 8;
H2 = 6;

// Dimensions of solid finger
L = 1;
H = 4;

// Bottom part of the domain
Point(1) = {-LL/2, 0, 0, cl};
Point(2) = { -L/2, 0, 0, cl};
Point(3) = { L/2, 0, 0, cl};
Point(4) = { LL/2,  0, 0, cl};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};



Extrude {0, H, 0} {
    Curve{1}; Curve{2}; Curve{3}; Layers {cd*Ceil(H)}; 
}

// Now compute the new points positions:
dh = H2-H;
m = (H1-H2)/(LL+L);

d1 = (LL-L)/2 * m + dh;
d2 = (LL+L)/2 * m + dh;
d3 = LL * m + dh;

Extrude {0, dh, 0} {
    Point{10}; Layers {Ceil(H2-H)*cd}; 
}


Extrude {0, d1, 0} {
    Point{8}; Layers {Ceil(H2-H)*cd}; 
}

Extrude {0, d2, 0} {
    Point{6}; Layers {Ceil(H2-H)*cd}; 
}

Extrude {0, d3, 0} {
    Point{5}; Layers {Ceil(H2-H)*cd}; 
}

Line(20) = {14, 13};
Line(21) = {13, 12};
Line(22) = {12, 11};


Curve Loop(1) = {19, 20, -18, -4};
Plane Surface(16) = {1};
Curve Loop(2) = {18, 21, -17, -8};
Plane Surface(17) = {2};
Curve Loop(3) = {17, 22, -16, -12};
Plane Surface(18) = {3};

Transfinite Curve {2, 8, 21} = Ceil(L)*cd Using Progression 1;
Transfinite Curve {1, 4, 20, 3, 12, 22} = Ceil((LL-L)/2)*cd Using Progression 1;
Transfinite Curve {5, 6, 10, 14} = Ceil(H)*cd Using Progression 1;

Transfinite Surface {:} Right;

Recombine Surface {:};

// ------------------------------------------------------------
// Material IDs
// ------------------------------------------------------------

// Finger material
Physical Surface("MaterialID: 1", 1) = {11};

// Flow material
Physical Surface("MaterialID: 0", 2) = {7,15,16,17,18};

// ------------------------------------------------------------
// Boundary and Manifold IDs
// ------------------------------------------------------------

// Bottom of finger
Physical Curve("BoundaryID: 1", 3) = {2};

// Bottom of flow domain
Physical Curve("BoundaryID: 2", 4) = {1, 3};

// Side surfaces (no normal flux boundary condition)
Physical Curve("BoundaryID: 3", 5) = {5, 14, 16, 19};

// Top flow surface (free surface boundary condition)
Physical Curve("BoundaryID: 4", 6) = {20, 21, 22};

// The surface Gamma, between the solid and the fluid
Physical Curve("ManifoldID: 1, BoundaryID:-1", 7) = {6, 8, 10};


// Mesh.Algorithm = 8; // Delaunay for quads

// Mesh.Smoothing = 100;//+