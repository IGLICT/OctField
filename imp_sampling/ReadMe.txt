-- Compilation (only tested in Ubuntu)
1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./generateSamplesWind 

-- Algorithm
The code will generate 3D samples with importance sampling that focuses around the mesh surface -- that means sampling more points around the surface and less points as moving further away fromt the surface.

The idea of the code is 1) generate Octree that containing the input mesh. 2) when the Octree cell intersects with the surface, we keep subdivide it; 3) iterate the step 2) until the target cell number is achieved. Hence, in the end, the code will generate a large amount of small cells around the surface but a few large cells in the void space.
4) We uniformly distribute samples into each cell. Say we want to generate 10000 samples and we created 300 Octree cells. Then each cell contains floor(10000/300) = 33 samples. For the remaining 100 (10000 - 300*33) samples, we distribute tehm to the first 100 cells.

-- Output format
The code will output the samples in a file ending with ".sdf" format, which can be opened in a common text editor.
Line 1 of output -- number of sampling points
For rest of lines, from left to right: 

x y z (coordinate) inside/outside_flag signed_distance

inside/outside_flag: 1 is outside; 0 is inside
