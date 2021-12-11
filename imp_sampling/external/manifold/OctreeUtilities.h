/*
        Implemented 1) Octree Generation
            2) Octree based importance sampling
            3) Octree based inside/outside determination
        Weikai Chen
*/
#ifndef OctreeUtilities_H_
#define OctreeUtilities_H_

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "Octree.h"
// #include "BVH.h"
#include <map>
#include <Eigen/Core>
#include <cstdlib>
#include <igl/readOBJ.h>
#include <set>
#include <random>
using namespace std;
using namespace Eigen;
/*************************************************************************** 
    OBJ Loading 
 ***************************************************************************/

class Model_OBJ
{
public: 

    vector<set<int> > v_faces;

    Model_OBJ();
    int Load(string filename);    // Loads the model

    // get the octree leaf nodes 
    // pair<Vector3d, Vector3d> : pair<min_cornor of the cell, the 3D length of the cell>
    vector<pair<Vector3d, Vector3d>> getTreeCells(int depth, MatrixXd& cellCornerPts, vector<Vector4i> &tree_id);
    vector<pair<Vector3d, Vector3d>> getSmpTreeCells(int resolution, MatrixXd& cellCornerPts);
    // perform importance sampling on the obtained octree
    // [param] - cellRes: number of Octree child cells we want to create
    // [param] - targetSampleNum: target number of sampling points
    // return: a vector of 3D sampling points based on importance sampling around the surface
    vector<Vector3d> generateImpSamples(
        int maxDepth, int resolution, int numPerCell, float overlap, 
        vector<Vector4i> &tree_id , vector<pair<Vector3d, Vector3d>> &cells);
    void Calc_Bounding_Box();


    void Build_Tree(int depth);
    void Build_SmpTree(int resolution);
    
    vector<glm::dvec3> vertices, vertices_buf;
    vector<glm::dvec3> colors;
    vector<glm::ivec3> face_indices, face_indices_buf;
    
    glm::dvec3 min_corner, max_corner;
    Octree* tree;
    // BVH* bvh;
    // vector<BV*> bvs;
    string fn;
    // vector field
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

};

#endif
