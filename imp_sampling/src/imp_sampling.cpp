/*
    1) Generate samples based on importance sampling
    2) Also generate the samples' signed distance (IGL winding number) to the input mesh
    For each mesh, the output is saved to a sdf file
    Weikai Chen
*/
/*
    1) Generate samples based on importance sampling
    2) Also generate the samples' signed distance (IGL winding number) to the input mesh
    For each mesh, the output is saved to a sdf file
    Weikai Chen
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "OctreeUtilities.h"
// #include "computeDistField.h"
#include <igl/signed_distance.h>
#include <igl/read_triangle_mesh.h>
using namespace std;


class SamplesGenerator
{
private:

    std::vector<Eigen::Vector4i> tree_id;
	std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> cells;
    std::vector<Eigen::Vector3d> samples;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

public:

    SamplesGenerator(string objName, int numPerCell, int maxDepth, int resolution, float overlap);
    Eigen::MatrixXi getTreeNodes();
    Eigen::MatrixXd getTreeCells();
    Eigen::MatrixXd getSamplePoints();
};

// ----------------
// regular C++ code
// ----------------

SamplesGenerator::SamplesGenerator(string objName, int numPerCell, int maxDepth, int resolution, float overlap)
{
    // generate samples
    Model_OBJ obj;
    obj.Load(objName);
    samples = obj.generateImpSamples(maxDepth, resolution, numPerCell, overlap, tree_id, cells);
    igl::readOBJ(objName,V,F);
}


Eigen::MatrixXi SamplesGenerator::getTreeNodes()
{
    Eigen::MatrixXi nodes(tree_id.size(), 4);
    for(int i=0; i < tree_id.size(); ++i)
    {
        nodes.row(i) = tree_id[i].transpose();
    }
    return nodes;
}


Eigen::MatrixXd SamplesGenerator::getTreeCells()
{
    Eigen::MatrixXd center_scale(cells.size(), 6);
    for(int i=0; i < cells.size(); ++i)
    {
        Eigen::Vector3d minP = cells[i].first;
        Eigen::Vector3d length = cells[i].second;
        center_scale(i,0) = minP[0] + length[0] / 2;
        center_scale(i,1) = minP[1] + length[1] / 2;
        center_scale(i,2) = minP[2] + length[2] / 2;
        center_scale(i,3) = length[0];
        center_scale(i,4) = length[1];
        center_scale(i,5) = length[2];
    }
    return center_scale;
}


Eigen::MatrixXd SamplesGenerator::getSamplePoints()
{
    Eigen::MatrixXd sample3D(samples.size(), 3);
    for(int i=0; i < samples.size(); ++i)
    {
        sample3D.row(i) = samples[i].transpose();
    }
    // load obj mesh
    Eigen::VectorXd S,B;
    Eigen::VectorXi I;
    Eigen::MatrixXd C,N;
    igl::signed_distance(sample3D,V,F,igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER,S,I,C,N);
    Eigen::MatrixXd sdf(samples.size(),5);
    for(int i=0; i < samples.size(); ++i)
    {
        int outside = (S(i) > 0);
        sdf(i,0) = samples[i][0];
        sdf(i,1) = samples[i][1];
        sdf(i,2) = samples[i][2];
        sdf(i,3) = outside;
        sdf(i,4) = S(i);
    }
    return sdf;
}
// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(imp_sampling,m)
{
    m.doc() = "pybind11 example plugin";

    py::class_<SamplesGenerator>(m, "SamplesGenerator")
    .def(py::init<string, int, int, int, float>())
    .def("getTreeNodes", &SamplesGenerator::getTreeNodes)
    .def("getTreeCells", &SamplesGenerator::getTreeCells)
    .def("getSamplePoints", &SamplesGenerator::getSamplePoints)
    .def("__repr__",
        [](const SamplesGenerator &a) {
            return "<imp_sampling.SamplesGenerator>";
        }
    );
}