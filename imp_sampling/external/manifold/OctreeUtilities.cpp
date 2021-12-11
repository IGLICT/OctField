#include "OctreeUtilities.h"
#include <queue>
#define ITER_NUM 20
int g_sharp = 0;
Model_OBJ::Model_OBJ()
{
}

int Model_OBJ::Load(string filename)
{
    using namespace Eigen;
    using namespace std;
    // Load a mesh in OBJ format
    igl::readOBJ(filename, V, F);
    fn = filename;
    // Make the example deterministic
    srand(0);
    vertices.resize(V.rows());
    face_indices.resize(F.rows());
    for (int i = 0; i < V.rows(); ++i)
        vertices[i] = glm::dvec3(V(i,0),V(i,1),V(i,2));
    for (int i = 0; i < F.rows(); ++i) {
        face_indices[i] = glm::ivec3(F(i,0),F(i,1),F(i,2));
	}
	// cout << "Loaded mesh " << filename << endl;
	return 0;
}

void Model_OBJ::Calc_Bounding_Box()
{
	min_corner = glm::dvec3(1e30,1e30,1e30);
	max_corner = -min_corner;
	for (int i = 0; i < (int)vertices.size(); ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			if (vertices[i][j] < min_corner[j])
			{
				min_corner[j] = vertices[i][j];
			}
			if (vertices[i][j] > max_corner[j])
			{
				max_corner[j] = vertices[i][j];
			}
		}
	}
	glm::dvec3 length = max_corner - min_corner;
	// min_corner -= length * 0.2;
	// max_corner += length * 0.2;
	// originally the value is 0.2, for generating SDF samples, we set to 
	min_corner -= length * 0.05;
	max_corner += length * 0.05;
}

void Model_OBJ::Build_Tree(int depth)
{
	// cout << "building tree ...";
	Calc_Bounding_Box();
	// tree = new Octree(min_corner, max_corner, face_indices, 0.0001);
	tree = new Octree(min_corner, max_corner, face_indices, 0.0);
	while (tree->level < depth)
	{
		tree->Split(vertices);
	}

	tree->BuildConnection();
	tree->BuildExteriorEmptyConnection();

	list<Octree*> empty_list;
	set<Octree*> empty_set;
	for (int i = 0; i < 6; ++i)
	{
		tree->ExpandEmpty(empty_list, empty_set, i);
	}

	while ((int)empty_list.size() > 0)
	{
		Octree* empty = empty_list.front();
		empty->exterior = 1;
		for (list<Octree*>::iterator it = empty->empty_neighbors.begin();
			it != empty->empty_neighbors.end(); ++it)
		{
			if (empty_set.find(*it) == empty_set.end())
			{
				empty_list.push_back(*it);
				empty_set.insert(*it);
			}
		}
		empty_list.pop_front();
	}
	// cout << "Done" << endl;
}


void Model_OBJ::Build_SmpTree(int resolution)
{
	// cout << "building smptree ...";
	Calc_Bounding_Box();
	while (tree->number < resolution)
	{
		tree->Split(vertices);
	}

	tree->BuildConnection();
	tree->BuildExteriorEmptyConnection();

	list<Octree*> empty_list;
	set<Octree*> empty_set;
	for (int i = 0; i < 6; ++i)
	{
		tree->ExpandEmpty(empty_list, empty_set, i);
	}

	while ((int)empty_list.size() > 0)
	{
		Octree* empty = empty_list.front();
		empty->exterior = 1;
		for (list<Octree*>::iterator it = empty->empty_neighbors.begin();
			it != empty->empty_neighbors.end(); ++it)
		{
			if (empty_set.find(*it) == empty_set.end())
			{
				empty_list.push_back(*it);
				empty_set.insert(*it);
			}
		}
		empty_list.pop_front();
	}
	// cout << "Done" << endl;
}


vector<pair<Vector3d, Vector3d>> Model_OBJ::getTreeCells(int depth, MatrixXd& cellCornerPts, vector<Vector4i> &tree_id)
{
	vertices_buf = vertices;
	face_indices_buf = face_indices;
//	Build_BVH();
	Build_Tree(depth);
	vector<pair<Vector3d, Vector3d>> output;
	int count=0;
	tree->traverseOccupiedCells(output, count, tree_id, 0, -1);

	// cout << output.size() << " " << count << endl;
	MatrixXd GV(output.size()*8, 3);
	double minLength = 10000;
	for(int i=0; i<output.size(); i++)
	{
		Vector3d minP = output[i].first;
		Vector3d len = output[i].second;
		if (len[0] < minLength)
			minLength = len[0];
		double x = len[0], y = len[1], z = len[2];
		GV.row(8*i) = minP.transpose();
		GV.row(8*i + 1) = (minP + Vector3d(x, 0, 0)).transpose();
		GV.row(8*i + 2) = (minP + Vector3d(0, y, 0)).transpose();
		GV.row(8*i + 3) = (minP + Vector3d(0, 0, z)).transpose();
		GV.row(8*i + 4) = (minP + Vector3d(x, y, 0)).transpose();
		GV.row(8*i + 5) = (minP + Vector3d(x, 0, z)).transpose();
		GV.row(8*i + 6) = (minP + Vector3d(0, y, z)).transpose();
		GV.row(8*i + 7) = (minP + Vector3d(x, y, z)).transpose();
	}
	cellCornerPts = GV;
	// cout << "minLength of cells: " << minLength << endl;
	return output;
}


vector<pair<Vector3d, Vector3d>> Model_OBJ::getSmpTreeCells(int resolution, MatrixXd& cellCornerPts)
{
	vertices_buf = vertices;
	face_indices_buf = face_indices;
//	Build_BVH();
	Build_SmpTree(resolution);
	vector<pair<Vector3d, Vector3d>> output;
	int count=0;
	tree->traverseBreadth(output, count);

	// cout << output.size() << " " << count << endl;
	MatrixXd GV(output.size()*8, 3);
	double minLength = 10000;
	for(int i=0; i<output.size(); i++)
	{
		Vector3d minP = output[i].first;
		Vector3d len = output[i].second;
		if (len[0] < minLength)
			minLength = len[0];
		double x = len[0], y = len[1], z = len[2];
		GV.row(8*i) = minP.transpose();
		GV.row(8*i + 1) = (minP + Vector3d(x, 0, 0)).transpose();
		GV.row(8*i + 2) = (minP + Vector3d(0, y, 0)).transpose();
		GV.row(8*i + 3) = (minP + Vector3d(0, 0, z)).transpose();
		GV.row(8*i + 4) = (minP + Vector3d(x, y, 0)).transpose();
		GV.row(8*i + 5) = (minP + Vector3d(x, 0, z)).transpose();
		GV.row(8*i + 6) = (minP + Vector3d(0, y, z)).transpose();
		GV.row(8*i + 7) = (minP + Vector3d(x, y, z)).transpose();
	}
	cellCornerPts = GV;
	// cout << "minLength of cells: " << minLength << endl;
	return output;
}


// perform importance sampling on the obtained octree
vector<Vector3d> Model_OBJ::generateImpSamples(
	int maxDepth, int resolution, int numPerCell, float overlap, 
	vector<Vector4i> &tree_id , vector<pair<Vector3d, Vector3d>> &cells)
{
	MatrixXd cellCorners;
	cells = getTreeCells(maxDepth, cellCorners, tree_id);
	vector<pair<Vector3d, Vector3d>> smpcells;
	MatrixXd smpcellCorners;
	smpcells = getSmpTreeCells(resolution, smpcellCorners);
	// int numPerCell = numPerCell / cells.size();
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1); //uniform distribution between 0 and 1
	vector<Vector3d> output;
	// cout << "Max depth: " << maxDepth
	// 		<< " number of cells: " << cells.size()
	// 		<< " number per cell: " << numPerCell 
	// 		<< " round up total: " << numPerCell * cells.size() << endl;
	for(int k=0; k < cells.size(); k++)
	{
		vector<int> smpcell_idx;
		Vector3d min_corner = cells[k].first;
		Vector3d length = cells[k].second;
		for(int i=0; i < smpcells.size(); i++)
		{
			Vector3d smp_min_corner = smpcells[i].first;
			Vector3d smp_length = smpcells[i].second;
			if ((min_corner[0]-length[0]*overlap-0.0001<=smp_min_corner[0]) && 
				(min_corner[1]-length[1]*overlap-0.0001<=smp_min_corner[1]) &&
				(min_corner[2]-length[2]*overlap-0.0001<=smp_min_corner[2]) &&
				(smp_min_corner[0]+smp_length[0]<=min_corner[0]+length[0]*(1.+overlap)+0.0001) &&
				(smp_min_corner[1]+smp_length[1]<=min_corner[1]+length[1]*(1.+overlap)+0.0001) &&
				(smp_min_corner[2]+smp_length[2]<=min_corner[2]+length[2]*(1.+overlap)+0.0001))
				{
					smpcell_idx.push_back(i);
				}
		}
		assert(smpcell_idx.size()>0);
		int numPerSmpcell = numPerCell / smpcell_idx.size();
		int res = numPerCell - numPerSmpcell * smpcell_idx.size();
		for(int i=0; i < smpcell_idx.size(); i++)
		{
			int idx = smpcell_idx[i];
			min_corner = smpcells[idx].first;
			length = smpcells[idx].second;
			int tarnum = numPerSmpcell + (int)(res > i);
			for(int j = 0; j < tarnum; j++)
			{
				double x = dis(gen), y = dis(gen), z = dis(gen);
				// cout << x << " " << y << " " << z << endl;
				Vector3d sample = min_corner + length.cwiseProduct(Vector3d(x,y,z));
				output.push_back(sample);
			}
		}
	}
	tree->~Octree();
	return output;
}
