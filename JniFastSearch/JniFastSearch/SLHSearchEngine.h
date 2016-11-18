/************************************************************************/
/*
file	: file_search.h
desc	: This class is the engine for the entire search. 
			Its implementation is based on SLH hash search
author	: anshan
date	: 2016-10-xx
*/
/************************************************************************/

#ifndef __SLHSEARCHENGINE_H__
#define __SLHSEARCHENGINE_H__
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <float.h>
#include <nmmintrin.h>
#include <stdint.h>
#include <string>
#include <fstream>
#include <map>
#include <pthread.h>
#include <Eigen/Dense>
using Eigen::MatrixXf;
using Eigen::MatrixXi;
using namespace std;


namespace slh
{
	struct ThreadParam
	{
		int topk_num,dim;
		pair<int, int> rangeIndex;
		uint64_t *bincodes;
		vector<uint64_t*>* ptr_codes;
		vector<string>* ptr_names;
		vector<pair<int, int> >* pNresult;
	};
	
	class SLHSearchEngine
	{
	protected:
	    static void* searchThread(void* param);
		
	    static pthread_mutex_t mutex_lock;
	public:
	    bool searchEx(uint64_t *bincodes, int assign_cluster, int topk_num, vector<pair<string,int> > &result);
	public:
	    SLHSearchEngine();
	    ~SLHSearchEngine();
		bool init(string center_file, string model_path, string codes_path, string cat_name, int cluster_num, int feature_dim);
		int predict(float *feature);
		int encodeFeatures(float *feature, int feature_num, uint64_t *bincodes);
		bool search(uint64_t *bincodes, int assign_cluster, int topk_num, vector<pair<string,int> > &result);
	
	private:
	        bool loadCenters(const char *centerfile);
		bool initEncode(string &model_path);
		bool loadNamesAndCodes(string codes_path, string cat_name);
		bool loadNames(const char *filename, vector<string> &name_vec);
	    bool loadCodes(const char *filename, vector<uint64_t*> &code_vec, int db_num);
	    int loadMat(const char *filename, vector<MatrixXf> &matvec);
	    int loadMat(const char *filename, MatrixXf &mat);
		string int2string(const int num);
		int computeNearest(float *feature, float **centers, int center_num, int feature_dim);
		float computeL2(float *feature, float *singlecenter, int feature_dim);
	    void projectFeatures(MatrixXf &featmat, MatrixXf &projfeat, MatrixXf &mean_mat, 
										  MatrixXf &rotation_mat);
		void squareDist(MatrixXf &subFeat, MatrixXf &center, MatrixXf &distFeat);
	
		vector<vector<float> > centers_;
		map<int, vector<string> > names_;
		map<int, vector<uint64_t*> > codes_;
		int cluster_num_;
		int feature_dim_;
		int num_bits_;
		int num_bits_subspace_;
		vector<MatrixXf> centers_table_;
		MatrixXf mean_mat_;
		MatrixXf rotation_mat_;
	};
}

//pthread_mutex_t SLHSearchEngine::mutex_lock;

#endif //__SLHSEARCHENGINE_H__
