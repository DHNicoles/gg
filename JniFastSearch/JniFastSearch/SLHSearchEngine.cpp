#include "SLHSearchEngine.h"
#include <glog/logging.h>
namespace slh
{
	
	pthread_mutex_t SLHSearchEngine::mutex_lock = PTHREAD_MUTEX_INITIALIZER;
	
	SLHSearchEngine::SLHSearchEngine()
	{
		cluster_num_ = 20;
		feature_dim_ = 1024;
	}
	
	SLHSearchEngine::~SLHSearchEngine()
	{
		for (map<int, vector<uint64_t*> >::iterator it = codes_.begin();
	            it != codes_.end(); it++)
	    {
	        for (size_t i = 0; i < it->second.size(); i++)
	            delete[] it->second[i];
	    }
	}
	
	bool SLHSearchEngine::init(string center_file, string model_path, string codes_path, 
							   string cat_name, int cluster_num, int feature_dim)
	{
		cluster_num_ = cluster_num;
		feature_dim_ = feature_dim;
	
		if (!loadCenters(center_file.c_str()))
			return false;
		if (!initEncode(model_path))
			return false;
		if (!loadNamesAndCodes(codes_path, cat_name))
			return false;
	
		return true;
	}
	
	int SLHSearchEngine::predict(float *feature)
	{
		float **centers = new float *[cluster_num_];
		for(int i = 0; i < cluster_num_; i++)
		{
			centers[i] = new float [feature_dim_];
		}
		for(int i = 0; i < cluster_num_; i++)
			for(int j = 0; j < feature_dim_; j++)
	             centers[i][j] = centers_[i][j];
		
		int index =computeNearest(feature, centers, cluster_num_, feature_dim_);
	
		for(int i = 0; i < cluster_num_; i++)
		{
			delete [] centers[i];
		}
		delete [] centers;
	
		return index;
	}
	
	bool SLHSearchEngine::search(uint64_t *bincodes, int assign_cluster, int topk_num, 
								 vector<pair<string,int> > &result)
	{
		map<int, vector<string> >::iterator name_iter;
		name_iter = names_.find(assign_cluster);
	    if (name_iter == names_.end()){
	        //printf("Can't find names for class %d\n", assign_cluster);
	        LOG(ERROR)<<"Can't find names for class "<< assign_cluster;
	        return false;
	    }
		map<int, vector<uint64_t*> >::iterator code_iter;
		code_iter = codes_.find(assign_cluster);
	    if (code_iter == codes_.end()){
	        //printf("Can't find codes for class %d\n", assign_cluster);
	        LOG(ERROR)<<"Can't find codes for class "<< assign_cluster;
	        return false;
	    }
	
		int db_num = name_iter->second.size();
		int num_bit_64 = num_bits_/64;
		typedef pair<int, int> dist_pair;
	    vector<dist_pair> ep;
		for (int i = 0; i < db_num; i++)
		{
			int hammingdist = 0;
			for(int j = 0; j < num_bit_64; j++)
				hammingdist += (int)_mm_popcnt_u64(code_iter->second[i][j] ^ bincodes[j]);
			ep.push_back(make_pair(hammingdist, i));
		}  
		partial_sort(ep.begin(), ep.begin() + topk_num, ep.end());
	    for(int i = 0; i < topk_num; i++)
		{
	       result.push_back(make_pair(name_iter->second[ep[i].second], ep[i].first));
		}
		return true;
	}
	
	bool SLHSearchEngine::loadCenters(const char *centerfile)
	{
		FILE *fp = fopen(centerfile, "r");
		if(fp == NULL)
			return false;
		float value = 0.0;
		for(int i = 0; i < cluster_num_; i++)
		{
			vector<float> single_center;
			for(int j = 0; j < feature_dim_; j++)
			{
				fscanf(fp, "%f", &value);
				single_center.push_back(value);
			}
			centers_.push_back(single_center);
		}
		fclose(fp);
		return true;
	}
	
	bool SLHSearchEngine::initEncode(string &model_path)
	{
		string mean_file_name = model_path + "/samplemean.txt";
		string rotation_file_name = model_path + "/R.txt";
		string center_file_name = model_path + "/centers_table.txt";
	
		if(loadMat(center_file_name.c_str(), centers_table_) == -1)
			return false;
		//printf("Load training parameters: centers_table ready!\n");
	        LOG(INFO)<<"Load training parameters : centers_table ready!";
	
		if(loadMat(mean_file_name.c_str(), mean_mat_) == -1)
			return false;
		//printf("Load training parameters: mean_mat ready!\n");
	        LOG(INFO)<<"Load training parameters : mean_nat ready!";
	
		if(loadMat(rotation_file_name.c_str(), rotation_mat_) == -1)
			return false;
		//printf("Load training parameters: rotation_mat ready!\n");
	        LOG(INFO)<<"Load training parameters : rotation_mat ready!";
	
		num_bits_ = 1024;
		num_bits_subspace_ = 2;
	
		return true;
	}
	
	bool SLHSearchEngine::loadNamesAndCodes(string codes_path, string cat_name)
	{
		for(int i = 0; i < cluster_num_; i++)
		{
			LOG(INFO)<<"loadNamesAndCodes for cluster "<<i;
			string namefile = codes_path + "/" + cat_name + "_names_" + int2string(i);
			string codefile = codes_path + "/" + cat_name + "_codes_" + int2string(i);
			vector<string> name_vec;
			bool sign1 = loadNames(namefile.c_str(), name_vec);
			if(!sign1){
				LOG(ERROR)<<"Load namefile failed : "<<namefile.c_str();
				return false;
			}
			names_.insert(pair<int, vector<string> >(i, name_vec));
	
			int num = name_vec.size();
			vector<uint64_t*> code_vec;
			bool sign2 = loadCodes(codefile.c_str(), code_vec, num);
			if(!sign2){
				//printf("Load codefile %s failed\n", codefile.c_str());
				LOG(ERROR)<<"Load codefile failed : "<<codefile.c_str();
				return false;
			}
			codes_.insert(pair<int,  vector<uint64_t*> >(i, code_vec));
		}
		return true;
	}
	
	int SLHSearchEngine::loadMat(const char *filename, vector<MatrixXf> &matvec)
	{
		FILE *fp = fopen(filename, "r");
		if(fp == NULL)
			return -1;
		float value = 0.0;
		int n1 = 0;
		int n2 = 0; 
		int n3 = 0;
		fscanf(fp, "%d", &n1);
		fscanf(fp, "%d", &n2);
		fscanf(fp, "%d", &n3);
		matvec.reserve(n1);
		for(int i = 0; i < n1; i++){
			MatrixXf mat = MatrixXf::Zero(n2, n3);
			for(int j = 0; j < n2; j++){
				for(int k = 0; k < n3; k++){
					fscanf(fp, "%f", &value);
					mat(j,k) = value;
				}
			}
			matvec.push_back(mat);
		}
		return 0;
	}
	
	int SLHSearchEngine::loadMat(const char *filename, MatrixXf &mat)
	{
		FILE *fp = fopen(filename, "r");
		if (fp == NULL){
			return -1;
		}
		float value = 0.0;
		int n1 = 0;
		int n2 = 0; 
		fscanf(fp, "%d", &n1);
		fscanf(fp, "%d", &n2);
		mat = MatrixXf::Zero(n1, n2);
		for(int i = 0; i < n1; i++){
			for(int j = 0; j < n2; j++){
				fscanf(fp, "%f", &value);
				mat(i,j) = value;
			}
		}
		return 0;
	}
	
	string SLHSearchEngine::int2string(const int num)
	{
		std::stringstream ss;
		ss<<num;
		std::string str_num = ss.str();
		return str_num;
	}
	
	bool SLHSearchEngine::loadNames(const char *filename, vector<string> &name_vec)
	{
		std::ifstream ifs;
		ifs.open(filename);
		if(!ifs)
			return false;
		string line;
		while(getline(ifs, line)) 
		{
			name_vec.push_back(line);
		}
		return true;
	}
	
	bool SLHSearchEngine::loadCodes(const char *filename, vector<uint64_t*> &code_vec, int db_num)
	{
		int num_bit_64 = 1024/64;
		FILE *fp = fopen(filename, "r");
		if(fp == NULL)
			return false;
		for(int i = 0; i < db_num; i++)
		{
			uint64_t *code = new uint64_t[num_bit_64];
			for(int j = 0; j < num_bit_64; j++)
			{
				uint64_t value = 0;
				fscanf(fp, "%lu", &value);
				code[j] = value;
			}
			code_vec.push_back(code);
		}
		fclose(fp);
		return true;
	}
	
	int SLHSearchEngine::computeNearest(float *feature, float **centers, int center_num, int feature_dim)
	{
		float min_dist = DBL_MAX;
		int label = 0;
		for(int i = 0; i < center_num; i++)
		{
			float *singlecenter = new float[feature_dim];
			for(int j = 0; j < feature_dim; j++)
				singlecenter[j] = centers[i][j];
			float dist = computeL2(feature, singlecenter, feature_dim);
			if (dist < min_dist)
			{
				min_dist = dist;
				label = i;
			}
			delete []singlecenter;
		}
		return label;
	}
	
	float SLHSearchEngine::computeL2(float *feature, float *singlecenter, int feature_dim)
	{
		float dist = 0.0;
		for(int i = 0; i < feature_dim; i++)
		{
			float temp = feature[i] - singlecenter[i];
			dist += pow(temp, 2);
		}
		return dist;
	}
	
	int SLHSearchEngine::encodeFeatures(float *feature, int feature_num, uint64_t *bincodes)
	{
		if(feature == NULL || bincodes == NULL) return -1;
		MatrixXf featmat = MatrixXf::Zero(feature_num, feature_dim_);
		int feat_index = 0;
		for (int i = 0; i < feature_num; i++){
			for (int j = 0; j < feature_dim_; j++){
				featmat(i, j) = feature[feat_index];
				feat_index++;
			}
		}
	
		MatrixXf projmat = MatrixXf::Zero(feature_num, feature_dim_);
		this->projectFeatures(featmat, projmat, mean_mat_, rotation_mat_);
	
		int num_subspaces = num_bits_ / num_bits_subspace_;
		int d = feature_dim_ / num_subspaces;
		MatrixXi codeMat = MatrixXi::Zero(feature_num, num_bits_);
		for(int i = 0; i < num_subspaces; i++)
		{
			MatrixXf subFeat = projmat.block(0, i * d, feature_num, d).transpose();
			MatrixXf center = centers_table_[i].transpose();
			MatrixXf distFeat;
			this->squareDist(subFeat, center, distFeat);
			MatrixXi idxcenters = MatrixXi::Zero(feature_num, 1);
			for(int j = 0; j < feature_num; j++)
			{
				float min = DBL_MAX; 
				int idx = 0;
				for(int k = 0; k < distFeat.cols(); k++)
				{
					if(distFeat(j, k) < min)
					{
						min = distFeat(j, k);
						idx = k;
					}
				}
				int b1 = 0; int b2 = 0;
				if(idx == 0) {
					b1 = 0; b2 = 0;
				}
				else if(idx == 1) {
					b1 = 1; b2 = 0;
				}
				else if(idx == 2) {
					b1 = 0; b2 = 1;
				}
				else if(idx == 3) {
					b1 = 1; b2 = 1;
				}
				codeMat(j, i * 2) = b1;
				codeMat(j, i * 2 + 1) = b2;
			}
		}
	
		MatrixXi codeByteMat = MatrixXi::Zero(feature_num, num_bits_/8);
		for (int i = 1; i < (num_bits_ + 1); i++)
		{
			int w = (int) ceil((double)i / 8);
			for (int j = 0; j < feature_num; j++)
			{
				int t = (i - 1) % 8;
				if (codeMat(j, i - 1) == 1)
					codeByteMat(j, w - 1) |= (0x01 << t);
			}
		}
	
		int code_index = 0;
		for (int i = 0; i < feature_num; i++)
		{
			for(int j = 0; j < num_bits_/8; j+=8)
			{
				uint64_t value = 
					((uint64_t)codeByteMat(i, j) << 56) + ((uint64_t)codeByteMat(i, j+1) << 48)
					+ ((uint64_t)codeByteMat(i, j+2) << 40) + ((uint64_t)codeByteMat(i, j+3) << 32)
					+ ((uint64_t)codeByteMat(i, j+4) << 24) + ((uint64_t)codeByteMat(i, j+5) << 16) 
					+ ((uint64_t)codeByteMat(i, j+6) << 8) + (uint64_t)codeByteMat(i, j+7);
				bincodes[code_index] = value;
				code_index++;
			}
		}
		return 0;
	}
	
	void SLHSearchEngine::projectFeatures(MatrixXf &featmat, MatrixXf &projfeat, MatrixXf &mean_mat, 
										  MatrixXf &rotation_mat)
	{
		for (int i = 0; i < featmat.cols(); i++)
			for(int j = 0; j < featmat.rows(); j++)
				featmat(j, i) -= mean_mat(0, i);
		projfeat = featmat * rotation_mat;
	}
	
	void SLHSearchEngine::squareDist(MatrixXf &subFeat, MatrixXf &center, MatrixXf &distFeat)
	{
		MatrixXf dotcenter = MatrixXf::Zero(center.rows(), center.cols());
		MatrixXf sumcenter = MatrixXf::Zero(1, center.cols());
		for(int i = 0; i < center.rows(); i++)
			for(int j = 0; j < center.cols(); j++)
				dotcenter(i, j) = center(i, j) * center(i, j);
	
		for(int j = 0; j < center.cols(); j++)
			for(int i = 0; i < center.rows(); i++)
				sumcenter(0, j) += dotcenter(i, j);
	
		MatrixXf temp = -2 * subFeat.transpose() * center;
		MatrixXf dtemp = MatrixXf::Zero(temp.rows(), temp.cols());
		for(int i = 0; i < temp.rows(); i++)
			for(int j = 0; j < temp.cols(); j++)
				dtemp(i, j) = temp(i, j) + sumcenter(0, j);
	
		MatrixXf dotsubFeat = MatrixXf::Zero(subFeat.rows(), subFeat.cols());
		MatrixXf sumsubFeat = MatrixXf::Zero(1, subFeat.cols());
		for(int i = 0; i < subFeat.rows(); i++)
			for(int j = 0; j < subFeat.cols(); j++)
				dotsubFeat(i, j) = subFeat(i, j) * subFeat(i, j);
	
		for(int j = 0; j < subFeat.cols(); j++)
			for(int i = 0; i < subFeat.rows(); i++)
				sumsubFeat(0, j) += dotsubFeat(i, j);
	
		MatrixXf dtemp_trans = dtemp.transpose();
		MatrixXf dtemp_plus = MatrixXf::Zero(dtemp_trans.rows(), dtemp_trans.cols());
		for(int i = 0; i < dtemp_trans.rows(); i++)
			for(int j = 0; j < dtemp_trans.cols(); j++)
				dtemp_plus(i, j) = dtemp_trans(i, j) + sumsubFeat(0, j);
	
		distFeat = dtemp_plus.cwiseAbs().transpose();
	}
	
	bool SLHSearchEngine::searchEx(uint64_t *bincodes, int assign_cluster, int topk_num, vector<pair<string, int> >& result)
	{
		map<int, vector<string> >::iterator name_iter;
		name_iter = names_.find(assign_cluster);
		if(name_iter == names_.end()){
			printf("Cannot find names for class %d\n",
				assign_cluster);
			return false;
		}
	
		map<int, vector<uint64_t*> >::iterator code_iter;
		code_iter = codes_.find(assign_cluster);
		if(code_iter == codes_.end()){
			printf("Cannot find codes for class %d\n",
				assign_cluster);
			return false;
		}
	
		vector<pair<int,int> > Nresult;
		int db_num = name_iter->second.size();
		int num_bit_64 = num_bits_ / 64;
	        const int ThreadNum = 15;
		int range = db_num / ThreadNum, rest = db_num % ThreadNum;;
	        vector<pthread_t> pidarr(ThreadNum + (rest!=0) );
		for(int i=0;i<ThreadNum;++i){
			ThreadParam* tpm = new ThreadParam;
			tpm->topk_num = topk_num;
			tpm->dim = num_bit_64;
			tpm->rangeIndex = make_pair(i*range, (i + 1)*range);
			tpm->bincodes = bincodes;
			tpm->ptr_codes = &(code_iter->second);
			tpm->pNresult = &Nresult;
			int r = pthread_create(&pidarr[i], NULL, searchThread, (void*)tpm);
			if(r!=0){
				printf("thread cread false! exit.\n");
				exit(-1);
			}
		}
		if(rest){
			ThreadParam* tpm = new ThreadParam;
			tpm->topk_num = topk_num;
			tpm->dim = num_bit_64;
			tpm->rangeIndex = make_pair(db_num-rest, db_num);
			tpm->bincodes = bincodes;
			tpm->ptr_codes = &(code_iter->second);
			tpm->pNresult = &Nresult;
			int r = pthread_create(&pidarr[ThreadNum], NULL, searchThread, (void*)tpm);
			if(r!=0){
				printf("thread cread false! exit.\n");
				exit(-1);
			}
		}
		for(int i=0;i<pidarr.size();++i)
			pthread_join(pidarr[i], NULL); 
		partial_sort(Nresult.begin(), Nresult.begin()+topk_num, Nresult.end());
		for(int i = 0; i < topk_num; i++)
		{
		       result.push_back(make_pair(name_iter->second[Nresult[i].second], Nresult[i].first));
		}
		return true;
	}
		
	void* SLHSearchEngine::searchThread(void* param)
	{
		ThreadParam* tpm = (ThreadParam*)param;
		int firstIndex = tpm->rangeIndex.first;
		int lastIndex = tpm->rangeIndex.second;
		vector<uint64_t *>* p_codes = tpm->ptr_codes;
		uint64_t *bincodes = tpm->bincodes;
		int topk_num= tpm->topk_num;
		int dim = tpm->dim;
		vector<pair<int, int> >*  pNresult = tpm->pNresult;
		delete tpm;
	
		multimap<int, int> dist;
		for (int i = firstIndex; i < lastIndex; ++i)
		{
			int hammingdist = 0;
			for (int j = 0; j < dim; j++)
				hammingdist += (int)_mm_popcnt_u64((*p_codes)[i][j] ^ bincodes[j]);
			if (dist.size() < topk_num)
			{
				dist.insert(make_pair(hammingdist, i));
			}
			else if (hammingdist < dist.rbegin()->first)
			{
				dist.erase(--dist.end());
				dist.insert(make_pair(hammingdist, i));
			}
		}
		pthread_mutex_lock(&mutex_lock);
		pNresult->insert(pNresult->end(), dist.begin(), dist.end());
		pthread_mutex_unlock(&mutex_lock);
		return NULL;
	}
}
