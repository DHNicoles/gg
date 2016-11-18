#include "fast_search.h"
#include "SLHSearchEngine.h"
#include "files_capture.h"
#include "net_manager.h"
#include "utils.h"
#include "Stopwatch.hpp"

namespace slh
{
	const char* FastSearch::extarct_feature_layer_name_ = "pool5/7x7_s1";

	FastSearch::FastSearch() :
		file_cap_(NULL), net_manager_(NULL)
	{

	}

	FastSearch::~FastSearch()
	{
		OnDestory();
	}

	void FastSearch::OnInit(const char* configFile)
	{
		file_cap_ = new FilesCapture();
		net_manager_ = new NetManager();

		int initStep = -1;
		std::ifstream ifs(configFile);
		LOG(INFO) << "configing from " << configFile;
		char buffer[1000];
		while (!ifs.eof())
		{
			ifs.getline(buffer, 1000);
			std::stringstream line(buffer);
			if (line.str().empty())
				break;
			std::string str;
			line >> str;
			if (str.substr(0, 2) == "##")
				++initStep;
			else if (str.substr(0, 2) == "--")
				continue;
			else
				InitMode(initStep, str);
		}
		LOG(ERROR) << "initial success.";
	}

	void FastSearch::OnDestory()
	{
		if (file_cap_)
		{
			delete file_cap_;
		}
		file_cap_ = NULL;
		if (net_manager_)
		{
			delete net_manager_;
		}
		net_manager_ = NULL;

		std::map<std::string, SLHSearchEngine*>::iterator it = catgory_.begin();
		std::map<std::string, SLHSearchEngine*>::iterator endIt = catgory_.end();
		for (; it != endIt; ++it)
		{
			if (it->second)
				delete it->second;
		}
	}

	void FastSearch::Run()
	{
		int size = file_cap_->Instance()->size(), topk_num = 10, ist = 0;
		int num_bit_64 = 1024 / 64;
		vector<float> featureVec(feature_dim_, 0.0);
		std::vector<uint64_t> bincodes(num_bit_64);
		cv::Mat img;
		std::string catName;
		Stopwatch T0("0"),T1("2");
		T0.Reset();	T0.Start();
		while (!file_cap_->Instance()->IsEmpty())
		{
			std::string fileName = file_cap_->Instance()->GetImgName();
			*file_cap_ >> img >> catName;
			T1.Reset();	T1.Start();
			net_manager_->Instance()->ForwardAndGetfeature(img, featureVec, extarct_feature_layer_name_);
			int assign_cluster = catgory_[catName]->predict(featureVec.data());

			catgory_[catName]->encodeFeatures(featureVec.data(), 1, bincodes.data());
			std::vector<std::pair<std::string, int> > result;
			catgory_[catName]->searchEx(bincodes.data(), assign_cluster, topk_num, result);
			T1.Stop();
			LOG(INFO) << "#" << ist++ << "\tcost\t" << T1.GetTime() << "s." << std::endl;
			file_cap_->Instance()->WriteToFile(fileName, topk_num, result);
		}

		T0.Stop();
		LOG(INFO) << "total cost time is : " << T0.GetTime() << "s." << std::endl;
		LOG(INFO) << "average cost time is : " << T0.GetTime() / size << "s." << std::endl;
	}

	void FastSearch::InitMode(int initStep, std::string& str)
	{
		LOG(INFO)<<"initStep:"<<initStep<<"\ntext:"<<str;
		switch (initStep)
		{
		case 0://caffe init
		{
				   std::vector<std::string> elems;
				   split(str, elems, ':');
				   assert(elems.size() == 3 && net_manager_);
				   net_manager_->Instance()->OnInit(elems[0], elems[1], elems[2], -1);
		}
			break;
		case 1://output result init + input init
		{
				   std::vector<std::string> elems;
				   split(str, elems, ':');
				   if(elems.size()==4&&elems[2]=="http")
					{
						elems[2] += ":" + elems[3];
						elems.resize(3);
					}
				   assert((elems.size() == 3 || elems.size() == 2) && file_cap_);
				   if(elems.size()==2) elems.push_back("");
				   file_cap_->Instance()->OnInit(elems[0], elems[1], elems[2]);
		}
			break;
		case 2://catgory_ init
		{
				   std::vector<std::string> elems;
				   split(str, elems, ':');
				   assert(elems.size() == 4 && !catgory_.count(elems.front()));
				   LOG(INFO)<<"catgory name "<<elems[0]<<" data loading...";
				   catgory_[elems.front()] = CreateSLHSearchEngine(elems[1], elems[2], elems[3], elems[0]);
				   LOG(INFO)<<"catgory name "<<elems[0]<<" load done.";
		}
			break;
		default:
			break;
		}
	}

	SLHSearchEngine* FastSearch::CreateSLHSearchEngine(std::string& centerFile, std::string& modelFile, std::string& codesFile, std::string& cat_name)
	{

		SLHSearchEngine* engine = new SLHSearchEngine();
		engine->init(centerFile, modelFile, codesFile, cat_name, cluster_num_, feature_dim_);
		return engine;
	}
	void FastSearch::Query(const char* URL, const char* catName, int topK, std::vector<std::pair<std::string, int> >& result)
	{
		cv::Mat img;
		int num_bit_64 = 1024 / 64;
		std::vector<float> featureVec(feature_dim_, 0.0);
		std::vector<uint64_t> bincodes(num_bit_64);
		//url
		if (urlToMat(URL, img) || !img.data || !INPUT_MAT_OK(img)) {
			assert("image load error");
		}
		Stopwatch T1("1");
		T1.Reset();	T1.Start();
		//forward 
		net_manager_->Instance()->ForwardAndGetfeature(img, featureVec, extarct_feature_layer_name_);
		//search
		int assign_cluster = catgory_[catName]->predict(featureVec.data());

		catgory_[catName]->encodeFeatures(featureVec.data(), 1, bincodes.data());
		catgory_[catName]->searchEx(bincodes.data(), assign_cluster, topK, result);
		T1.Stop();
		LOG(INFO) << "Alg.cost\t" << T1.GetTime() << "s.";
		
	}
}

