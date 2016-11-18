/************************************************************************/
/*
file	: file_capture.h
desc	: This class is to manage the operation of the network, initialize 
			the network, forward propagation, extract features.
author	: cheguangfu
date	: 2016-11-09
*/
/************************************************************************/

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include "singleton.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using boost::shared_ptr;
using std::vector;
namespace slh
{
	
	class NetManager :public Singleton<NetManager>
	{
	public:
		NetManager(){}
		~NetManager(){}
		void OnInit(const string& model_file, const string& trained_file, const string& mean_file, const int device_id);
		void ForwardAndGetfeature(cv::Mat &img, vector<float>& featrueVec, const char* feat_layer_name);
	private:
		void SetMean(const string& mean_file);
	
		void WrapInputLayer(vector<cv::Mat>* input_channels);
	
		void Preprocess(const cv::Mat& img,
			vector<cv::Mat>* input_channels);
	
	private:
		shared_ptr< Net<float> > net_;
		cv::Size input_geometry_;
		int num_channels_;
		cv::Mat mean_;
		int device_id_;
		std::vector<cv::Mat> input_channels;
	};
}


