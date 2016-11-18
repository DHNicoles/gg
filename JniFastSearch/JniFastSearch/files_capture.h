/************************************************************************/
/*
file	: file_capture.h
desc	: This class is to manage file operations, including input files, output files, 
			read the file, save the file, etc.
author	: cheguangfu
date	: 2016-11-09
*/
/************************************************************************/

#ifndef __LOAD_FILE_H__
#define __LOAD_FILE_H__
#include <vector>
#include <list>
#include <string>
#include <fstream>
#include "singleton.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
/************************************************************************/
/* load images from URL or Local store.                                                                     */
/************************************************************************/
namespace slh
{
	class FilesCapture : public Singleton<FilesCapture>
	{
	public:
		FilesCapture(){}
		~FilesCapture()
		{
			OnDestory();
			if (input_filelist_.is_open()) input_filelist_.close();
			if (output_filelist_.is_open()) output_filelist_.close();
		}
		/************************************************************************/
		/* initial function ,继续解析后面的参数，初始化成员                                                                  */
		/************************************************************************/
		void OnInit(int argc, char* argv[]);
		void OnInit(std::string& outputFile, std::string& inputFile, std::string& prefix);
		void OnDestory(){}
		/************************************************************************/
		/* 将结果写入文件中，title 文件名，featureVec特征向量                                                                     */
		/************************************************************************/
		void WriteToFile(std::string& title, std::vector<float>& featureVec);
		void WriteToFile(std::string& title, int topk_num, std::vector<std::pair<std::string, int> >&result);
		std::string& GetImgName(){ return file_list_.front(); }
		/************************************************************************/
		/* override >> to get images by order                                                          */
		/************************************************************************/
		FilesCapture& operator>>(cv::Mat& img);
		/************************************************************************/
		/* override >> to get images catgory by order                                                         */
		/************************************************************************/
		FilesCapture& operator>>(std::string& catName);
		bool IsEmpty();
		size_t size(){ return file_list_.size(); }
	private:
		std::ofstream output_filelist_;
		std::ifstream input_filelist_;
		int LOAD_MODE_; //1->local \ 2->url
		std::list<std::string> file_list_;
		std::list<std::string> cat_name_;
		std::string prefix_;
	};
}

#endif//__LOAD_FILE_H__
