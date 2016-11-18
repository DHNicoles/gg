/************************************************************************/
/* 
file	: file_search.h
desc	: This class is to manage the logic of the entire code, load the 
			configuration file, initialize the key components and so on.
author	: cheguangfu
date	: 2016-11-09
*/
/************************************************************************/
#ifndef __FASTSEARCH_H__
#define __FASTSEARCH_H__

#include <fstream>
#include <vector>
#include <map>
#include "singleton.h"
#include <string>
namespace slh
{
	class SLHSearchEngine;
	class FilesCapture;
	class NetManager;
	
	class FastSearch :public Singleton<FastSearch>
	{
	public:
		FastSearch();
		~FastSearch();

		void OnInit(const char* configFile);
		void OnDestory();
		/************************************************************************/
		/* run the whole projection                                             */
		/************************************************************************/
		void Run();
		void Query(const char* URL, const char* catName, int topK, std::vector<std::pair<std::string, int> >& result);
	private:
		/************************************************************************/
		/*initialize  catgory_  file_cap_ net_manager_ accroding to str+initStep
		initStep	:	0,1,2 catgory_->net_manager_->file_cap_
		str			:	initial param
		*/
		/************************************************************************/
		void InitMode(int initStep, std::string& str);
		/************************************************************************/
		/*function to create SLHSearchEngine
		*/
		/************************************************************************/
		SLHSearchEngine* CreateSLHSearchEngine(std::string& centerFile, std::string& modelFile, std::string& codesFile, std::string& cat_nam);
	private:
		std::map<std::string, SLHSearchEngine*> catgory_;				//map key:cat_name-->value:engine;
		FilesCapture* file_cap_;										//FilesCapture*
		NetManager* net_manager_;										//NetManager*
		static const char* extarct_feature_layer_name_;
		static const int cluster_num_ = 20;
		static const int feature_dim_ = 1024;
	};
}

#endif//__FASTSEARCH_H__
