#include "files_capture.h"
#include <sstream>
#include <string>
#include <iostream>
#include "utils.h"

namespace slh
{
	void FilesCapture::OnInit(int argc, char* argv[])
	{
		output_filelist_.open(argv[10]);
		LOG(INFO) << "result save to : " << argv[10];
		input_filelist_.open(argv[11]);
		LOG(INFO) << "image list load from : " << argv[11];
		LOAD_MODE_ = 1;

		if (argc == 13)
		{
			//url
			LOAD_MODE_ = 2;
			prefix_ = (argv[12]);

		}
		char buffer[288];
		LOG(INFO) << "read fileList...";
		while (!input_filelist_.eof())
		{

			input_filelist_.getline(buffer, 256);
			std::stringstream line(buffer);
			if (line.str().empty())
				break;
			std::string str;
			line >> str;
			int end_pos = LOAD_MODE_ == 1 ? str.size() : str.find(' ', 1);
			std::string path = prefix_ + str.substr(0, end_pos);
			file_list_.push_back(path);

		}
		LOG(ERROR) << "FilesCapture initialize success.";
		if (LOAD_MODE_ == 2)
			LOG(ERROR) << "Load files from URL.";
		else
			LOG(ERROR) << "Load files from Local.";
		LOG(ERROR) << "FilesCapture list size = " << file_list_.size();
	}

	void FilesCapture::OnInit(std::string& outputFile, std::string& inputFile, std::string& prefix)
	{
		output_filelist_.open(outputFile.c_str());
		LOG(INFO) << "result save to : " << outputFile;
		input_filelist_.open(inputFile.c_str());
		LOG(INFO) << "image list load from : " << inputFile;
		LOAD_MODE_ = 1;

		if (!prefix.empty())
		{
			//url
			LOAD_MODE_ = 2;
			prefix_ = (prefix);

		}
		char buffer[1000];
		LOG(INFO) << "read fileList...";
		while (!input_filelist_.eof())
		{

			input_filelist_.getline(buffer, 1000);
			std::stringstream line(buffer);
			if (line.str().empty())
				break;
			std::string str;
			line >> str;
			std::vector<std::string> elems;
			split(str, elems, ':');
			assert(elems.size() == 2);
			//int end_pos = LOAD_MODE_ == 1 ? elems[0].size() : elems[0].find(' ', 1);
			//std::string path = prefix_ + elems[0].substr(0, end_pos);
			std::string path = prefix_ + elems[0];
			file_list_.push_back(path);
			cat_name_.push_back(elems[1]);

		}
		LOG(ERROR) << "FilesCapture initialize success.";
		if (LOAD_MODE_ == 2)
			LOG(ERROR) << "Load files from URL.";
		else
			LOG(ERROR) << "Load files from Local.";
		LOG(ERROR) << "FilesCapture list size = " << file_list_.size();
	}

	void FilesCapture::WriteToFile(std::string& title, std::vector<float>& featureVec)
	{
		if (LOAD_MODE_ == 2)
			title = title.substr(title.find(prefix_) + prefix_.size());
		int feature_dim = featureVec.size();

		output_filelist_ << title.c_str() << "\t";
		for (size_t idx = 0; idx < feature_dim - 1; idx++) {
			output_filelist_ << idx << ":" << featureVec[idx] << "\t";
		}
		output_filelist_ << feature_dim - 1 << ":" << featureVec[feature_dim - 1] << "\n";
	}
	void FilesCapture::WriteToFile(std::string& title, int topk_num, std::vector<std::pair<std::string, int> >&result)
	{
		if (LOAD_MODE_ == 2)
			title = title.substr(title.find(prefix_) + prefix_.size());
		output_filelist_ << title.c_str() << " ";
		for (int j = 0; j < topk_num; j++){
			output_filelist_ << result[j].first.c_str() << ":" << result[j].second << " ";
		}
		output_filelist_ << "\n";
	}
	bool FilesCapture::IsEmpty()
	{
		return file_list_.empty();
	}

	FilesCapture& FilesCapture::operator>>(cv::Mat& img)
	{
		if (!IsEmpty())
		{
			std::string& name(file_list_.front());
			if (LOAD_MODE_ == 1)
			{
				//local
				img = cv::imread(name, -1);
			}
			else if (LOAD_MODE_ == 2)
			{
				//url
				if (urlToMat(name.c_str(), img) || !img.data || !INPUT_MAT_OK(img)) {
					assert("image load error");
				}
			}
			file_list_.pop_front();
		}
		return *this;
	}

	slh::FilesCapture& FilesCapture::operator>>(std::string& catName)
	{
		if (!IsEmpty())
		{
			catName = cat_name_.front();
			cat_name_.pop_front();
		}
		return *this;
	}

}
