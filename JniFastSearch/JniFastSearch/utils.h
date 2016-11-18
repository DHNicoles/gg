#ifndef __UTILS_H__
#define __UTILS_H__

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <glog/logging.h>
#include <curl/multi.h>
#include "SLHSearchEngine.h"
#include <string>

#define INPUT_MAT_OK(mat)  (mat.rows>10&&mat.cols>10)

namespace slh
{
	class FilesCapture;
	class NetManager;
	
	struct MemoryStruct {
		char *memory;
		size_t size;
	};
	/************************************************************************/
	/* string to int                                                        */
	/************************************************************************/
	int StringToInt(const char* s);
	/************************************************************************/
	/* ¿¿¿¿¿¿¿¿¿¿¿¿  ¿¿?FilesCapture, NetManager                                                        */
	/************************************************************************/
	/************************************************************************/
	/*  int to string                                                        */
	/************************************************************************/
	std::string IntToString(int x);
	void parseOrDie(int argc, char *argv[], SLHSearchEngine&, FilesCapture&, NetManager&);
	
	/************************************************************************/
	/* ¿url¿¿¿¿¿¿¿¿cv::Mat¿¿                                                        */
	/************************************************************************/
	int urlToMat(const char * url, cv::Mat & outArr);
	/************************************************************************/
	/* ¿urlToMat¿¿¿¿                                                                     */
	/************************************************************************/
	int GetImgFromStream(const char *pBuffer, long nLength, cv::Mat &mat_img);
	size_t write_data(void *ptr, size_t size, size_t nmemb, void * data);
	/************************************************************************/
	/* split into vector by char                                                                     */
	/************************************************************************/
	void split(std::string& str, std::vector<std::string>& stringArr,char c);
}

#endif//__UTILS_H__

