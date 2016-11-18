#include "utils.h"
#include "files_capture.h"
#include "net_manager.h"

#include <time.h>
#include <iostream>

/************************************************************************/
/*  local	12
./test															0
center_file = "/export/anshan/slh/centers/woman_centers";		1
model_path ="/export/anshan/slh/model/woman_0818" ;				2
codes_path = "/export/anshan/data/woman";						3
cat_name = "woman";												4
cluster_num = 20;												5
feature_dim = 1024;												6
//../npx_model/deploy.prototxt									7
//../npx_model/trained_model.caffemodel							8
//../npx_model/train_mean.binaryproto							9
//output.txt(4)													10
//list.txt(5)                                                   11      */
/************************************************************************/

/************************************************************************/
/*  url		13
./test														0
center_file = "/export/anshan/slh/centers/woman_centers";		1
model_path ="/export/anshan/slh/model/woman_0818" ;				2
codes_path = "/export/anshan/data/woman";						3
cat_name = "woman";												4
cluster_num = 20;												5
feature_dim = 1024;												6
deploy.prototxt													7
trained_model.caffemodel										8
train_mean.binaryproto											9
output.txt														10
list.txt														11
prefix															12		*/
/************************************************************************/
namespace slh
{

	int StringToInt(const char* s)
	{
		int res = 0;
		const char* ptr = s;
		while (*ptr != '\0') res = res * 10 + *(ptr++) - '0';
		return res;
	}
	void parseOrDie(int argc, char *argv[], SLHSearchEngine& engine, FilesCapture& fileMagager, NetManager& netManager)
	{
		if (argc<12 || argc>13) exit(-1);
		//exit(-1);
		LOG(INFO) << "search engine initializing...";
		engine.init(argv[1], argv[2], argv[3], argv[4], StringToInt(argv[5]), StringToInt(argv[6]));
		LOG(INFO) << "search engine initial success.";
		LOG(INFO) << "CNN initializing...";
		netManager.Instance()->OnInit(argv[7], argv[8], argv[9], -1);
		LOG(INFO) << "CNN initial success.";
		LOG(INFO) << "filecapture initializing...";
		fileMagager.Instance()->OnInit(argc, argv);
		LOG(INFO) << "filecapture initial success.";
	}
	std::string IntToString(int x)
	{
		assert(x>=0);
		string str = x == 0 ? "0" : "";
		while (x)
		{
			int r = x % 10;
			x = x / 10;
			str += '0' + r;
		}
		int begin = 0, end = str.size() - 1;
		//reverse
		for (; begin < end; ++begin, --end)
		{
			str[begin] ^= str[end];
			str[end] ^= str[begin];
			str[begin] ^= str[end];
		}
		return str;
	}
	size_t write_data(void *ptr, size_t size, size_t nmemb, void * data) {
		size_t sz = size * nmemb;
		struct MemoryStruct * mem = (struct MemoryStruct *) data;
		mem->memory = (char *)realloc(mem->memory, mem->size + sz + 1);

		if (mem->memory == NULL) {
			/* out of memory! */
			LOG(ERROR) << "not enough memory (realloc returned NULL)";
			return 0;
		}
		memcpy(&(mem->memory[mem->size]), ptr, sz);
		mem->size += sz;
		mem->memory[mem->size] = 0;
		return sz;
	}

	void split(std::string& str, std::vector<std::string>& stringArr, char c)
	{
		std::string elem;
		for (size_t i = 0; i <= str.size(); ++i)
		{
			if (i == str.size() && !elem.empty())
				stringArr.push_back(elem);
			else if (str[i] == c)
			{
				if (!elem.empty())	stringArr.push_back(elem);
				elem.clear();
			}
			else
				elem += str[i];
		}
	}

	int urlToMat(const char * url, cv::Mat & outArr) {
		CURL *curl;
		CURLcode res;
		curl_global_init(CURL_GLOBAL_ALL);
		int retCode(0);
		curl = curl_easy_init();
		if (curl) {
			clock_t st = clock();
			struct MemoryStruct chunk;
			chunk.memory = (char *)malloc(1);  /* will be grown as needed by the realloc above */
			chunk.size = 0;
			curl_easy_setopt(curl, CURLOPT_URL, url);
			curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
			curl_easy_setopt(curl, CURLOPT_WRITEDATA, &chunk);
			res = curl_easy_perform(curl);
			if (CURLE_OK != res) {
				LOG(ERROR) << "curl_easy_perform() failed" << std::endl;
				retCode = 1;
			}
			else {
				GetImgFromStream(chunk.memory, chunk.size, outArr);
			}
			curl_easy_cleanup(curl);
			free(chunk.memory);
			clock_t end = clock();
			LOG(INFO) << "Download(s): " << (end - st) / (double)CLOCKS_PER_SEC << std::endl;
		}
		return retCode;
	}
	int GetImgFromStream(const char *pBuffer, long nLength, cv::Mat &mat_img)
	{
		//get img from stream
		cv::Mat mat_buffer(1, static_cast<int>(nLength), CV_8UC1, const_cast<char*>(pBuffer));
		//std::cout << "mat_buffer\n" << mat_buffer << std::endl;
		try {
			mat_img = imdecode(mat_buffer, 1);
			//std::cout << "imdecode\n" << mat_img << std::endl;
		}
		catch (std::exception e) {
			LOG(ERROR) << e.what() << std::endl;
			return 1;
		}
		if (NULL == mat_img.data)
		{
			return 1;
		}
		return 0;
	}
}

