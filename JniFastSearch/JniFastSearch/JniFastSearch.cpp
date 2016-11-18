#include "JniFastSearch.h"
#include "fast_search.h"
#include "utils.h"
#include <vector>
#include <string>
#include <iostream>
/*****************************
 *global var
 * */
slh::FastSearch fs;

JNIEXPORT void JNICALL Java_JniFastSearch_OnInit(JNIEnv *env, jclass args, jstring configPath)	
{
	const char *config_name = (env)->GetStringUTFChars(configPath, 0);
	fs.Instance()->OnInit(config_name);
	(env)->ReleaseStringUTFChars(configPath, config_name);
}
JNIEXPORT jobjectArray JNICALL Java_JniFastSearch_Query(JNIEnv *env, jclass args, jstring URL, jstring catName, jint topK)
{
	const char *url = (env)->GetStringUTFChars(URL, 0);
	const char *cat_name = (env)->GetStringUTFChars(catName, 0);
	jclass Cls = (env)->FindClass("java/lang/Object");
	jobjectArray Array = (env)->NewObjectArray((jsize)topK, Cls, 0);
	std::vector<std::pair<std::string, int> > result;
	LOG(INFO)<<"Query Information:";
	LOG(INFO)<<"category\t:\t"<<cat_name<<std::endl;
	LOG(INFO)<<"url\t:\t"<<url<<std::endl;
	fs.Instance()->Query(url, cat_name, topK, result);
	for (int i = 0; i < topK; ++i)
	{
		const char *res = (result[i].first + ':' + slh::IntToString(result[i].second)).c_str();
		jstring str = env->NewStringUTF(res);;
		env->SetObjectArrayElement(Array, i, str);
	}
	(env)->ReleaseStringUTFChars(URL, url);
	(env)->ReleaseStringUTFChars(catName, cat_name);
	return Array;
}
JNIEXPORT void JNICALL Java_JniFastSearch_Run(JNIEnv *, jclass)
{
	fs.Instance()->Run();
}
