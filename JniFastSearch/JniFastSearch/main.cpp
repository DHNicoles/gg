#include "fast_search.h"

int main(int argc,char* argv[])
{
	assert(argc == 2);
	slh::FastSearch fs;
	fs.Instance()->OnInit(argv[1]);
	fs.Instance()->Run();
	return 0;
}