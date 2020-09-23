#include <iostream>
#include <vector>

using namespace std;


class BaseRecogDll
{
public:
	virtual void init_model(const int mode) = 0;
	virtual int predict(const string& image_path) = 0;
};

BaseRecogDll * create_model_handle();
void delete_model_handle(BaseRecogDll * ssbr);
