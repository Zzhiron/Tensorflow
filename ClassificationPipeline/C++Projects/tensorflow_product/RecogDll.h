#pragma once
#include "BaseRecogDll.h"
#include "ExModule.h"

class RecogDll : public BaseRecogDll
{
public:
	void init_model(const int mode);
	int predict(const string& image_path);
private:
	ExModule _ex_module_vvc;
};