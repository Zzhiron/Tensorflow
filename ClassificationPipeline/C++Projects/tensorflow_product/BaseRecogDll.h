#pragma once
#include <iostream>
#include <vector>
#include <windows.h>

using namespace std;

class BaseRecogDll
{
public:
	virtual void init_model(const int mode) = 0;
	virtual int predict(const string& image_path) = 0;
}; 

__declspec(dllexport) BaseRecogDll* create_model_handle();
__declspec(dllexport) void delete_model_handle(BaseRecogDll * ssbr);
