#include "RecogDll.h"


void RecogDll::init_model(const int mode)
{
    _ex_module.initialize(mode, 512, 640, 3);
}

int RecogDll::predict(const string& image_path)
{
    Mat img_src = imread(image_path, -1);
    Mat img_resized;
    resize(img_src, img_resized, Size(512, 512));
    int result = _ex_module.predict(img_resized);
   
    return result;
}
