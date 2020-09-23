#pragma once
#include "common.h"


class ExModule
{
public:
    ExModule();
    ~ExModule();
    void initialize(const int mode, const int input_height, const int input_width, const int input_channel);
    int predict(Mat input_image);
private:
    void _cv_mat_to_tensor(Mat& image, Tensor* output_tensor);
    Session* _session;
    GraphDef _graphdef;
    SessionOptions _options;
    int _input_height;
    int _input_width;
    int _input_channel;
    string _input_tensor_name;
    string _output_tensor_name;
    vector<Tensor> _outputs;
};