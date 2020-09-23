#include "ExModule.h"


ExModule::ExModule()
{
    _input_height = 0;
    _input_width = 0;
    _input_channel = 0;

    _input_tensor_name = "Input";
    _output_tensor_name = "Identity";

    _session = NULL;
}

ExModule::~ExModule()
{
    delete _session;
}

void ExModule::initialize(const int mode, const int input_height, const int input_width, const int input_channel)
{
    _input_height = input_height;
    _input_width = input_width;
    _input_channel = input_channel;
    string model_path = "models\\model.pb";

    // 设置 CPU / GPU模式
    ReadBinaryProto(Env::Default(), model_path, &_graphdef);
    _options.config.mutable_device_count()->insert({ "GPU", mode });
    if (mode)
        // 限制内存增长
        _options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
    NewSession(_options, &_session);
    _session->Create(_graphdef);
}

int ExModule::predict(Mat input_image)
{
    Tensor tensor = Tensor(DT_FLOAT, TensorShape({ 1,_input_height, _input_width, _input_channel }));
    _cv_mat_to_tensor(input_image, &tensor);
    _session->Run({ {_input_tensor_name, tensor} }, { _output_tensor_name }, {}, &_outputs);                                                                                     // Fetch the first tensor
    auto result = _outputs[0].tensor<float, 2>();
    int output_dim = _outputs[0].shape().dim_size(1);

    float max_prob = result(0, 0);
    int label = 0;
    for (int j = 0; j < output_dim; j++)
    {
        if (max_prob < result(0, j))
        {
            max_prob = result(0, j);
            label = j;
        }
    }
    return label;
}

void ExModule::_cv_mat_to_tensor(Mat& image, Tensor* output_tensor)
{
    float* p = output_tensor->flat<float>().data();

    Scalar channel_mean = mean(image);
    float global_mean = (channel_mean[0] + channel_mean[1] + channel_mean[2]) / 3.0;
    image.convertTo(image, CV_32FC3);
    image = image - Scalar(global_mean, global_mean, global_mean);
    image = image / 255.0;

    Mat tempMat(_input_height, _input_width, CV_32FC3, p);
    image.convertTo(tempMat, CV_32FC3);
}
