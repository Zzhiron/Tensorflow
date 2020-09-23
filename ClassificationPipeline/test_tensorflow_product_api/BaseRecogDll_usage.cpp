#include "BaseRecogDll_usage.h"


void main()
{
	// 设置图像路径
	string image_path = "test.jpg";

	// mode: 设置 0 使用 CPU, 设置 1 使用 GPU
	int mode = 1;

	// 加载模型
	BaseRecogDll* SSR = create_model_handle();
	SSR->init_model(mode);

	// 使用模型进行预测
	int result = SSR->predict(image_path);

	cout << "Prediction for " << image_path << ":  " << result << endl;

	delete_model_handle(SSR);
	cin.get();
}