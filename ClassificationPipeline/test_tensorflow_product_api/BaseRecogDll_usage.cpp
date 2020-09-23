#include "BaseRecogDll_usage.h"


void main()
{
	// ����ͼ��·��
	string image_path = "test.jpg";

	// mode: ���� 0 ʹ�� CPU, ���� 1 ʹ�� GPU
	int mode = 1;

	// ����ģ��
	BaseRecogDll* SSR = create_model_handle();
	SSR->init_model(mode);

	// ʹ��ģ�ͽ���Ԥ��
	int result = SSR->predict(image_path);

	cout << "Prediction for " << image_path << ":  " << result << endl;

	delete_model_handle(SSR);
	cin.get();
}