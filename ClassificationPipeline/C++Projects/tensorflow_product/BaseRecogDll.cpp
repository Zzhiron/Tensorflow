#include "BaseRecogDll.h"
#include "RecogDll.h"

BaseRecogDll* create_model_handle()
{
	RecogDll* ssbr = new RecogDll();
	return (BaseRecogDll*) ssbr;
}
void  delete_model_handle(BaseRecogDll* ssbr)
{
	if (ssbr != NULL)
		delete (RecogDll*)ssbr;
	ssbr = NULL;
}