/*Start Header
******************************************************************/
/*!
\file kernel.cu
\author ANG HIAP LEE, a.hiaplee, 390000318
		Chloe Lim Jia-Han, j.lim, 440003018
\par a.hiaplee\@digipen.edu
\date 19/4/2021
\brief	main entry point for project
Copyright (C) 2021 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/

#include "Application.h"

int main()
{
	Application application{ 1366, 768 , "CS398 N-Body Simulation" };

	application.Start();
	application.Run();
}