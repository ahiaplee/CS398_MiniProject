#include "Application.h"



int main()
{
	Application application{ 1366, 768 , "CS398 N-Body Simulation" };

	application.Start();
	application.Run();
}