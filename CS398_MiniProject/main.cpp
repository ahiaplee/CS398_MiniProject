#include "Application.h"



int main()
{
	Application application{ 1366, 768 , "Application Window" };

	application.Start();
	application.Run();
}