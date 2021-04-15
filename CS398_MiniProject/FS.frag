#version 450 core

out vec4 FragColor;

in vec4 myColor;

void main()
{
	//FragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
	FragColor = myColor;
}