#version 450 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in mat4 aMtx;
layout(location = 5) in vec4 aColor;
//layout(location = 8) in vec4 aColour;

uniform mat4 view;
uniform mat4 projection;

out vec4 myColor;

void main()
{
	//gl_Position = projection * view * aMtx * vec4(aPos, 1.0);
	gl_Position =  projection * view * aMtx * vec4(aPos, 1.0f);
	myColor = aColor;
}