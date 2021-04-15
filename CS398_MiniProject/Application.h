#pragma once

#define DEFAULT_PARTICLE_SIZE 1000

#define GLEW_STATIC
#include <glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "Rendering.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <unordered_map>
#include <vector>
#include <memory>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/matrix.hpp>

using uint = unsigned;


inline float RAND_FLOAT(float LO, float HI)
{
	return LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
}

enum ShaderTypes
{
	DEFAULT_INSTANCED
};

struct NormalObject
{
	glm::vec3 scale = {1.0f, 1.0f, 1.0f};
	float rotate = 0.0f;
	glm::vec3 translate = { 0.0f, 0.0f, 0.0f };
	float color[4] = { 1.0f,1.0f,1.0f,1.0f };
};

struct RenderData
{
	glm::mat4 transform;
	float color[4] = { 1.0f,1.0f,1.0f,1.0f };
};


struct InstancedObject
{
	uint VAO;
	uint VBO;
	uint transforms;
	uint EBO;

	//std::vector<glm::mat4> datas { DEFAULT_PARTICLE_SIZE };
	std::vector<RenderData> datas { DEFAULT_PARTICLE_SIZE };
};

class Application
{
private:

	std::string _windowTitle;
	int _width;
	int _height;

	GLFWwindow* _window;
	double _startTime = 0;
	double _deltaTime = 0;
	double _fps = 0;


	glm::mat4 projection;
	glm::mat4 view;

	glm::vec3 eye{ 0,0, 25.0f };
	glm::vec3 target{ 0,0,0 };


	std::unordered_map<ShaderTypes, std::unique_ptr<Shader>> _shaders;

	std::vector<std::unique_ptr<NormalObject>> _objects;
	InstancedObject _InstancedObject;

	void GUI();
	void InputProcess(GLFWwindow* window);
	void Init_RenderObject(InstancedObject& obj);
	void Rebind_RenderObject(InstancedObject& obj);

	void PrintErrors();
	void Update();
	void Draw();

public:

	Application(int width, int height, const std::string& window_title = "Application Window");
	~Application();

	Application(const Application& app) = delete;
	Application& operator=(const Application& app) = delete;


	void Start();
	void Run();

};



