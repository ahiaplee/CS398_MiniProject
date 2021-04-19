/*Start Header
******************************************************************/
/*!
\file Application.h
\author ANG HIAP LEE, a.hiaplee, 390000318
		Chloe Lim Jia-Han, j.lim, 440003018
\par a.hiaplee\@digipen.edu
\date 19/4/2021
\brief	Header file for application framework
Copyright (C) 2021 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/

#pragma once

#define DEFAULT_PARTICLE_SIZE 1000
#define BLOCK_SIZE 32

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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <helper_functions.h>  
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using uint = unsigned;


//Helper functions and structs for framework

inline float RAND_FLOAT(float LO, float HI)
{
	return LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
}

inline float EXP(float lambda)
{
	return -logf(1.0f - RAND_FLOAT(0.0f, 1.0f)) / lambda;
}

enum ShaderTypes
{
	DEFAULT_INSTANCED
};

//struct to store objects and their data
struct NormalObject
{
	glm::vec3 scale = {1.0f, 1.0f, 1.0f};
	float rotate = 0.0f;
	glm::vec3 translate = { 0.0f, 0.0f, 0.0f }; // position

	glm::vec2 velocity = { 0.0f, 0.0f };
	glm::vec2 force = { 0.0f, 0.0f };
	float mass = 1.0f;

	float color[4] = { 1.0f,1.0f,1.0f,1.0f };
	float basecolor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	float altcolor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
};

//struct for raw rendering data
struct RenderData
{
	glm::mat4 transform;
	float color[4] = { 1.0f,1.0f,1.0f,1.0f };
};

//struct for OpenGL instanced object
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

	//low level variables
	std::string _windowTitle;
	int _width;
	int _height;

	//glfw variables
	GLFWwindow* _window;
	double _startTime = 0;
	double _deltaTime = 0;
	double _fps = 0;
	double updateTime = 1.0;
	double accuDelta = 0;
	uint accuCount = 0;

	//camera variables
	glm::mat4 projection;
	glm::mat4 view;

	glm::vec3 eye{ 0,0, 300.0f };
	glm::vec3 target{ 0,0,0 };

	//default values for simulation
	size_t N = 1000;
	float G = /*6.673e-4f*/ 52017.875f;
	float solarMass = 1.98892e-3f;

	bool useBaseColor = false;
	glm::vec3 endColor = glm::vec3{ 0.0f, 0.1843f, 0.4235f };

	//rendering values
	std::unordered_map<ShaderTypes, std::unique_ptr<Shader>> _shaders;
	std::vector<std::unique_ptr<NormalObject>> _objects;
	InstancedObject _InstancedObject;

	//helper functions
	void GUI();
	void InputProcess(GLFWwindow* window);
	void Init_RenderObject(InstancedObject& obj);
	void Rebind_RenderObject(InstancedObject& obj);
	void PrintErrors();
	void Update();
	void Draw();
	void Draw_Cuda();
	void Print_GPU_Info();
	void Init_CudaResource(InstancedObject& obj);
	void Reset();

	//misc variables
	bool use_cuda = false;
	bool use_changed = false;
	bool pause = false;
	bool view_changed = false;
	NormalObject* d_Objects = 0;
	cudaGraphicsResource* resources[1];

	std::vector<NormalObject> copy_objs;
	dim3 DimBlock;
	dim3 DimGrid2;



public:

	Application(int width, int height, const std::string& window_title = "Application Window");
	~Application();

	Application(const Application& app) = delete;
	Application& operator=(const Application& app) = delete;


	void Start();
	void Run();

	// individual body functions
	float InitialForceCalcNormalObject(const glm::vec2& pv);
	void UpdateNormalObject(NormalObject& obj);
	void AddForceNormalObject(NormalObject& obja, NormalObject& objb);

	// N body functions
	void InitNBody();
	void UpdateNBody();
};

/// <summary>
/// Helper function to call kernel to compute nbody movement
/// </summary>
/// <param name="d_Objects">device pointer to array of objects</param>
/// <param name="resource">pointer to cuda resource to map </param>
/// <param name="DimBlock">block size</param>
/// <param name="DimGrid2">grid size</param>
/// <param name="N">N-body count</param>
/// <param name="G">Gravity constant</param>
/// <param name="_deltaTime">delta time</param>
/// <param name="_useBaseColor">to use base color</param>
void compute_cuda(
	NormalObject* d_Objects,
	cudaGraphicsResource* resource,
	dim3& DimBlock,
	dim3& DimGrid2,
	size_t N,
	float G,
	float _deltaTime,
	bool _useBaseColor
);






