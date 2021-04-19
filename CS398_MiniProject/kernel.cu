/*Start Header
******************************************************************/
/*!
\file kernel.cu
\author ANG HIAP LEE, a.hiaplee, 390000318
		Chloe Lim Jia-Han, j.lim, 440003018
\par a.hiaplee\@digipen.edu
\date 19/4/2021
\brief	Kernel functions for project
Copyright (C) 2021 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/

#include "Application.h"

__device__ void GPU_AddForceNormalObject(NormalObject& obja, NormalObject& objb, float G)
{
	static float softeningsq = 3.0f * 3.0f;
	glm::vec2 difference = objb.translate - obja.translate;
	float magnitudesq = difference.x * difference.x + difference.y * difference.y;
	float F = (G * obja.mass * objb.mass) / (magnitudesq + softeningsq);
	obja.force += F / sqrtf(magnitudesq) * difference;
}

__device__ void UpdateNormalObject(NormalObject& obj, float _deltaTime)
{
	obj.velocity += (float)_deltaTime * obj.force;
	obj.translate += (float)_deltaTime * glm::vec3(obj.velocity.x, obj.velocity.y, 0.0f);
}

__global__ void compute_kernel(
	NormalObject* d_Objects,
	RenderData* vbo_data,
	size_t N,
	float G,
	float _deltaTime,
	bool _useBaseColor
)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N)
	{
		RenderData& data = vbo_data[x];
		NormalObject* obj = &(d_Objects[x]);


		obj->force = glm::vec2(0.0f, 0.0f);
		for (size_t j = 0; j < N; ++j)
			if (x != j) GPU_AddForceNormalObject(d_Objects[x], d_Objects[j], G);


		UpdateNormalObject(d_Objects[x], _deltaTime);

		data.transform = glm::mat4(1.0f);
		data.transform = glm::scale(data.transform, obj->scale);
		data.transform = glm::rotate(data.transform, glm::radians(obj->rotate), glm::vec3(0.0, 0.0, 1.0));
		data.transform = glm::translate(data.transform, obj->translate);

		if (_useBaseColor)
		{
			data.color[0] = obj->basecolor[0];
			data.color[1] = obj->basecolor[1];
			data.color[2] = obj->basecolor[2];
			data.color[3] = obj->basecolor[3];
		}
		else
		{
			data.color[0] = obj->altcolor[0];
			data.color[1] = obj->altcolor[1];
			data.color[2] = obj->altcolor[2];
			data.color[3] = obj->altcolor[3];
		}
	}

}

void compute_cuda(
	NormalObject* d_Objects,
	cudaGraphicsResource* resource,
	dim3& DimBlock,
	dim3& DimGrid2,
	size_t N,
	float G,
	float _deltaTime,
	bool _useBaseColor
)
{
	RenderData* cuda_data;
	size_t size;
	cudaGraphicsResourceGetMappedPointer((void**)(&cuda_data), &size, resource);

	compute_kernel << < DimGrid2, DimBlock >> >
		(
			d_Objects,
			cuda_data,
			N,
			G,
			_deltaTime,
			_useBaseColor
		);
	getLastCudaError("compute_kernel failed\n");
	cudaDeviceSynchronize();
}