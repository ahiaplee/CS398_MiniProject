#include "Application.h"

//__device__ RenderData* cuda_data = NULL;

//void map_resource(cudaGraphicsResource * resource)
//{
//	size_t size;
//	
//	cudaGraphicsResourceGetMappedPointer((void**)(&cuda_data), &size, resource);
//	printf("Can access %d @ %p\n", size, cuda_data);
//	getLastCudaError("map_resource failed\n");
//}

__global__ void compute_kernel(
	NormalObject* d_Objects,
	RenderData* cuda_data
	//,
	//cudaGraphicsResource* resource,
	//uint max_objects
)
{


	//printf("im running %d %d %d \n", blockIdx.x , blockDim.x , threadIdx.x);
	//printf("im running %d %d %d \n", blockIdx.y , blockDim.y , threadIdx.y);
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	RenderData& data = cuda_data[x];
	
	NormalObject* obj = &(d_Objects[x]);


	//printf("Can access @ %p\n", cuda_data);
	
	data.transform = glm::mat4(1.0f);
	data.transform = glm::scale(data.transform, obj->scale);
	data.transform = glm::rotate(data.transform, glm::radians(obj->rotate), glm::vec3(0.0, 0.0, 1.0));
	data.transform = glm::translate(data.transform, obj->translate);
	data.color[0] = 1.0f;
	data.color[1] = 0.0f;
	data.color[2] = 0.0f;
	data.color[3] = 1.0f;

	//auto ptr = glm::value_ptr(data.transform);

	//for (int i = 0; i < 16; i += 4)
	//{
	//	printf("%f %f %f %f \n", ptr[i], ptr[i + 1], ptr[i + 2], ptr[i + 3]);
	//}
	//printf("==========\n");


	//printf("im running");

	//if(y )
}

void compute_cuda(
	NormalObject* d_Objects,
	cudaGraphicsResource* resource,
	uint max_objects,
	dim3& DimBlock,
	dim3& DimGrid2
)
{
	RenderData* cuda_data;
	size_t size;
	cudaGraphicsResourceGetMappedPointer((void**)(&cuda_data), &size, resource);
	//printf("Can access %d @ %p\n", size, cuda_data);


	compute_kernel << < DimGrid2, DimBlock >> >
		(
			d_Objects,
			cuda_data
			//resource,
			//max_objects,
		);
	getLastCudaError("compute_kernel failed\n");
	cudaDeviceSynchronize();
}