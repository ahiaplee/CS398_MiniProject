#include "Application.h"

float camera_speed = 100.0f;

Application::Application(int width, int height, const std::string& window_title):
	_width {width},
	_height { height },
	_windowTitle{ window_title },
	_window {nullptr}
{
}

Application::~Application()
{
    if (d_Objects)
    {
        cudaFree(d_Objects);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

	glfwTerminate();
}

void Application::Start()
{
    use_cuda = false;
    Print_GPU_Info();

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    _window = glfwCreateWindow(_width, _height, _windowTitle.c_str() , NULL, NULL);
    if (_window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(_window);
    glfwSetFramebufferSizeCallback(_window, 
        [](GLFWwindow* window, int width, int height) 
        { 
            glViewport(0, 0, width, height);
        }
    );

    glewInit();
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(_window, true);
    ImGui_ImplOpenGL3_Init("#version 450 core");
    glfwSwapInterval(0);
    glfwMaximizeWindow(_window);



    //Shader* shader = new Shader("VS.vert", "FS.frag");
    _shaders.emplace(
        DEFAULT_INSTANCED, 
        std::move(std::make_unique<Shader>("VS.vert", "FS.frag"))
    );


    projection = glm::perspective(glm::radians(45.0f), (float)_width / (float)_height, 0.1f, 1000.0f);
    view = glm::lookAt(
        eye,
        target,
        glm::vec3{ 0,1.0f,0.0f }
    );

    _shaders[DEFAULT_INSTANCED]->use();
    _shaders[DEFAULT_INSTANCED]->setMat4("projection", glm::value_ptr(projection));
    _shaders[DEFAULT_INSTANCED]->setMat4("view", glm::value_ptr(view));
    
    InitNBody();

   
    Init_RenderObject(_InstancedObject); 
}

void Application::Print_GPU_Info()
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;

    int dev = findCudaDevice(0, 0);

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
        deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
}

void Application::Init_CudaResource(InstancedObject& objs)
{

    DimBlock = dim3 (BLOCK_SIZE, 1, 1);
    DimGrid2 = dim3( 
        ceil(((float)_objects.size()) / BLOCK_SIZE),
        1,
        1);

    auto size = _objects.size() * sizeof(NormalObject);

    if (d_Objects)
    {
        checkCudaErrors(cudaFree(d_Objects));
    }

    checkCudaErrors(cudaMalloc((void**)&d_Objects, size));
    checkCudaErrors(cudaMemcpy(d_Objects, copy_objs.data(), size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(resources, objs.transforms, cudaGraphicsMapFlagsNone));

}

void Application::Run()
{

    while (!glfwWindowShouldClose(_window))
    {
        _startTime = glfwGetTime();

        glfwPollEvents();
        InputProcess(_window);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        glClearColor(0.f, 0.f, 0.f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        
        //ImGui::ShowDemoWindow(false);
        Update();

        if (use_cuda)
            Draw_Cuda(); 
        else
            Draw();

        GUI();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(_window);
       
        auto endTime = glfwGetTime();
        _deltaTime = endTime - _startTime;
        //_fps = 1 / _deltaTime;

        if (updateTime < 0.0f)
        {
            _fps = 1 / (accuDelta / accuCount);
            updateTime = 1.0f;
            accuCount = 0;
            accuDelta = 0;
        }
        else
        {
            accuDelta += _deltaTime;
            updateTime -= _deltaTime;
            ++accuCount;
        }
          
       
    }
}

void Application::Reset()
{
    InitNBody();
    Rebind_RenderObject(_InstancedObject);


    if (use_cuda)
        Init_CudaResource(_InstancedObject);


    use_changed = false;
}

void Application::InputProcess(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void Application::Init_RenderObject(InstancedObject& obj)
{
    glGenVertexArrays(1, &obj.VAO);
    glGenBuffers(1, &obj.VBO);
    glGenBuffers(1, &obj.transforms);
    glGenBuffers(1, &obj.EBO);


    float vertices[]
    {
        -0.5f, -0.5f, 
         0.5f, -0.5f,
        -0.5f,  0.5f,
         0.5f,  0.5f
    };

    unsigned indices[] =
    {
        0, 1, 2,
        1, 3, 2
    };

    glBindVertexArray(obj.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, obj.VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, obj.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    //Vertices
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);



    Rebind_RenderObject(obj);


    if (use_cuda)
        Init_CudaResource(_InstancedObject);

    PrintErrors();
}

void Application::Rebind_RenderObject(InstancedObject& objs)
{

    if (use_cuda)
    {
        //precompute first set of matrices for cuda
        objs.datas.clear();


        copy_objs.clear();
        for (auto& object : _objects)
        {
            RenderData data;
            std::copy(std::begin(object->color), std::end(object->color), data.color);

            data.transform = glm::mat4(1.0f);
            //data.transform = glm::scale(data.transform, obj->scale);
            //data.transform = glm::rotate(data.transform, glm::radians(obj->rotate), glm::vec3(0.0, 0.0, 1.0));
            //data.transform = glm::translate(data.transform, obj->translate);

            objs.datas.push_back(data);
            copy_objs.push_back(*object);
        }
    }



    glBindBuffer(GL_ARRAY_BUFFER, objs.transforms);
    glBufferData(GL_ARRAY_BUFFER, sizeof(RenderData) * objs.datas.size(), objs.datas.data(), GL_DYNAMIC_DRAW);


    glBindVertexArray(objs.VAO);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(RenderData), (void*)0);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(RenderData), (void*)(sizeof(glm::vec4)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(RenderData), (void*)(2 * sizeof(glm::vec4)));
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(RenderData), (void*)(3 * sizeof(glm::vec4)));

    PrintErrors();

    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(RenderData), (void*)(4 * sizeof(glm::vec4)));

    PrintErrors();

    glVertexAttribDivisor(1, 1);	// Divisor Mat4x4
    glVertexAttribDivisor(2, 1);	// Divisor Mat4x4
    glVertexAttribDivisor(3, 1);	// Divisor Mat4x4
    glVertexAttribDivisor(4, 1);	// Divisor Mat4x4
    glVertexAttribDivisor(5, 1);	// Divisor Mat4x4
    //glVertexAttribDivisor(8, 1);	// Divisor Vec4
    //glVertexAttribDivisor(9, 1);	// Divisor Vec4

    
}

void Application::PrintErrors()
{

    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        std::string error;
        switch (errorCode)
        {
        case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
        case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
        case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
        case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
        case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
        case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
        case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        std::cout << error << std::endl;
    }
}

void Application::Update()
{
    

    if (glfwGetKey(_window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        eye.z -= camera_speed * _deltaTime;
        view_changed = true;
    }
    if (glfwGetKey(_window, GLFW_KEY_E) == GLFW_PRESS)
    {
        eye.z += camera_speed * _deltaTime;
        view_changed = true;
    }

    if (glfwGetKey(_window, GLFW_KEY_W) == GLFW_PRESS)
    {
        eye.y += camera_speed * _deltaTime;
        target.y += camera_speed * _deltaTime;
        view_changed = true;
    }

    if (glfwGetKey(_window, GLFW_KEY_S) == GLFW_PRESS)
    {
        eye.y -= camera_speed * _deltaTime;
        target.y -= camera_speed * _deltaTime;
        view_changed = true;
    }

    if (glfwGetKey(_window, GLFW_KEY_A) == GLFW_PRESS)
    {
        eye.x -= camera_speed * _deltaTime;
        target.x -= camera_speed * _deltaTime;
        view_changed = true;
    }

    if (glfwGetKey(_window, GLFW_KEY_D) == GLFW_PRESS)
    {
        eye.x += camera_speed * _deltaTime;
        target.x += camera_speed * _deltaTime;
        view_changed = true;
    }

    if (view_changed)
    {
        view = glm::lookAt(
            eye,
            target,
            glm::vec3{ 0,1.0f,0.0f }
        );
    }

    if(!pause && !use_cuda)
        UpdateNBody();

    if (use_changed)
    {
        Reset();
    }

}

void Application::Draw()
{
    _InstancedObject.datas.clear();
    for (auto& obj : _objects)
    {
        RenderData data;
        std::copy(std::begin(obj->color), std::end(obj->color), data.color);

        data.transform = glm::mat4(1.0f);
        data.transform = glm::scale(data.transform, obj->scale);
        data.transform = glm::rotate(data.transform, glm::radians(obj->rotate), glm::vec3(0.0, 0.0, 1.0));
        data.transform = glm::translate(data.transform, obj->translate);
        //auto ptr = glm::value_ptr(transform);

        //for (int i = 0; i < 16; i += 4)
        //    std::cout << ptr[i] << ptr[i+1] << ptr[i+2] << ptr[i+3] << std::endl;

        _InstancedObject.datas.push_back(data);
    }

    Rebind_RenderObject(_InstancedObject);
    PrintErrors();

    _shaders[DEFAULT_INSTANCED]->use();
    _shaders[DEFAULT_INSTANCED]->setMat4("view", glm::value_ptr(view));
    _shaders[DEFAULT_INSTANCED]->setMat4("projection", glm::value_ptr(projection));

    glBindVertexArray(_InstancedObject.VAO);
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, static_cast<GLsizei>(_InstancedObject.datas.size()));



}

void Application::Draw_Cuda()
{
    _shaders[DEFAULT_INSTANCED]->use();
    _shaders[DEFAULT_INSTANCED]->setMat4("view", glm::value_ptr(view));
    _shaders[DEFAULT_INSTANCED]->setMat4("projection", glm::value_ptr(projection));

    //RenderData* datas;

    if (!pause)
    {
        cudaGraphicsMapResources(1, resources);
        compute_cuda
        (
            d_Objects, resources[0], _objects.size(), DimBlock, DimGrid2, N, G, _deltaTime, useBaseColor
        );
        cudaGraphicsUnmapResources(1, resources);
    }

    

    glBindVertexArray(_InstancedObject.VAO);
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, _objects.size());
}


void Application::GUI()
{
    size_t NValuesAvail[3] = { 500, 1000, 5000 };
    const char* NValues[3] = { "500", "1000", "5000" };
    static const char* NCurrent = "1000";
    static int currIndex = 1;

    static bool colorGradient = false;
    static float editablebaseColor[3] = { 0.0f, 0.4843f, 0.7235f };

    if (ImGui::Begin("Tools"))
    {
        ImGui::Text("FPS: %f", _fps);
        ImGui::Text("Frame Time: %f", _deltaTime);

        ImGui::Text("WASD to move");
        ImGui::Text("Q & E to adjust zoom");

        ImGui::Separator();
        if (ImGui::Checkbox("Use CUDA", &use_cuda))
        {
            use_changed = true;

        }
        if (ImGui::Checkbox("Pause", &pause))
        {
            //use_changed = true;

        }
        
        if (ImGui::BeginCombo("N Value", NCurrent))
        {
            for (int i = 0; i < 3; ++i)
            {
                if (ImGui::Selectable(NValues[i], i == currIndex))
                {
                    currIndex = i;
                    NCurrent = NValues[i];
                    N = NValuesAvail[i];
                    _objects.clear();              
                    Reset();
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::Checkbox("Display Mass Gradient", &colorGradient) && colorGradient != useBaseColor)
        {
            useBaseColor = colorGradient;

            if (useBaseColor)
                for (auto& obj : _objects)
                {
                    obj->color[0] = obj->basecolor[0];
                    obj->color[1] = obj->basecolor[1];
                    obj->color[2] = obj->basecolor[2];
                    obj->color[3] = obj->basecolor[3];
                }
            else
                for (auto& obj : _objects)
                {
                    obj->color[0] = obj->altcolor[0];
                    obj->color[1] = obj->altcolor[1];
                    obj->color[2] = obj->altcolor[2];
                    obj->color[3] = obj->altcolor[3];
                }
        }
        if (colorGradient)
        {
            ImGui::Text("The hevier the object, the lighter the color");
            ImGui::Text("Hover the colors below for details");
            ImGui::ColorButton("Central Object", ImVec4{ 1.0f, 0.64f, 0.55f, 1.0f });
            ImGui::SameLine();
            ImGui::ColorButton("Heavy Object\n can be up to 10 times the mass\n of the central object", ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
            ImGui::SameLine();
            ImGui::ColorButton("Light Object", ImVec4{ endColor.r, endColor.g, endColor.b, 1.0f });


        }

        ImGui::End();
    }
}
