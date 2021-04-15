#include "Application.h"

Application::Application(int width, int height, const std::string& window_title):
	_width {width},
	_height { height },
	_windowTitle{ window_title },
	_window {nullptr}
{
}

Application::~Application()
{
    if (use_cuda)
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
    use_cuda = true;
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



    //Shader* shader = new Shader("VS.vert", "FS.frag");
    _shaders.emplace(
        ShaderTypes::DEFAULT_INSTANCED, 
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
    
    //hardcode object to test draw
    //for (int i = 0; i < N; ++i)
    //{
    //    auto obj = std::make_unique<NormalObject>();
    //    obj->translate = glm::vec3{
    //        RAND_FLOAT (-100.0f, 100.0f),
    //        RAND_FLOAT (-100.0f, 100.0f),
    //        0.0f//RAND_FLOAT (-10.0f, 10.0f)
    //    };
    //    obj->color[0] = RAND_FLOAT(0.0f, 1.0f);
    //    obj->color[1] = RAND_FLOAT(0.0f, 1.0f);
    //    obj->color[2] = RAND_FLOAT(0.0f, 1.0f);
    //    obj->color[3] = 1.0f;
    //    _objects.push_back(std::move(obj));
    //}
    InitNBody();

   
    Init_RenderObject(_InstancedObject); 


    //cudaGraphicsMapResources(1, resources);
    //map_resource(resources[0]);
    //cudaGraphicsUnmapResources(1, resources);
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
    //dim3 DimGrid2(
    //    1, 1 , 1
    //);
    DimGrid2 = dim3( 
        ceil(((float)_objects.size()) / BLOCK_SIZE),
        1,
        1);




    auto size = _objects.size() * sizeof(NormalObject);




    checkCudaErrors(cudaMalloc((void**)&d_Objects, size));
    checkCudaErrors(cudaMemcpy(d_Objects, copy_objs.data(), size, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(resources, objs.transforms, cudaGraphicsMapFlagsNone));

    std::cout << "test" << std::endl;

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
        _fps = 1 / _deltaTime;
    }
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
    bool view_changed = false;

    if (glfwGetKey(_window, GLFW_KEY_W) == GLFW_PRESS)
    {
        eye.z -= 0.05f;
        view_changed = true;
    }
    if (glfwGetKey(_window, GLFW_KEY_S) == GLFW_PRESS)
    {
        eye.z += 0.05f;
        view_changed = true;
    }

    if (glfwGetKey(_window, GLFW_KEY_A) == GLFW_PRESS)
    {
        eye.x -= 0.05f;
        target.x -= 0.05f;
        view_changed = true;
    }

    if (glfwGetKey(_window, GLFW_KEY_D) == GLFW_PRESS)
    {
        eye.x += 0.05f;
        target.x += 0.05f;
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

    UpdateNBody();

    if (use_changed)
    {
        Rebind_RenderObject(_InstancedObject);


        if (use_cuda)
            Init_CudaResource(_InstancedObject);


        use_changed = false;
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
    cudaGraphicsMapResources(1, resources);
   // map_resource(resources[0]);

    compute_cuda
        (
            d_Objects, resources[0], _objects.size(), DimBlock, DimGrid2
        );
    cudaGraphicsUnmapResources(1, resources);
    

    glBindVertexArray(_InstancedObject.VAO);
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, 1000);
}


void Application::GUI()
{
    size_t NValuesAvail[4] = { 100, 500, 1000, 5000 };
    const char* NValues[4] = { "100", "500", "1000", "5000" };
    static const char* NCurrent = "1000";
    static int currIndex = 2;

    static bool colorGradient = false;
    static float editablebaseColor[3] = { 0.0f, 0.4843f, 0.7235f };

    if (ImGui::Begin("Tools"))
    {
        ImGui::Text("FPS: %f", _fps);
        ImGui::Text("Frame Time: %f", _deltaTime);
        if (ImGui::BeginCombo("N Value", NCurrent))
        {
            for (int i = 0; i < 4; ++i)
            {
                if (ImGui::Selectable(NValues[i], i == currIndex))
                {
                    currIndex = i;
                    NCurrent = NValues[i];
                    N = NValuesAvail[i];
                    _objects.clear();
                    InitNBody();
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::Checkbox("Display Mass Gradient", &colorGradient) && colorGradient != useBaseColor)
        {
            useBaseColor = colorGradient;

            if(useBaseColor)
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
            ImGui::Text("The closer in mass to the central object, the lighter the color");
            ImGui::Text("Hover the colors below for details");
            ImGui::ColorButton("Central Object", ImVec4{ 1.0f, 0.64f, 0.55f, 1.0f });
            ImGui::SameLine();
            ImGui::ColorButton("Heavy Object", ImVec4{ 1.0f, 1.0f, 1.0f, 1.0f });
            ImGui::SameLine();
            ImGui::ColorButton("Light Object", ImVec4{ endColor.r, endColor.g, endColor.b, 1.0f });

        if (ImGui::Checkbox("Use cuda", &use_cuda))
        {
            use_changed = true;

        }
    }
    ImGui::End();
}

// individual body functions
float Application::InitialForceCalcNormalObject(const glm::vec2& pv)
{
    return sqrtf(G * (7500.0f / (float(N*N))) * solarMass / glm::length(pv));
}
void Application::UpdateNormalObject(NormalObject& obj)
{
    obj.velocity += (float)_deltaTime * obj.force;
    obj.translate += (float)_deltaTime * glm::vec3(obj.velocity.x, obj.velocity.y, 0.0f);
}
void Application::ResetForceNormalObject(NormalObject& obj)
{
    obj.force = glm::vec2(0.0f, 0.0f);
}
void Application::AddForceNormalObject(NormalObject& obja, NormalObject& objb)
{
    static float softeningsq = 3.0f * 3.0f;
    glm::vec2 difference = objb.translate - obja.translate;
    float magnitudesq = difference.x * difference.x + difference.y * difference.y;
    float F = (G * obja.mass * objb.mass) / (magnitudesq + softeningsq);
    obja.force += F / sqrtf(magnitudesq) * difference;
}

// N body functions
void Application::InitNBody()
{
    float universeRad = 1e18;

    // center heavy body central mass
    auto first = std::make_unique<NormalObject>();
    first->translate = glm::vec3{ 0.0f, 0.0f, 0.0f };
    first->velocity = glm::vec2{ 0.0f, 0.0f };
    first->mass = solarMass; 
    first->altcolor[0] = 1.0f;
    first->altcolor[1] = 1.0f;
    first->altcolor[2] = 1.0f;
    first->altcolor[3] = 1.0f;
    first->basecolor[0] = 1.0f;
    first->basecolor[1] = 0.64f;
    first->basecolor[2] = 0.55f;
    first->basecolor[3] = 1.0f; /* obj color w gradient */
    if (useBaseColor)
    {
        first->color[0] = first->basecolor[0];
        first->color[1] = first->basecolor[1];
        first->color[2] = first->basecolor[2];
        first->color[3] = first->basecolor[3];
    }
    else
    {
        first->color[0] = first->altcolor[0];
        first->color[1] = first->altcolor[1];
        first->color[2] = first->altcolor[2];
        first->color[3] = first->altcolor[3];
    }

    _objects.push_back(std::move(first));

    for (size_t i = 1; i < N; ++i)
    {
        auto obj = std::make_unique<NormalObject>();
        //obj->translate = glm::vec3{
        //	RAND_FLOAT(-100.0f, 100.0f),
        //	RAND_FLOAT(-100.0f, 100.0f),
        //	0.0f//RAND_FLOAT (-10.0f, 10.0f)
        //};
        //obj->color[0] = RAND_FLOAT(0.0f, 1.0f);
        //obj->color[1] = RAND_FLOAT(0.0f, 1.0f);
        //obj->color[2] = RAND_FLOAT(0.0f, 1.0f);
        //obj->color[3] = 1.0f;

        float pvCoefficient = 1e2f * EXP(-1.8f);
        glm::vec2 pv{ RAND_FLOAT(-0.5f, 0.5f), RAND_FLOAT(-0.5f, 0.5f) };
        glm::vec2 position = pvCoefficient * pv;
        obj->translate = glm::vec3{ position, 0.0f }; /* obj translate */
        float magnitude = InitialForceCalcNormalObject(position);

        float absAngle = atanf(fabsf(pv.y / pv.x));
        float vTheta = glm::half_pi<float>() - absAngle;
        float vPhi = RAND_FLOAT(0.0f, glm::pi<float>());
        glm::vec2 v{ (position.y >= 0.0f ? position.y > 0.0f ? -1.0f : 0.0f : 1.0f) * cosf(vTheta) * magnitude,
            (position.x >= 0.0f ? position.x > 0.0f ? 1.0f : 0.0f : -1.0f) * sinf(vTheta) * magnitude };
        if (RAND_FLOAT(0.0f, 1.0f) >= 0.5f) v *= -1.0f;
        obj->velocity = v; /* obj velocity */

        obj->mass = RAND_FLOAT(0.0f, solarMass) * 10.0f; /* obj mass */

        float massConstant = solarMass * 10.0f;
        float colorCoefficient = floorf(obj->mass * 254.0f / massConstant) / 255.0f;

        obj->altcolor[0] = colorCoefficient * RAND_FLOAT(0.0f, 1.0f);
        obj->altcolor[1] = colorCoefficient * RAND_FLOAT(0.0f, 1.0f);
        obj->altcolor[2] = colorCoefficient * RAND_FLOAT(0.0f, 1.0f);
        obj->altcolor[3] = 1.0f; /* obj color */

        obj->basecolor[0] = 1.0f - colorCoefficient + colorCoefficient * endColor.r;
        obj->basecolor[1] = 1.0f - colorCoefficient + colorCoefficient * endColor.g;
        obj->basecolor[2] = 1.0f - colorCoefficient + colorCoefficient * endColor.b;
        obj->basecolor[3] = 1.0f; /* obj color w gradient */

        if (useBaseColor)
        {
            obj->color[0] = obj->basecolor[0];
            obj->color[1] = obj->basecolor[1];
            obj->color[2] = obj->basecolor[2];
            obj->color[3] = obj->basecolor[3];
        }
        else
        {
            obj->color[0] = obj->altcolor[0];
            obj->color[1] = obj->altcolor[1];
            obj->color[2] = obj->altcolor[2];
            obj->color[3] = obj->altcolor[3];
        }

        _objects.push_back(std::move(obj));
    }
}
void Application::UpdateNBody()
{
    for (size_t i = 0; i < N; ++i)
    {
        ResetForceNormalObject(*_objects[i]);
        // N squared algo
        for (size_t j = 0; j < N; ++j)
            if (i != j) AddForceNormalObject(*_objects[i], *_objects[j]);
    }
    for (size_t i = 0; i < N; ++i)
    {
        UpdateNormalObject(*_objects[i]);
    }

    //std::cout << "Object 1: (" << _objects[1]->translate.x << ", " 
    //    << _objects[1]->translate.y << " " << _objects[1]->translate.z << ")" << std::endl;
}
