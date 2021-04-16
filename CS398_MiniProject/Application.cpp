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
        _fps = 1 / _deltaTime;
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

// individual body functions
float Application::InitialForceCalcNormalObject(const glm::vec2& pv)
{
    // math time
    /* This function will determine the magnitude of the initial velocity
       vector of the body, given the position within the unit circle it was
       initially generated in.

       Given the universe as a unit circle, the central body is in the origin.
       The length of pv gives an idea of how far the object is from the center,
       with the furtherst it could be being sqrt(0.5) units away, given that
       we are treating the bodies to be within a ciruclar orbit (circle of radius
       0.5).

       This implementation plays on the function to calculate orbital speed
       around the central body. The original function is 
       sqrtf(G * solarMass / distance). What we have done was include an
       additional constant coefficient of 7500 / (N * N), to allow for a more
       suitable approximation of speed given the number of objects in the 
       world. This coefficient is a custom implementation for our project
       and has a primary role of beautifying the thing over any actual scientific
       application. The rationale was to take a large enough value and scale it
       down by the square of the number of bodies in the system. */
    return sqrtf(G * (7500.0f / (float(N*N))) * solarMass / glm::length(pv));
}
void Application::UpdateNormalObject(NormalObject& obj)
{
    // basic application of force
        /* apply the force acting on the body onto the velocity 
           second differential application applied to first differential */
    obj.velocity += (float)_deltaTime * obj.force;
        /* transform the body with respect to movement defined
           by newly updated velocity from above (simple moving) */
    obj.translate += (float)_deltaTime * glm::vec3(obj.velocity.x, obj.velocity.y, 0.0f);
}
void Application::AddForceNormalObject(NormalObject& obja, NormalObject& objb)
{
    // calculating the force objb acts on obja and applying it
    static const float softeningsq = 9.0f; /* force no division by 0 */
    glm::vec2 difference = objb.translate - obja.translate; /* difference in position */
    float magnitudesq = difference.x * difference.x + difference.y * difference.y; /* square of magnitude */

    // math time
    /* F here represents the gravitational force of the two objects. The formula
       to calculate F is Gravitational_Constant * m_1 * m_2 / distance^2. The following
       line implements the above equation. Per object, Gravitaional_Constant * m_1
       will be constant, with m_2 and distance^2 varying per the object it is
       interacting with.

       The magnitude of a displacement vector (difference vector between two objects)
       is the distance between them. To get distance^2, we squared the magnitude.

       Our implementation includes an additional variable in the denominator called
       softeningsq, used to prevent a division by 0. This division by 0 in the original
       function is only there when the two objects are in the same position, which
       results in the displacement vector being a null vector. Our implementation aims
       to be as accurate as possible, such as ignoring collisions between bodies (which
       is actually a thing in astrophysical simulations it is really cool). Thus the
       ultimate addition of the softening factor. */
    float F = (G * obja.mass * objb.mass) / (magnitudesq + softeningsq);
    // math time part 2
    /* The final force applied onto the object will be the force calculated above (F)
       multiplied by the unit vector of the displaceent vector. This force, as stated
       above, represents the gravitational force both objects are acting on each other.
       Thus the application of the force will be having both objects move towards each
       other by the magnitude of the force (F) in each other's direction (unit vector
       of the displacement vector). */
    obja.force += F / sqrtf(magnitudesq) * difference; /* apply the forcee acting on obja from objb */
}

// N body functions
    /* This function provides a brute force way to initialise the data of every body */
void Application::InitNBody()
{
    // destroy all existing objects
    _objects.clear();

    //float universeRad = 1e18; // arbitrary radius of the universe for calculations

    // center heavy body central mass
    auto first = std::make_unique<NormalObject>(); // make an object
    first->translate = glm::vec3{ 0.0f, 0.0f, 0.0f }; // center of the world
    first->velocity = glm::vec2{ 0.0f, 0.0f }; // no movement
    first->mass = solarMass; // predetermined solar mass
        // main color of object
    first->altcolor[0] = 1.0f; 
    first->altcolor[1] = 1.0f;
    first->altcolor[2] = 1.0f;
    first->altcolor[3] = 1.0f;
        // color to show when looking at mass displacement
    first->basecolor[0] = 1.0f;
    first->basecolor[1] = 0.64f;
    first->basecolor[2] = 0.55f;
    first->basecolor[3] = 1.0f;

    if (useBaseColor) // setting of which color to use
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
        auto obj = std::make_unique<NormalObject>(); // make an object

            /* Take the arbitrary radius of the world (1e2f) and scale it by
               a randombly generated ratio per the specified lambda value (-1.8f).
               A basic base 10 logarithmic function was chosen to capitalise on
               its wide variance in output values between the input values of
               0 and 1. The output is negated to become a positive value and
               ultimately scaled down to a more reasonable rance (division by -1.8) */
        float pvCoefficient = 1e2f * EXP(-1.8f);
            /* randomly generated vector within unit circle */
        glm::vec2 pv{ RAND_FLOAT(-0.5f, 0.5f), RAND_FLOAT(-0.5f, 0.5f) };
        glm::vec2 position = pvCoefficient * pv; /* final position */
        obj->translate = glm::vec3{ position, 0.0f }; /* obj translate */
        float magnitude = InitialForceCalcNormalObject(position); /* get force magnitude */

        float absAngle = atanf(fabsf(pv.y / pv.x)); /* get the positive angle from unit circle */
        float vTheta = glm::half_pi<float>() - absAngle; /* get theta angle */
            /* the signs are determined by sqapping the "opposite (y)" and "adjacent (x)" axes units
               with the application of theta treated as a standard euclidean projection:
               v.x will take the cosine of theta, while v.y will take the since of theta.
               everything will then be multiplied by the magnitude calculated above. */
        glm::vec2 v{ (position.y >= 0.0f ? position.y > 0.0f ? -1.0f : 0.0f : 1.0f) * cosf(vTheta) * magnitude,
            (position.x >= 0.0f ? position.x > 0.0f ? 1.0f : 0.0f : -1.0f) * sinf(vTheta) * magnitude };
        if (RAND_FLOAT(0.0f, 1.0f) >= 0.5f) v *= -1.0f; /* randombly set an orientation, 50% chance */
        obj->velocity = v; /* obj velocity */

        obj->mass = RAND_FLOAT(0.0f, solarMass) * 10.0f; /* obj mass */

        static const float massConstant = solarMass * 10.0f; /* largest possible mass */
        float colorCoefficient = floorf(obj->mass * 254.0f / massConstant) / 255.0f; /* mass ratio for color gradient */

        /* randomly calculate color (with mass gradient applied for the fun of it) */
        obj->altcolor[0] = colorCoefficient * RAND_FLOAT(0.0f, 1.0f);
        obj->altcolor[1] = colorCoefficient * RAND_FLOAT(0.0f, 1.0f);
        obj->altcolor[2] = colorCoefficient * RAND_FLOAT(0.0f, 1.0f);
        obj->altcolor[3] = 1.0f;

        /* calculate mass color value when applying mass gradient via barycentric ratioing */
        obj->basecolor[0] = 1.0f - colorCoefficient + colorCoefficient * endColor.r;
        obj->basecolor[1] = 1.0f - colorCoefficient + colorCoefficient * endColor.g;
        obj->basecolor[2] = 1.0f - colorCoefficient + colorCoefficient * endColor.b;
        obj->basecolor[3] = 1.0f; 

        /* setting of color */
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
        _objects[i]->force= glm::vec2(0.0f, 0.0f); /* resetting the force acting on the body */
        // N squared algo
        for (size_t j = 0; j < N; ++j) /* handle interaction with every object */
            if (i != j) AddForceNormalObject(*_objects[i], *_objects[j]);
    }
    for (size_t i = 0; i < N; ++i)
    {
        UpdateNormalObject(*_objects[i]); /* handle the movement */
    }
}
