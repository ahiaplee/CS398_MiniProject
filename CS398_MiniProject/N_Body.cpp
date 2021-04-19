/*Start Header
******************************************************************/
/*!
\file kernel.cu
\author ANG HIAP LEE, a.hiaplee, 390000318
        Chloe Lim Jia-Han, j.lim, 440003018
\par a.hiaplee\@digipen.edu
\date 19/4/2021
\brief	CPU functions for project
Copyright (C) 2021 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/

#include "Application.h"


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
    return sqrtf(G * (7500.0f / (float(N * N))) * solarMass / glm::length(pv));
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
        _objects[i]->force = glm::vec2(0.0f, 0.0f); /* resetting the force acting on the body */
        // N squared algo
        for (size_t j = 0; j < N; ++j) /* handle interaction with every object */
            if (i != j) AddForceNormalObject(*_objects[i], *_objects[j]);
    }
    for (size_t i = 0; i < N; ++i)
    {
        UpdateNormalObject(*_objects[i]); /* handle the movement */
    }
}
