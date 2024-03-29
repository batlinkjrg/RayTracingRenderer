// Walnut Libs
#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"
#include "Walnut/Image.h"

// Custom Libs
#include "Renderer/Renderer.h"
#include "Camera/GpuCamera.h"

// STD Libs
#include <iostream>

// GLM Libs
#include <glm/gtc/type_ptr.hpp>

using namespace Walnut;

class ExampleLayer : public Walnut::Layer
{
public:
	// Set camera
	ExampleLayer() : camera(45.0f, 0.1f, 100.0f) {
		scene.SetScene();
	}

	virtual void OnUpdate(float ts) override
	{
		camera.OnUpdate(ts);
	}

	virtual void OnUIRender() override
	{
		// Info window
		ImGui::Begin("Info");

			if (r.m_Image) {
				ImGui::Text("Last render: %.3fms", r.m_LastRenderTime);
				ImGui::Text("Width:	%ipx",				r.getCurrentWidth()			);
				ImGui::Text("Height: %ipx",				r.getCurrentHeight()		);
				ImGui::Text("Total Pixels: %ipx",		r.getCurrentTotalPixels()	);
				ImGui::Text("Total Spheres:	%i",		scene.SphereTotal()			);
				ImGui::Text("Current Ray Bounces: %i",	*r.getCurrentBounceCount()	);
			}

		ImGui::End();

		// Settings
		ImGui::Begin("Settings");

			// Set Ray Bounces
			ImGui::SliderInt(": Ray Bounces", r.getCurrentBounceCount(), 0, 50);

			// Seperate Ray from Sphere settings
			ImGui::Separator();

			// See if spheres need to be added
			ImGui::SliderInt(": Spheres to add", &numOfSphereToAdd, 0, 1000);

			if (ImGui::Button("Add Sphere")) {
				sphereHandleCount = numOfSphereToAdd;
				for (int i = 0; i < sphereHandleCount; i++) {
					scene.addSphere(Scene::createSphere());
					scene.updateScene();
				}

				sphereHandleCount = 0;
			}

			if (ImGui::Button("Delete Sphere")) {
				sphereHandleCount = numOfSphereToAdd;

				for (int i = 0; i < sphereHandleCount; i++) {
					scene.removeSphere(scene.SphereTotal() - 1);
					scene.updateScene();
				}

				sphereHandleCount = 0;
			}

		ImGui::End();


		// Scene Config panel
		ImGui::Begin("SceneConfig");

			// Sphere controls
			for (int i = 0; i < scene.sphereSet.size(); i++) {
				ImGui::PushID(i);

				// Create a control panel for the sphere
				int sphereNumber = i + 1;
				std::string txt = "Sphere: " + std::to_string(sphereNumber);
				ImGui::Text(txt.c_str());

				// Set up Controls for the sphere
				SimpleSphereInfo& sphere = scene.sphereSet[i];
				if ( ImGui::Checkbox("Visable", &sphere.visable)					 ) { scene.updateScene(); }
				if ( ImGui::DragFloat3("Position", value_ptr(sphere.position), 0.1f) ) { scene.updateScene(); }
				if ( ImGui::DragFloat("Radius", &sphere.radius, 0.1f)				 ) { scene.updateScene(); }
				if ( ImGui::ColorEdit3("Color", value_ptr(sphere.material.color))	 ) { scene.updateScene(); }

				ImGui::Separator();
				ImGui::PopID();
			}

		ImGui::End();

		// Render Window
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(2.0f, 2.0f));
			ImGui::Begin("ViewPort");

				// Set dimensions and render image
				camera.OnResize(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y);

				r.ResizeImage(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y);
				r.RenderImage(camera, scene);

				// Then render image if one is avaliable
				if (r.m_Image)
					ImGui::Image(r.m_Image->GetDescriptorSet(), { (float)r.m_Image->GetWidth(), (float)r.m_Image->GetHeight() }, ImVec2(0,1), ImVec2(1,0));

			ImGui::End();
		ImGui::PopStyleVar();
	}

private:
	Scene scene;
	Camera camera;
	Renderer r;
	int sphereHandleCount = 0;
	int numOfSphereToAdd = 0;
};

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{

	Walnut::ApplicationSpecification spec;
	spec.Name = "RayTracingEngine";

	Walnut::Application* app = new Walnut::Application(spec);
	app->PushLayer<ExampleLayer>();
	app->SetMenubarCallback([app]()
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Exit"))
			{
				app->Close();
			}
			ImGui::EndMenu();
		}
	});
	return app;
}