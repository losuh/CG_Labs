#include <cstdint>
#include <climits>

#include <iostream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>

#include "graphics_internal.hpp"
#include "application.hpp"

namespace {

// Размер окна задаётся в логических пикселях. Фактический размер области
// рисования берётся из swapchain и может отличаться на дисплее с масштабированием.
constexpr int32_t default_window_width = 1280;
constexpr int32_t default_window_height = 720;

constexpr char default_window_title[] = "CG Lab #1 - Truncated Tetrahedron";

GLFWwindow* glfw_window;

} // namespace

int main() {
	int status = EXIT_SUCCESS;

	// GLFW создаёт окно и доставляет события ввода. Само окно не получает
	// контекст OpenGL: изображения будут выводиться через поверхность Vulkan.
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW\n";
		return EXIT_FAILURE;
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	glfw_window = glfwCreateWindow(default_window_width, default_window_height,
	                               default_window_title, nullptr, nullptr);
	if (glfw_window == nullptr) {
		status = EXIT_FAILURE;
		goto err_null_window;
	}

	// При изменении размера сообщаем графическому модулю новые размеры.
	// Нулевой размер бывает при сворачивании окна; пересоздавать для него
	// swapchain нельзя, поэтому такое событие пропускается.
	glfwSetFramebufferSizeCallback(glfw_window, [](GLFWwindow*, int width, int height){
		if (width == 0 || height == 0) {
			return;
		}

		graphics::internal::resize(width, height);
	});

	// ImGui создаёт интерфейс, но не владеет сценой. GLFW backend передаёт
	// ему ввод, а Vulkan backend создаётся внутри graphics::internal.
	if (ImGui::CreateContext() == nullptr) {
		std::cerr << "Failed to create ImGUI context\n";
		status = EXIT_FAILURE;
		goto err_imgui_init;
	}

	if (!ImGui_ImplGlfw_InitForVulkan(glfw_window, true)) {
		std::cerr << "Failed to initialize ImGUI GLFW backend for Vulkan renderer\n";
		status = EXIT_FAILURE;
		goto err_imgui_glfw_init;
	}

	if (!graphics::internal::initialize(glfw_window)) {
		std::cerr << "Failed to initialize graphics\n";
		status = EXIT_FAILURE;
		goto err_graphics_init;
	}

	// Здесь создаются ресурсы конкретного варианта: геометрия тетраэдра,
	// uniform-буфер, дескриптор и графический pipeline.
	if (!application::initialize()) {
		std::cerr << "Failed to initialize application\n";
		status = EXIT_FAILURE;
		goto err_application_init;
	}

	// Основной цикл продолжается, пока пользователь не закроет окно.
	// Время GLFW используется для независимой от частоты кадров скорости
	// вращения фигуры при удержании клавиш W, A, S и D.
	while (!glfwWindowShouldClose(glfw_window)) {
		const double time = glfwGetTime();

		// GLFW собирает события клавиатуры и окна. ImGui начинает новый кадр,
		// после чего update() меняет только углы поворота, тип проекции и UI.
		// В этой точке предыдущий кадр GPU ещё может читать uniform-буфер.
		glfwPollEvents();
		ImGui_ImplGlfw_NewFrame();

		ImGui::NewFrame();
		application::update(time);
		ImGui::Render();

		// prepare() ждёт fence прошлого кадра и получает изображение swapchain.
		// Только после этого render() записывает новую MVP-матрицу в общий
		// uniform-буфер и формирует команды рисования для текущего кадра.
		graphics::internal::FrameData fd = graphics::internal::prepare();
		application::render(fd);
		// Обе группы команд (фигура и ImGui) отправляются GPU, затем готовое
		// изображение передаётся swapchain для показа в окне.
		graphics::internal::submitAndPresent();
	}

	// Порядок освобождения обратен порядку создания: сначала ресурсы
	// фигуры, затем общие ресурсы Vulkan, ImGui, окно и библиотека GLFW.
	// Метки ниже обеспечивают ту же последовательность при ошибке запуска.
	application::shutdown();
err_application_init:
	graphics::internal::shutdown();
err_graphics_init:
	ImGui_ImplGlfw_Shutdown();
err_imgui_glfw_init:
	ImGui::DestroyContext();
err_imgui_init:
	glfwDestroyWindow(glfw_window);
err_null_window:
	glfwTerminate();

	return status;
}
