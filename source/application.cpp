#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

namespace application {

namespace {

constexpr float pi = 3.14159265358979323846f;

// Одна вершина содержит три координаты и три компоненты цвета.
// Порядок полей должен совпадать с атрибутами location 0 и 1 в шейдере
// и с описанием входа графического конвейера ниже.
struct Vertex {
	float position[3];
	float color[3];
};
// Исходный правильный тетраэдр имеет вершины A(1,1,1), B(1,-1,-1),
// C(-1,1,-1) и D(-1,-1,1). На каждом из шести рёбер взяты две точки
// на расстоянии одной трети ребра от его концов: всего 12 вершин.
// В паре AB/BA первая буква обозначает исходную вершину, от которой
// отложена треть ребра. Цвет хранится рядом с координатами.
const Vertex vertices[] = {
	{{ 1.0f,  1.0f / 3.0f,  1.0f / 3.0f}, {0.20f, 0.70f, 0.90f}}, // AB
	{{ 1.0f, -1.0f / 3.0f, -1.0f / 3.0f}, {0.85f, 0.35f, 0.80f}}, // BA
	{{ 1.0f / 3.0f,  1.0f,  1.0f / 3.0f}, {0.95f, 0.55f, 0.20f}}, // AC
	{{-1.0f / 3.0f,  1.0f, -1.0f / 3.0f}, {0.60f, 0.60f, 0.60f}}, // CA
	{{ 1.0f / 3.0f,  1.0f / 3.0f,  1.0f}, {0.95f, 0.30f, 0.25f}}, // AD
	{{-1.0f / 3.0f, -1.0f / 3.0f,  1.0f}, {0.60f, 0.60f, 0.60f}}, // DA
	{{ 1.0f / 3.0f, -1.0f / 3.0f, -1.0f}, {0.50f, 0.35f, 0.90f}}, // BC
	{{-1.0f / 3.0f,  1.0f / 3.0f, -1.0f}, {0.60f, 0.60f, 0.60f}}, // CB
	{{ 1.0f / 3.0f, -1.0f, -1.0f / 3.0f}, {0.25f, 0.80f, 0.35f}}, // BD
	{{-1.0f / 3.0f, -1.0f,  1.0f / 3.0f}, {0.60f, 0.60f, 0.60f}}, // DB
	{{-1.0f,  1.0f / 3.0f, -1.0f / 3.0f}, {0.25f, 0.50f, 0.95f}}, // CD
	{{-1.0f, -1.0f / 3.0f,  1.0f / 3.0f}, {0.90f, 0.75f, 0.20f}}, // DC
};
//*************************************************Индексы и грани*********************************************
// Vulkan рисует треугольники, поэтому каждая из четырёх треугольных
// граней занимает три индекса, а каждый шестиугольник разбит веером
// на четыре треугольника. Итого: 4 + 4*4 = 20 треугольников, 60 индексов.
// Обход всех треугольников направлен наружу и согласован с отсечением
// обратных граней. В каждом веере первый индекс повторяется: flat
// в шейдерах берёт цвет этой вершины без интерполяции по грани.
const uint32_t indices[] = {
	// Four triangular faces.
	4, 0, 2,
	8, 6, 1,
	10, 3, 7,
	11, 9, 5,

	// Hexagon ABC: 0, 1, 6, 7, 3, 2.
	0, 1, 6,  0, 6, 7,  0, 7, 3,  0, 3, 2,

	// Hexagon BAD: 1, 0, 4, 5, 9, 8.
	1, 0, 4,  1, 4, 5,  1, 5, 9,  1, 9, 8,

	// Hexagon ACD: 2, 3, 10, 11, 5, 4.
	2, 3, 10,  2, 10, 11,  2, 11, 5,  2, 5, 4,

	// Hexagon BDC: 6, 8, 9, 11, 10, 7.
	6, 8, 9,  6, 9, 11,  6, 11, 10,  6, 10, 7,
};

static_assert(sizeof(vertices) / sizeof(vertices[0]) == 12);
static_assert(sizeof(indices) / sizeof(indices[0]) == 60);

// Матрицы хранятся по столбцам, как mat4 в GLSL. elements[column*4+row]
// адресует элемент в строке row и столбце column; транспонировать
// матрицу перед копированием в uniform-буфер не требуется.
// ****************************************************Матрица, uniform и сохранённые объекты********************************
struct Matrix4 {
	float elements[16]{};
};

struct GlobalUniforms {
	Matrix4 matrix;
};

VkBuffer vk_vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vma_vertex_buffer_allocation = VK_NULL_HANDLE; // буфер вершин и выделенная под него память

VkBuffer vk_index_buffer = VK_NULL_HANDLE;
VmaAllocation vma_index_buffer_allocation = VK_NULL_HANDLE;// буфер индексов

VkBuffer vk_uniform_buffer = VK_NULL_HANDLE;
VmaAllocation vma_uniform_buffer_allocation = VK_NULL_HANDLE;
GlobalUniforms* vk_uniform_buffer_memory = nullptr; // буфер матрицы, выделенная память и указатель, через который процессор записывает матрицу

VkDescriptorSetLayout vk_descriptor_set_layout = VK_NULL_HANDLE; // описание того, какие данные ожидает шейдер
VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE; // источник, из которого выделяется набор дескрипторов
VkDescriptorSet vk_descriptor_set = VK_NULL_HANDLE; // конкретная связь шейдера с нашим uniform buffer

VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE; // описание интерфейса конвейера, включая дескрипторы
VkPipeline vk_pipeline = VK_NULL_HANDLE; // готовые настройки рисования

float rotation_x = 0.35f;
float rotation_y = 0.55f;
bool orthographic_projection = false;
double previous_time = 0.0;// Это состояние программы: два начальных угла поворота, выбранный тип проекции и время прошлого кадра. false означает, что вначале включена перспектива.

// ****************************************Функции для матриц***********************************************

Matrix4 identityMatrix() {
	Matrix4 result;
	result.elements[0] = 1.0f;
	result.elements[5] = 1.0f;
	result.elements[10] = 1.0f;
	result.elements[15] = 1.0f;
	return result;
}// создаёт единичную матрицу. На главной диагонали стоят четыре единицы. Применение такой матрицы не меняет точку.

Matrix4 multiply(const Matrix4& left, const Matrix4& right) {
	Matrix4 result;
	for (int column = 0; column < 4; ++column) {
		for (int row = 0; row < 4; ++row) {
			for (int i = 0; i < 4; ++i) {
				result.elements[column * 4 + row] +=
					left.elements[i * 4 + row] * right.elements[column * 4 + i];
			}
		}
	}
	return result;
} // перемножает две матрицы. Внешние циклы обходят столбцы и строки результата, внутренний цикл складывает произведения соответствующих элементов. Так собирается общая матрица из отдельных преобразований.

Matrix4 translation(float x, float y, float z) {
	Matrix4 result = identityMatrix();
	result.elements[12] = x;
	result.elements[13] = y;
	result.elements[14] = z;
	return result;
} // создаёт матрицу переноса. Значения x, y, z записываются в последний столбец.

Matrix4 rotationX(float angle) {
	Matrix4 result = identityMatrix();
	result.elements[5] = std::cos(angle);
	result.elements[6] = std::sin(angle);
	result.elements[9] = -std::sin(angle);
	result.elements[10] = std::cos(angle);
	return result;
} // матрица поворота вокруг оси X. Используются sin и cos угла.

Matrix4 rotationY(float angle) {
	Matrix4 result = identityMatrix();
	result.elements[0] = std::cos(angle);
	result.elements[2] = -std::sin(angle);
	result.elements[8] = std::sin(angle);
	result.elements[10] = std::cos(angle);
	return result;
} // аналогично

Matrix4 perspective(float fov, float aspect, float near_plane, float far_plane) {
	Matrix4 result;
	const float tangent = std::tan(fov / 2.0f);
	result.elements[0] = 1.0f / (aspect * tangent);
	result.elements[5] = -1.0f / tangent;
	result.elements[10] = far_plane / (near_plane - far_plane);
	result.elements[11] = -1.0f;
	result.elements[14] = far_plane * near_plane / (near_plane - far_plane);
	return result;
} // перспективная проекция. fov ? угол обзора, aspect ? отношение ширины окна к высоте, near_plane и far_plane ? ближняя и дальняя границы видимости. При перспективе далёкие части выглядят меньше.

Matrix4 orthographic(float half_height, float aspect,
					 float near_plane, float far_plane) {
	Matrix4 result = identityMatrix();
	result.elements[0] = 1.0f / (half_height * aspect);
	result.elements[5] = -1.0f / half_height;
	result.elements[10] = 1.0f / (near_plane - far_plane);
	result.elements[14] = near_plane / (near_plane - far_plane);
	return result;
} // ортографическая проекция. Она задаёт видимый объём через половину высоты, отношение сторон и границы глубины. Размер предмета в ней не уменьшается из-за удаления.
// Минус в elements[5] у обеих проекций переворачивает направление экранной оси Y для выбранной системы координат Vulkan.

// *****************************************************Загрузка шейдера*******************************************


// loadShaderModule(path) открывает файл шейдера в двоичном режиме. std::ios::ate сразу ставит указатель чтения в конец,
// поэтому tellg() даёт размер файла. Далее создаётся массив uint32_t, чтение переносится в начало файла, и байты загружаются в этот массив.
// VkShaderModuleCreateInfo передаёт Vulkan размер и адрес загруженного кода. vkCreateShaderModule создаёт объект шейдера.
// Если открыть файл или создать объект не удалось, функция печатает ошибку и возвращает VK_NULL_HANDLE.
// Здесь читаются уже скомпилированные файлы .spv, а не исходные .vert и .frag.
VkShaderModule loadShaderModule(const char path[]) {
	auto& context = graphics::internal::context;
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader: " << path << '\n';
		return VK_NULL_HANDLE;
	}

	const size_t size = static_cast<size_t>(file.tellg());
	std::vector<uint32_t> code(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size));

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = code.data(),
	};

	VkShaderModule result = VK_NULL_HANDLE;
	if (vkCreateShaderModule(context.device, &info, nullptr, &result) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module: " << path << '\n';
		return VK_NULL_HANDLE;
	}
	return result;
}

} // namespace

//************************************************ initialize(): три буфера ************************************************

bool initialize() {
	auto& context = graphics::internal::context;

	// Вершинный буфер хранит 12 структур Vertex. VMA выделяет память,
// доступную CPU, и сразу отображает её: pMappedData используется
// для memcpy. После копирования flush делает запись видимой GPU.
	const VkBufferCreateInfo vertex_buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(vertices),
		.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo vertex_allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
				 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	VmaAllocationInfo vertex_memory_info{};
	if (vmaCreateBuffer(context.allocator, &vertex_buffer_info, &vertex_allocation_info,
						&vk_vertex_buffer, &vma_vertex_buffer_allocation,
						&vertex_memory_info) != VK_SUCCESS) {
		std::cerr << "Failed to create vertex buffer\n";
		return false;
	}
	std::memcpy(vertex_memory_info.pMappedData, vertices, sizeof(vertices));
	vmaFlushAllocation(context.allocator, vma_vertex_buffer_allocation, 0, sizeof(vertices));

	// Индексный буфер создаётся отдельно, потому что vkCmdDrawIndexed
// читает номера вершин как uint32_t. Одна вершина может участвовать
// в нескольких треугольниках, не дублируясь в вершинном буфере.
	const VkBufferCreateInfo index_buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(indices),
		.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo index_allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
				 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	VmaAllocationInfo index_memory_info{};
	if (vmaCreateBuffer(context.allocator, &index_buffer_info, &index_allocation_info,
						&vk_index_buffer, &vma_index_buffer_allocation,
						&index_memory_info) != VK_SUCCESS) {
		std::cerr << "Failed to create index buffer\n";
		shutdown();
		return false;
	}
	std::memcpy(index_memory_info.pMappedData, indices, sizeof(indices));
	vmaFlushAllocation(context.allocator, vma_index_buffer_allocation, 0, sizeof(indices));

	// Uniform-буфер содержит одну матрицу MVP для всего объекта. Память
// остаётся отображённой на протяжении работы программы, но обновлять
// её можно только после завершения чтения предыдущим кадром GPU.
	const VkBufferCreateInfo uniform_buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(GlobalUniforms),
		.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo uniform_allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
				 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	VmaAllocationInfo uniform_memory_info{};
	if (vmaCreateBuffer(context.allocator, &uniform_buffer_info, &uniform_allocation_info,
						&vk_uniform_buffer, &vma_uniform_buffer_allocation,
						&uniform_memory_info) != VK_SUCCESS) {
		std::cerr << "Failed to create uniform buffer\n";
		shutdown();
		return false;
	}
	vk_uniform_buffer_memory = static_cast<GlobalUniforms*>(uniform_memory_info.pMappedData);

// *************************************************** initialize(): дескриптор ***********************************************

	// DescriptorSetLayout описывает binding 0: один uniform-буфер,
// доступный вершинному шейдеру. Номер совпадает с set=0, binding=0
// в файле truncated_tetrahedron.vert.
	const VkDescriptorSetLayoutBinding descriptor_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo descriptor_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &descriptor_binding,
	};
	if (vkCreateDescriptorSetLayout(context.device, &descriptor_layout_info, nullptr,
								&vk_descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		shutdown();
		return false;
	}

	// Pool выделяет один набор дескрипторов для единственного объекта.
// vkUpdateDescriptorSets связывает этот набор с уже созданным
// uniform-буфером, чтобы шейдер мог читать текущую матрицу MVP.
	const VkDescriptorPoolSize descriptor_pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
	};
	const VkDescriptorPoolCreateInfo descriptor_pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &descriptor_pool_size,
	};
	if (vkCreateDescriptorPool(context.device, &descriptor_pool_info, nullptr,
						   &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		shutdown();
		return false;
	}

	const VkDescriptorSetAllocateInfo descriptor_set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk_descriptor_set_layout,
	};
	if (vkAllocateDescriptorSets(context.device, &descriptor_set_info,
							 &vk_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set\n";
		shutdown();
		return false;
	}

	const VkDescriptorBufferInfo uniform_descriptor = {
		.buffer = vk_uniform_buffer,
		.offset = 0,
		.range = sizeof(GlobalUniforms),
	};
	const VkWriteDescriptorSet descriptor_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vk_descriptor_set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &uniform_descriptor,
	};
	vkUpdateDescriptorSets(context.device, 1, &descriptor_write, 0, nullptr);

	// Загружаются скомпилированные SPIR-V файлы: вершинный шейдер
// преобразует координаты, фрагментный выдаёт цвет пикселя.
// После создания pipeline временные VkShaderModule освобождаются.
	const VkShaderModule vertex_shader =
		loadShaderModule("shaders/truncated_tetrahedron.vert.spv");
	const VkShaderModule fragment_shader =
		loadShaderModule("shaders/truncated_tetrahedron.frag.spv");
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		if (vertex_shader != VK_NULL_HANDLE) {
			vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		}
		if (fragment_shader != VK_NULL_HANDLE) {
			vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		}
		shutdown();
		return false;
	}

	const VkPipelineShaderStageCreateInfo shader_stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	// Binding задаёт шаг между вершинами sizeof(Vertex). Два атрибута
// указывают смещения position и color; их форматы и location
// должны совпадать с объявлениями входов вершинного шейдера.
	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color),
		},
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = vertex_attributes,
	};
	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};
	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};
	// Наружные грани заданы обходом против часовой стрелки. Отсечение
// обратных граней не даёт рисовать невидимую сторону замкнутой фигуры.
const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};
	const VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};
	// Тест глубины оставляет ближний фрагмент, если грани перекрываются
// после проецирования на экран.
const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};
	const VkPipelineColorBlendAttachmentState color_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
						 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};
	const VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_attachment,
	};
	// Размеры viewport и scissor задаются при записи каждого кадра.
// Поэтому изменение размера окна не требует пересоздания pipeline.
const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	// PipelineLayout фиксирует интерфейс ресурсов для шейдеров:
// единственный набор дескрипторов с матрицей MVP в binding 0.
	const VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &vk_descriptor_set_layout,
	};
	if (vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr,
						   &vk_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		shutdown();
		return false;
	}

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blending,
		.pDynamicState = &dynamic_state,
		.layout = vk_pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};
	const VkResult pipeline_result = vkCreateGraphicsPipelines(
		context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &vk_pipeline);

	vkDestroyShaderModule(context.device, fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);

	if (pipeline_result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		shutdown();
		return false;
	}

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	// Сначала ждём окончания работы очереди GPU. Только затем можно
	// уничтожать pipeline, дескрипторы и буферы, на которые ссылаются кадры.
	vkQueueWaitIdle(context.graphics_queue);

	if (vk_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, vk_pipeline, nullptr);
		vk_pipeline = VK_NULL_HANDLE;
	}
	if (vk_pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
		vk_pipeline_layout = VK_NULL_HANDLE;
	}
	if (vk_descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
		vk_descriptor_pool = VK_NULL_HANDLE;
	}
	if (vk_descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout, nullptr);
		vk_descriptor_set_layout = VK_NULL_HANDLE;
	}
	if (vk_uniform_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vk_uniform_buffer, vma_uniform_buffer_allocation);
		vk_uniform_buffer = VK_NULL_HANDLE;
	}
	if (vk_index_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vk_index_buffer, vma_index_buffer_allocation);
		vk_index_buffer = VK_NULL_HANDLE;
	}
	if (vk_vertex_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vma_vertex_buffer_allocation);
		vk_vertex_buffer = VK_NULL_HANDLE;
	}
}

void update(double time) {
	// Время между кадрами определяет скорость поворота в радианах/с.
	// Ограничение 0,1 с предотвращает большой скачок после паузы окна.
	const float delta_time = previous_time == 0.0
		? 0.0f
		: static_cast<float>(std::min(time - previous_time, 0.1));
	previous_time = time;

	// Когда клавиатура нужна ImGui, управление фигурой не перехватывается.
	// A/D вращают вокруг Y, W/S вокруг X; P меняет тип проекции один раз
	// при нажатии, а не каждый кадр удержания клавиши.
	if (!ImGui::GetIO().WantCaptureKeyboard) {
		if (ImGui::IsKeyDown(ImGuiKey_A)) rotation_y -= delta_time;
		if (ImGui::IsKeyDown(ImGuiKey_D)) rotation_y += delta_time;
		if (ImGui::IsKeyDown(ImGuiKey_W)) rotation_x -= delta_time;
		if (ImGui::IsKeyDown(ImGuiKey_S)) rotation_x += delta_time;
		if (ImGui::IsKeyPressed(ImGuiKey_P, false)) {
			orthographic_projection = !orthographic_projection;
		}
	}

	// Панель только показывает состояние и подсказки. Матрица и запись
	// в память GPU выполняются в render() после ожидания fence в prepare().
	ImGui::Begin("CG Lab #1");
	ImGui::TextUnformatted("Variant 4: truncated regular tetrahedron");
	ImGui::TextUnformatted("12 vertices, 4 triangles, 4 hexagons");
	ImGui::Separator();
	ImGui::TextUnformatted("W/A/S/D - rotate");
	ImGui::TextUnformatted("P - change projection");
	ImGui::Text("Projection: %s", orthographic_projection ? "orthographic" : "perspective");
	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	// prepare() уже дождался fence предыдущего кадра. Теперь GPU больше
	// не читает общий uniform-буфер, поэтому его можно перезаписать.
	// Проекция вычисляется по актуальному размеру swapchain: это также
	// учитывает окно, пересозданное в prepare() после resize.
	const float aspect = static_cast<float>(context.swapchain_extent.width) /
		static_cast<float>(context.swapchain_extent.height);
	const Matrix4 model = multiply(rotationY(rotation_y), rotationX(rotation_x));
	const Matrix4 view = translation(0.0f, 0.0f, -4.5f);
	const Matrix4 projection = orthographic_projection
		? orthographic(2.0f, aspect, 0.1f, 100.0f)
		: perspective(50.0f * pi / 180.0f, aspect, 0.1f, 100.0f);

	// Вершина преобразуется справа налево: сначала модельный поворот,
	// затем перенос в координаты камеры, затем проекция. Запись в
	// отображённую память завершается flush для видимости на GPU.
	vk_uniform_buffer_memory->matrix = multiply(projection, multiply(view, model));
	vmaFlushAllocation(context.allocator, vma_uniform_buffer_allocation,
				   0, sizeof(GlobalUniforms));

	// Перезаписываем команды единственного кадра после ожидания fence.
	vkResetCommandBuffer(fd.command_buffer, 0);
	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

	// Перед отрисовкой очищаем цветовой буфер тёмным цветом, а буфер
	// глубины значением 1, соответствующим самой дальней точке.
	VkClearValue clear_values[2]{};
	clear_values[0].color.float32[0] = 0.03f;
	clear_values[0].color.float32[1] = 0.04f;
	clear_values[0].color.float32[2] = 0.07f;
	clear_values[0].color.float32[3] = 1.0f;
	clear_values[1].depthStencil.depth = 1.0f;

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	// Динамические viewport и scissor занимают всё текущее изображение
	// swapchain. Их размеры берутся заново после изменения окна.
	const VkViewport viewport = {
		.width = static_cast<float>(context.swapchain_extent.width),
		.height = static_cast<float>(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor = { .extent = context.swapchain_extent };
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	// Команда отрисовки использует четыре связанных ресурса: буфер
	// вершин, буфер индексов, набор дескрипторов с MVP и pipeline.
	// vkCmdDrawIndexed читает все 60 индексов и рисует один объект.
	const VkDeviceSize vertex_buffer_offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1,
					   &vk_vertex_buffer, &vertex_buffer_offset);
	vkCmdBindIndexBuffer(fd.command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
						vk_pipeline_layout, 0, 1, &vk_descriptor_set, 0, nullptr);
	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);
	vkCmdDrawIndexed(fd.command_buffer,
		static_cast<uint32_t>(sizeof(indices) / sizeof(indices[0])), 1, 0, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
