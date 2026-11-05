#version 450 core

// Атрибуты одной вершины поступают из VkBuffer. location 0 читает
// три координаты position, location 1 - три компоненты цвета RGB;
// их смещения и шаг заданы в VkVertexInputAttributeDescription.
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

// binding 0 соответствует VkDescriptorSetLayoutBinding в application.cpp.
// Матрица уже содержит model, view и projection в нужном порядке.
layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    mat4 mvp;
} global_uniforms;

// flat запрещает смешивать цвета вершин внутри треугольника. Цвет
// берётся из его первой вершины, выбранной в массиве индексов.
layout(location = 0) flat out vec3 out_color;

void main() {
    // Однородная координата 1 позволяет матрице MVP выполнять перенос.
    // Результат находится в clip space; деление на w, отсечение и перевод
    // в координаты окна затем выполняет графический конвейер Vulkan.
    gl_Position = global_uniforms.mvp * vec4(in_position, 1.0);
    // Цвет передаётся следующей стадии вместе с выходом вершины.
    out_color = in_color;
}
