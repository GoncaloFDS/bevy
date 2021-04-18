use std::ops::Range;

use ash::version::DeviceV1_0;
use ash::vk;

use bevy_asset::Handle;
use bevy_render::{
    pass::RenderPass,
    pipeline::{BindGroupDescriptorId, IndexFormat, PipelineDescriptor},
    renderer::{BindGroupId, BufferId, RenderContext},
};
use bevy_utils::tracing::*;

use crate::renderer::VulkanRenderContext;
use crate::VulkanResourceRefs;

pub struct VulkanRenderPass<'a> {
    pub render_context: &'a VulkanRenderContext,
    pub vulkan_resources: VulkanResourceRefs<'a>,
    pub pipeline_descriptor: Option<&'a PipelineDescriptor>,
}

impl<'a> RenderPass for VulkanRenderPass<'a> {
    fn get_render_context(&self) -> &dyn RenderContext {
        self.render_context
    }

    fn set_index_buffer(&mut self, _buffer: BufferId, _offset: u64, _index_format: IndexFormat) {
        warn!("set_index_buffer not implemented");
    }

    fn set_vertex_buffer(&mut self, _start_slot: u32, _buffer: BufferId, _offset: u64) {
        warn!("set_vertex_buffer not implemented");
    }

    fn set_pipeline(&mut self, pipeline_handle: &Handle<PipelineDescriptor>) {
        let pipeline = self
            .vulkan_resources.render_pipelines
            .get(pipeline_handle)
            .expect("Attempted to use a pipeline that does not exist in this `RenderPass`'s `RenderContext`.");
        let device = self.render_context.device.as_ref();
        let command_buffer = &self.render_context.command_buffer;
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer.get().unwrap(),
                vk::PipelineBindPoint::GRAPHICS,
                *pipeline,
            )
        }
    }

    fn set_viewport(
        &mut self,
        _x: f32,
        _y: f32,
        _w: f32,
        _h: f32,
        _min_depth: f32,
        _max_depth: f32,
    ) {
        warn!("set_viewport not implemented");
    }

    fn set_scissor_rect(&mut self, _x: u32, _y: u32, _w: u32, _h: u32) {
        warn!("set_viewport not implemented");
    }

    fn set_stencil_reference(&mut self, _reference: u32) {
        warn!("set_stencil_reference not implemented");
    }

    fn draw(&mut self, _vertices: Range<u32>, _instances: Range<u32>) {
        warn!("draw not implemented");
    }

    fn draw_indexed(&mut self, _indices: Range<u32>, _base_vertex: i32, _instances: Range<u32>) {
        warn!("draw_indexed not implemented");
    }

    fn set_bind_group(
        &mut self,
        _index: u32,
        _bind_group_descriptor_id: BindGroupDescriptorId,
        _bind_group: BindGroupId,
        _dynamic_uniform_indices: Option<&[u32]>,
    ) {
        warn!("set_bind_group not implemented");
    }
}
