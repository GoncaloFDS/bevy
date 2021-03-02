use std::ops::Range;

use bevy_asset::Handle;
use bevy_render::{
    pass::RenderPass,
    pipeline::{BindGroupDescriptorId, IndexFormat, PipelineDescriptor},
    renderer::{BindGroupId, BufferId, RenderContext},
};
use bevy_utils::tracing::*;

use crate::renderer::VulkanRenderContext;

pub struct VulkanRenderPass<'a> {
    pub render_context: &'a VulkanRenderContext,
    pub pipeline_descriptor: Option<&'a PipelineDescriptor>,
}

impl<'a> RenderPass for VulkanRenderPass<'a> {
    fn get_render_context(&self) -> &dyn RenderContext {
        self.render_context
    }

    fn set_index_buffer(&mut self, buffer: BufferId, offset: u64, index_format: IndexFormat) {
        warn!("set_index_buffer not implemented");
    }

    fn set_vertex_buffer(&mut self, start_slot: u32, buffer: BufferId, offset: u64) {
        warn!("set_vertex_buffer not implemented");
    }

    fn set_pipeline(&mut self, pipeline_handle: &Handle<PipelineDescriptor>) {
        let pipeline = self
            .render_context
            .render_resource_context
            .render_pipelines
            .read()
            .get(pipeline_handle)
            .unwrap();
    }

    fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        warn!("set_viewport not implemented");
    }

    fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        warn!("set_viewport not implemented");
    }

    fn set_stencil_reference(&mut self, reference: u32) {
        warn!("set_stencil_reference not implemented");
    }

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        warn!("draw not implemented");
    }

    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        warn!("draw_indexed not implemented");
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group_descriptor_id: BindGroupDescriptorId,
        bind_group: BindGroupId,
        dynamic_uniform_indices: Option<&[u32]>,
    ) {
        warn!("set_bind_group not implemented");
    }
}
