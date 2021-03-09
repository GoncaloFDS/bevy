use ash::vk;

use bevy_render::{
    pass::{PassDescriptor, RenderPass},
    renderer::{BufferId, RenderContext, RenderResourceBindings, RenderResourceContext, TextureId},
    texture::Extent3d,
};
use bevy_utils::tracing::*;

use crate::renderer::VulkanRenderResourceContext;

pub struct VulkanRenderContext {
    pub render_resource_context: VulkanRenderResourceContext,
}

impl VulkanRenderContext {
    pub fn new(resources: VulkanRenderResourceContext) -> Self {
        VulkanRenderContext {
            render_resource_context: resources,
        }
    }

    pub fn finish(&mut self) -> Option<vk::CommandBuffer> {
        None
    }
}

impl RenderContext for VulkanRenderContext {
    fn resources(&self) -> &dyn RenderResourceContext {
        &self.render_resource_context
    }

    fn resources_mut(&mut self) -> &mut dyn RenderResourceContext {
        &mut self.render_resource_context
    }

    #[allow(unused_variables)]
    fn copy_buffer_to_buffer(
        &mut self,
        source_buffer: BufferId,
        source_offset: u64,
        destination_buffer: BufferId,
        destination_offset: u64,
        size: u64,
    ) {
        warn!("copy_buffer_to_buffer not implemented")
    }

    #[allow(unused_variables)]
    fn copy_buffer_to_texture(
        &mut self,
        source_buffer: BufferId,
        source_offset: u64,
        source_bytes_per_row: u32,
        destination_texture: TextureId,
        destination_origin: [u32; 3],
        destination_mip_level: u32,
        size: Extent3d,
    ) {
        warn!("copy_buffer_to_texture not implemented")
    }

    fn copy_texture_to_buffer(
        &mut self,
        _source_texture: TextureId,
        _source_origin: [u32; 3],
        _source_mip_level: u32,
        _destination_buffer: BufferId,
        _destination_offset: u64,
        _destination_bytes_per_row: u32,
        _size: Extent3d,
    ) {
        todo!()
    }

    fn copy_texture_to_texture(
        &mut self,
        _source_texture: TextureId,
        _source_origin: [u32; 3],
        _source_mip_level: u32,
        _destination_texture: TextureId,
        _destination_origin: [u32; 3],
        _destination_mip_level: u32,
        _size: Extent3d,
    ) {
        todo!()
    }

    #[allow(unused_variables)]
    fn begin_pass(
        &mut self,
        pass_descriptor: &PassDescriptor,
        render_resource_bindings: &RenderResourceBindings,
        run_pass: &mut dyn FnMut(&mut dyn RenderPass),
    ) {
        info!("begin pass");
    }
}
