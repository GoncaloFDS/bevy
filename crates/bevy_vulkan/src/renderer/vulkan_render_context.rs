use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::{vk, Device};

use bevy_render::{
    pass::{PassDescriptor, RenderPass},
    renderer::{BufferId, RenderContext, RenderResourceBindings, RenderResourceContext, TextureId},
    texture::Extent3d,
};
use bevy_utils::tracing::*;

use crate::renderer::VulkanRenderResourceContext;
use crate::VulkanRenderPass;

#[derive(Debug, Default)]
pub struct LazyCommandBuffer {
    command_buffer: Option<vk::CommandBuffer>,
}

impl LazyCommandBuffer {
    pub fn get_or_create(
        &mut self,
        device: &Device,
        command_pool: vk::CommandPool,
    ) -> &mut vk::CommandBuffer {
        match self.command_buffer {
            Some(ref mut command_buffer) => command_buffer,
            None => {
                self.create(device, command_pool);
                self.command_buffer.as_mut().unwrap()
            }
        }
    }

    pub fn is_some(&self) -> bool {
        self.command_buffer.is_some()
    }

    pub fn create(&mut self, device: &Device, command_pool: vk::CommandPool) {
        let create_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&create_info).unwrap() };
        self.command_buffer = Some(command_buffer[0]);
    }

    pub fn take(&mut self) -> Option<vk::CommandBuffer> {
        self.command_buffer.take()
    }

    pub fn get(&self) -> Option<vk::CommandBuffer> {
        self.command_buffer
    }

    pub fn set(&mut self, command_buffer: vk::CommandBuffer) {
        self.command_buffer = Some(command_buffer);
    }
}

pub struct VulkanRenderContext {
    pub device: Arc<Device>,
    pub command_buffer: LazyCommandBuffer,
    pub render_resource_context: VulkanRenderResourceContext,
}

impl VulkanRenderContext {
    pub fn new(device: Arc<Device>, resources: VulkanRenderResourceContext) -> Self {
        VulkanRenderContext {
            device,
            render_resource_context: resources,
            command_buffer: LazyCommandBuffer::default(),
        }
    }

    pub fn finish(&mut self) -> Option<vk::CommandBuffer> {
        self.command_buffer.take().map(|command_buffer| unsafe {
            info!("finish pass");
            self.device.end_command_buffer(command_buffer).unwrap();
            command_buffer
        })
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
        warn!("copy_texture_to_buffer not implemented")
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
        warn!("copy_texture_to_texture not implemented")
    }

    #[allow(unused_variables)]
    fn begin_pass(
        &mut self,
        pass_descriptor: &PassDescriptor,
        render_resource_bindings: &RenderResourceBindings,
        run_pass: &mut dyn FnMut(&mut dyn RenderPass),
    ) {
        info!("begin pass");
        if !self.command_buffer.is_some() {
            self.command_buffer
                .create(&self.device, self.render_resource_context.command_pool);

            unsafe {
                self.device
                    .begin_command_buffer(
                        self.command_buffer.get().unwrap(),
                        &vk::CommandBufferBeginInfo::default(),
                    )
                    .unwrap()
            }
        }

        let resource_lock = self.render_resource_context.resources.read();
        let refs = resource_lock.refs();

        let mut vulkan_render_pass = VulkanRenderPass {
            render_context: self,
            vulkan_resources: refs,
            pipeline_descriptor: None,
        };

        run_pass(&mut vulkan_render_pass);
    }
}
