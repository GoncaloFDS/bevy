use ash::{version::DeviceV1_0, vk, Device};

use bevy_render::{
    pass::{PassDescriptor, RenderPass},
    renderer::{BufferId, RenderContext, RenderResourceBindings, RenderResourceContext, TextureId},
    texture::Extent3d,
};
use bevy_utils::tracing::*;

use crate::{renderer::VulkanRenderResourceContext, vulkan_render_pass::VulkanRenderPass};

pub struct VulkanRenderContext {
    pub render_resource_context: VulkanRenderResourceContext,
}

impl VulkanRenderContext {
    pub fn new(resources: VulkanRenderResourceContext) -> Self {
        VulkanRenderContext {
            // device,
            render_resource_context: resources,
        }
    }

    pub fn finish(&self, queue: &mut vk::Queue) -> Option<vk::CommandBuffer> {
        if self.render_resource_context.command_buffers.read().len() == 0 {
            return None;
        }
        let image_index = 0;
        let wait_semaphore = [*self
            .render_resource_context
            .image_available_semaphore
            .read()];
        let signal_semaphore = [*self
            .render_resource_context
            .render_finished_semaphore
            .read()];

        {
            // Submit command buffer
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers =
                [self.render_resource_context.command_buffers.read()[image_index]];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphore)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphore)
                .build();
            let submit_infos = [submit_info];
            unsafe {
                self.render_resource_context
                    .device
                    .queue_submit(*queue, &submit_infos, vk::Fence::null())
                    .unwrap()
            }
            info!("queue_submit");
        }
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
        unimplemented!()
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
        unimplemented!()
    }

    #[allow(unused_variables)]
    fn begin_pass(
        &mut self,
        pass_descriptor: &PassDescriptor,
        render_resource_bindings: &RenderResourceBindings,
        run_pass: &mut dyn Fn(&mut dyn RenderPass),
    ) {
        let mut swapchain_framebuffers = create_framebuffers(
            self.render_resource_context.device.as_ref(),
            &self.render_resource_context.swap_chain_image_views.read(),
            *self.render_resource_context.render_pass.read(),
            vk::Extent2D {
                height: 800,
                width: 800,
            },
        );


        // self.render_resource_context.swapchain_frame_buffers.write().clear();
        self.render_resource_context
            .swapchain_frame_buffers
            .write()
            .append(&mut swapchain_framebuffers);

        info!(" swap {:?} ", self.render_resource_context.swapchain_frame_buffers.read());

        let mut vulkan_render_pass = VulkanRenderPass {
            render_context: self,
            pipeline_descriptor: None,
        };

        run_pass(&mut vulkan_render_pass);
    }
}

fn create_framebuffers(
    device: &Device,
    image_views: &[vk::ImageView],
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
) -> Vec<vk::Framebuffer> {
    image_views
        .iter()
        .map(|view| [*view])
        .map(|attachments| {
            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1)
                .build();
            unsafe { device.create_framebuffer(&framebuffer_info, None).unwrap() }
        })
        .collect()
}
