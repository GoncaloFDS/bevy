use std::{ops::Range, sync::Arc};

use ash::extensions::khr::Swapchain;
use ash::version::DeviceV1_0;
use ash::{
    extensions::khr::{Surface, Win32Surface},
    vk, Device, Entry, Instance,
};

use bevy_asset::{Assets, Handle, HandleUntyped};
use bevy_render::{
    pipeline::{BindGroupDescriptorId, PipelineDescriptor},
    renderer::{
        BindGroup, BufferId, BufferInfo, BufferMapMode, RenderResourceContext, RenderResourceId,
        SamplerId, TextureId,
    },
    shader::{glsl_to_spirv, Shader, ShaderError, ShaderSource},
    texture::{SamplerDescriptor, TextureDescriptor},
};
use bevy_utils::tracing::*;
use bevy_window::{Window, WindowId};

use crate::renderer::SwapchainSupportDetails;
use crate::vulkan_types::AllocatedImage;
use crate::{
    vulkan_renderer::QueueFamiliesIndices, vulkan_resources::VulkanResources, VulkanRenderer,
};

pub const BIND_BUFFER_ALIGNMENT: usize = 256;
pub const TEXTURE_ALIGNMENT: usize = 256;

pub fn required_extension_names() -> Vec<*const i8> {
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}

#[derive(Clone)]
struct FrameSync {
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl FrameSync {
    fn new(device: &Device) -> Self {
        let image_available_semaphore = unsafe {
            device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap()
        };
        let render_finished_semaphore = unsafe {
            device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap()
        };
        let fence = unsafe {
            let create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            device.create_fence(&create_info, None).unwrap()
        };
        FrameSync {
            image_available_semaphore,
            render_finished_semaphore,
            fence,
        }
    }
}

#[derive(Clone)]
pub struct VulkanRenderResourceContext {
    pub entry: Arc<ash::Entry>,
    pub instance: Arc<ash::Instance>,
    pub device: Arc<ash::Device>,
    pub physical_device: Arc<vk::PhysicalDevice>,

    pub allocator: Arc<vk_mem::Allocator>,
    pub surface_loader: Arc<Surface>,
    pub swapchain_loader: Swapchain,

    pub graphics_queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    frame_sync: Vec<FrameSync>,

    pub resources: VulkanResources,
}

impl VulkanRenderResourceContext {
    pub fn new(renderer: &VulkanRenderer) -> Self {
        let entry = renderer.entry.clone();
        let instance = renderer.instance.clone();
        let device = renderer.device.clone();
        let physical_device = renderer.physical_device.clone();
        let graphics_queue = renderer.graphics_queue;

        let allocator = create_vulkan_allocator(&device, &instance, *physical_device);
        let allocator = Arc::new(allocator);

        let swapchain_loader = Swapchain::new(instance.as_ref(), device.as_ref());
        let surface_loader = Arc::new(Surface::new(entry.as_ref(), instance.as_ref()));

        let command_pool = create_command_pool(&device, renderer.queue_indices.graphics_index);

        let frame_sync = { (0..3).map(|_| FrameSync::new(&device)).collect() };

        let resources = VulkanResources::new(device.clone(), surface_loader.clone());

        VulkanRenderResourceContext {
            entry,
            instance,
            device,
            physical_device,
            allocator,
            surface_loader,
            swapchain_loader,
            graphics_queue,
            command_pool,
            frame_sync,
            resources,
        }
    }

    pub fn set_window_surface(&self, window_id: WindowId, surface: vk::SurfaceKHR) {
        let mut window_surfaces = self.resources.window_surfaces.write();
        window_surfaces.insert(window_id, surface);
    }
}

impl RenderResourceContext for VulkanRenderResourceContext {
    fn create_swap_chain(&self, _window: &Window) {
        warn!("create_swap_chain not implemented")
    }

    fn next_swap_chain_texture(&self, _window: &Window) -> TextureId {
        warn!("next_swap_chain_texture not implemented");
        TextureId::new()
    }

    fn drop_swap_chain_texture(&self, _render_resource: TextureId) {
        warn!("drop_swap_chain_texture not implemented")
    }

    fn drop_all_swap_chain_textures(&self) {
        warn!("drop_all_chain_textures not implemented")
    }

    fn create_sampler(&self, _sampler_descriptor: &SamplerDescriptor) -> SamplerId {
        warn!("create_sampler not implemented");
        SamplerId::new()
    }

    fn create_texture(&self, _texture_descriptor: TextureDescriptor) -> TextureId {
        warn!("create_texture not implemented");
        TextureId::new()
    }

    fn create_buffer(&self, _buffer_info: BufferInfo) -> BufferId {
        warn!("create_buffer not implemented");
        BufferId::new()
    }

    fn write_mapped_buffer(
        &self,
        _id: BufferId,
        _range: Range<u64>,
        _write: &mut dyn FnMut(&mut [u8], &dyn RenderResourceContext),
    ) {
        warn!("write_mapped_buffer not implemented")
    }

    fn read_mapped_buffer(
        &self,
        _id: BufferId,
        _range: Range<u64>,
        _read: &dyn Fn(&[u8], &dyn RenderResourceContext),
    ) {
        warn!("read_mapped_buffer not implemented")
    }

    fn map_buffer(&self, _id: BufferId, _mode: BufferMapMode) {
        warn!("map_buffer not implemented")
    }

    fn unmap_buffer(&self, _id: BufferId) {
        warn!("unmap_buffer not implemented")
    }

    fn create_buffer_with_data(&self, _buffer_info: BufferInfo, _data: &[u8]) -> BufferId {
        warn!("create_buffer_with_data not implemented");
        BufferId::new()
    }

    fn create_shader_module(&self, _shader_handle: &Handle<Shader>, _shaders: &Assets<Shader>) {
        todo!()
    }

    fn create_shader_module_from_source(&self, _shader_handle: &Handle<Shader>, _shader: &Shader) {
        todo!()
    }

    fn get_specialized_shader(
        &self,
        shader: &Shader,
        macros: Option<&[String]>,
    ) -> Result<Shader, ShaderError> {
        let spirv_data = match shader.source {
            ShaderSource::Spirv(ref bytes) => bytes.clone(),
            ShaderSource::Glsl(ref source) => glsl_to_spirv(&source, shader.stage, macros)?,
        };
        Ok(Shader {
            source: ShaderSource::Spirv(spirv_data),
            ..*shader
        })
    }

    fn remove_buffer(&self, _buffer: BufferId) {
        todo!()
    }

    fn remove_texture(&self, _texture: TextureId) {
        todo!()
    }

    fn remove_sampler(&self, _sampler: SamplerId) {
        todo!()
    }

    fn get_buffer_info(&self, _buffer: BufferId) -> Option<BufferInfo> {
        todo!()
    }

    fn get_aligned_uniform_size(&self, size: usize, dynamic: bool) -> usize {
        if dynamic {
            (size + BIND_BUFFER_ALIGNMENT - 1) & !(BIND_BUFFER_ALIGNMENT - 1)
        } else {
            size
        }
    }

    fn get_aligned_texture_size(&self, data_size: usize) -> usize {
        (data_size + TEXTURE_ALIGNMENT - 1) & !(TEXTURE_ALIGNMENT - 1)
    }

    fn set_asset_resource_untyped(
        &self,
        _handle: HandleUntyped,
        _render_resource: RenderResourceId,
        _index: u64,
    ) {
        warn!("set_asset_resource_untyped not implemented")
    }

    fn get_asset_resource_untyped(
        &self,
        _handle: HandleUntyped,
        _index: u64,
    ) -> Option<RenderResourceId> {
        warn!("get_asset_resource_untyped not implemented");
        None
    }

    fn remove_asset_resource_untyped(&self, _handle: HandleUntyped, _index: u64) {
        todo!()
    }

    fn create_render_pipeline(
        &self,
        _pipeline_handle: Handle<PipelineDescriptor>,
        _pipeline_descriptor: &PipelineDescriptor,
        _shaders: &Assets<Shader>,
    ) {
        warn!("create_render_pipeline not implemented");
    }

    fn bind_group_descriptor_exists(
        &self,
        _bind_group_descriptor_id: BindGroupDescriptorId,
    ) -> bool {
        warn!("bind_group_descriptor_exists not implemented");
        true
    }

    fn create_bind_group(
        &self,
        _bind_group_descriptor_id: BindGroupDescriptorId,
        _bind_group: &BindGroup,
    ) {
        warn!("create_bind_group not implemented");
    }

    fn clear_bind_groups(&self) {
        todo!()
    }

    fn remove_stale_bind_groups(&self) {
        warn!("remove_stale_bind_groups not implemented");
    }
}

fn create_vulkan_allocator(
    device: &Device,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> vk_mem::Allocator {
    let create_info = vk_mem::AllocatorCreateInfo {
        physical_device,
        device: device.clone(),
        instance: instance.clone(),
        flags: vk_mem::AllocatorCreateFlags::empty(),
        preferred_large_heap_block_size: 0,
        frame_in_use_count: 0,
        heap_size_limits: None,
    };
    vk_mem::Allocator::new(&create_info).unwrap()
}

fn create_swapchain(
    swapchain_loader: &Swapchain,
    surface: vk::SurfaceKHR,
    swapchain_config: &SwapchainSupportDetails,
    old_swapchain: Option<vk::SwapchainKHR>,
    preferred_dimensions: [u32; 2],
) -> vk::SwapchainKHR {
    let props = swapchain_config.get_ideal_swapchain_properties(preferred_dimensions);
    let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(3)
        .image_format(props.format.format)
        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .image_extent(props.extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(props.present_mode)
        .clipped(true);

    let swapchain_create_info = match old_swapchain {
        None => swapchain_create_info,
        Some(old_swapchain) => swapchain_create_info.old_swapchain(old_swapchain),
    };

    unsafe {
        swapchain_loader
            .create_swapchain(&swapchain_create_info, None)
            .unwrap()
    }
}

fn create_command_pool(device: &Device, queue_index: u32) -> vk::CommandPool {
    let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .build();
    unsafe {
        device
            .create_command_pool(&command_pool_create_info, None)
            .unwrap()
    }
}

fn create_command_buffers(
    device: &Device,
    command_pool: &vk::CommandPool,
    count: u32,
) -> Vec<vk::CommandBuffer> {
    let create_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(*command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(count);
    unsafe { device.allocate_command_buffers(&create_info).unwrap() }
}

fn create_default_render_pass(device: &Device, format: vk::Format) -> vk::RenderPass {
    let attachments = [
        // Color
        vk::AttachmentDescription::builder()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .build(),
        // Depth
        vk::AttachmentDescription::builder()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build(),
    ];

    let color_reference = [vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];
    let depth_reference = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .build();
    let subpasses = [vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_reference)
        .depth_stencil_attachment(&depth_reference)
        .build()];
    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .build();
    unsafe {
        device
            .create_render_pass(&render_pass_create_info, None)
            .unwrap()
    }
}

fn create_swapchain_image_views(
    device: &Device,
    swapchain_images: &[vk::Image],
    // swapchain_config: &SwapchainSupportDetails,
    format: vk::Format,
) -> Vec<vk::ImageView> {
    swapchain_images
        .iter()
        .map(|&image| {
            let image_view_create_info = vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                );
            unsafe {
                device
                    .create_image_view(&image_view_create_info, None)
                    .unwrap()
            }
        })
        .collect()
}

fn create_depth_image(
    device: &Device,
    allocator: &vk_mem::Allocator,
    extent: vk::Extent2D,
) -> (AllocatedImage, vk::ImageView) {
    let depth_image_create_info = vk::ImageCreateInfo::builder()
        .format(vk::Format::D32_SFLOAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .mip_levels(1)
        .array_layers(1)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .build();

    let depth_image_allocation_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::GpuOnly,
        flags: vk_mem::AllocationCreateFlags::empty(),
        required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        preferred_flags: vk::MemoryPropertyFlags::empty(),
        memory_type_bits: 0,
        pool: None,
        user_data: None,
    };

    let depth_allocated_image = AllocatedImage::new(
        allocator,
        depth_image_create_info,
        depth_image_allocation_info,
    );

    let depth_image_view = {
        let depth_image_view_create_info = vk::ImageViewCreateInfo::builder()
            .image(depth_allocated_image.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );
        unsafe {
            device
                .create_image_view(&depth_image_view_create_info, None)
                .unwrap()
        }
    };

    (depth_allocated_image, depth_image_view)
}

fn create_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    swapchain_image_views: &[vk::ImageView],
    depth_image_view: vk::ImageView,
    extent: vk::Extent2D,
    count: usize,
) -> Vec<vk::Framebuffer> {
    (0..count)
        .map(|i| {
            let attachments = [swapchain_image_views[i], depth_image_view];
            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            unsafe {
                device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .unwrap()
            }
        })
        .collect()
}
