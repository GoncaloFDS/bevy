use std::{ops::Range, sync::Arc};
use std::borrow::Cow;
use std::ffi::CString;

use ash::{
    Device,
    Entry, extensions::khr::{Surface, Win32Surface}, Instance, vk,
};
use ash::extensions::khr::Swapchain;
use ash::version::DeviceV1_0;

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
use bevy_render::pipeline::BindGroupDescriptor;
use bevy_utils::tracing::*;
use bevy_window::{Window, WindowId};

use crate::{QueueFamiliesIndices, vulkan_resources::VulkanResources, VulkanRenderer};
use crate::vulkan_type_converter::VulkanInto;
use crate::vulkan_types::{AllocatedImage, SwapchainDescriptor};

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
    pub queue_indices: QueueFamiliesIndices,
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
        let queue_indices = renderer.queue_indices;

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
            queue_indices,
            command_pool,
            frame_sync,
            resources,
        }
    }

    pub fn set_window_surface(&self, window_id: WindowId, surface: vk::SurfaceKHR) {
        let mut window_surfaces = self.resources.window_surfaces.write();
        window_surfaces.insert(window_id, surface);
    }

    fn create_bind_group_layout(&self, _descriptor: &BindGroupDescriptor) {}

    fn try_next_swapchain_texture(&self, _window_id: bevy_window::WindowId) -> Option<TextureId> {
        // let mut window_swapchains = self.resources.window_swapchains.write();
        // let mut swapchains_frames = self.resources.swapchain_image_views.write();
        //
        // let mut window_swapchain = window_swapchains.get_mut(&window_id).unwrap();
        None
    }
}

impl RenderResourceContext for VulkanRenderResourceContext {
    fn create_swap_chain(&self, window: &Window) {
        let surfaces = self.resources.window_surfaces.read();
        let mut window_swapchains = self.resources.window_swapchains.write();

        let swapchain_descriptor: SwapchainDescriptor = window.vulkan_into();
        let surface = surfaces
            .get(&window.id())
            .expect("No surface found for window.");

        let device_supports_surface = unsafe {
            self.surface_loader
                .get_physical_device_surface_support(
                    *self.physical_device,
                    self.queue_indices.graphics_index,
                    *surface,
                )
                .unwrap()
        };

        if !device_supports_surface {
            panic!("Device does not support surface")
        }

        let swapchain = create_swapchain(
            &self.swapchain_loader,
            *surface,
            &swapchain_descriptor,
            None,
        );


        window_swapchains.insert(window.id(), swapchain);
    }

    fn next_swap_chain_texture(&self, window: &Window) -> TextureId {
        if let Some(texture_id) = self.try_next_swapchain_texture(window.id()) {
            texture_id
        } else {
            TextureId::new()
        }
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
        let mut textures = self.resources.textures.write();

        let id = TextureId::new();
        let texture = id;

        textures.insert(id, texture);
        id
    }

    fn create_buffer(&self, buffer_info: BufferInfo) -> BufferId {
        let mut buffer_infos = self.resources.buffer_infos.write();
        let id = BufferId::new();
        buffer_infos.insert(id, buffer_info);
        id
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

    fn create_buffer_with_data(&self, buffer_info: BufferInfo, _data: &[u8]) -> BufferId {
        let mut buffer_infos = self.resources.buffer_infos.write();
        let id = BufferId::new();
        buffer_infos.insert(id, buffer_info);
        id
    }

    fn create_shader_module(&self, shader_handle: &Handle<Shader>, shaders: &Assets<Shader>) {
        if self
            .resources
            .shader_modules
            .read()
            .get(&shader_handle)
            .is_some()
        {
            return;
        }
        let shader = shaders.get(shader_handle).unwrap();
        self.create_shader_module_from_source(shader_handle, shader);
    }

    fn create_shader_module_from_source(&self, shader_handle: &Handle<Shader>, shader: &Shader) {
        let mut shader_modules = self.resources.shader_modules.write();
        let spirv: Cow<[u32]> = shader.get_spirv(None).unwrap().into();
        let create_info = vk::ShaderModuleCreateInfo::builder().code(&spirv);
        let shader_module = unsafe {
            self.device
                .create_shader_module(&create_info, None)
                .unwrap()
        };
        shader_modules.insert(shader_handle.clone_weak(), shader_module);
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

    fn remove_buffer(&self, buffer: BufferId) {
        let mut buffer_infos = self.resources.buffer_infos.write();
        // let mut buffers = self.resources.buffers.write();
        //
        // buffers.remove(&buffer);
        buffer_infos.remove(&buffer);
    }

    fn remove_texture(&self, texture: TextureId) {
        let mut textures = self.resources.textures.write();

        textures.remove(&texture);
    }

    fn remove_sampler(&self, _sampler: SamplerId) {
        todo!()
    }

    fn get_buffer_info(&self, buffer: BufferId) -> Option<BufferInfo> {
        self.resources.buffer_infos.read().get(&buffer).cloned()
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
        handle: HandleUntyped,
        render_resource: RenderResourceId,
        index: u64,
    ) {
        let mut asset_resources = self.resources.asset_resources.write();
        asset_resources.insert((handle, index), render_resource);
    }

    fn get_asset_resource_untyped(
        &self,
        handle: HandleUntyped,
        index: u64,
    ) -> Option<RenderResourceId> {
        let asset_resources = self.resources.asset_resources.read();
        asset_resources.get(&(handle, index)).cloned()
    }

    fn remove_asset_resource_untyped(&self, handle: HandleUntyped, index: u64) {
        let mut asset_resources = self.resources.asset_resources.write();
        asset_resources.remove(&(handle, index));
    }

    fn create_render_pipeline(
        &self,
        pipeline_handle: Handle<PipelineDescriptor>,
        pipeline_descriptor: &PipelineDescriptor,
        shaders: &Assets<Shader>,
    ) {
        if self
            .resources
            .render_pipelines
            .read()
            .get(&pipeline_handle)
            .is_some()
        {
            return;
        }

        let layout = pipeline_descriptor.get_layout().unwrap();
        for bind_group_descriptor in layout.bind_groups.iter() {
            self.create_bind_group_layout(&bind_group_descriptor);
        }

        self.create_shader_module(&pipeline_descriptor.shader_stages.vertex, shaders);

        if let Some(ref fragmente_handle) = pipeline_descriptor.shader_stages.fragment {
            self.create_shader_module(fragmente_handle, shaders)
        }

        let shader_modules = self.resources.shader_modules.read();
        let vertex_shader_module = shader_modules
            .get(&pipeline_descriptor.shader_stages.vertex)
            .unwrap();

        let fragment_shader_module = pipeline_descriptor
            .shader_stages
            .fragment
            .as_ref()
            .map(|fragment_handle| shader_modules.get(fragment_handle).unwrap());

        let shader_entry_point = CString::new("main").unwrap();
        let pipeline_shader_stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(*vertex_shader_module)
                .name(&shader_entry_point)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(*fragment_shader_module.unwrap())
                .name(&shader_entry_point)
                .build(),
        ];

        let pipeline_layout = {
            let create_info = vk::PipelineLayoutCreateInfo::builder().build();
            unsafe { self.device.create_pipeline_layout(&create_info, None) }.unwrap()
        };

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);

        let stencil_op = vk::StencilOpState::builder()
            .fail_op(vk::StencilOp::KEEP)
            .pass_op(vk::StencilOp::KEEP)
            .compare_op(vk::CompareOp::ALWAYS)
            .build();
        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .front(stencil_op)
            .back(stencil_op);
        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build()];
        let color_blend_info =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&color_blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let mut render_pass = self.resources.render_passes.write();
        *render_pass = create_default_render_pass(&self.device, vk::Format::B8G8R8A8_SRGB);

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&pipeline_shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blend_info)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .render_pass(*render_pass)
            .subpass(0)
            .build();

        let graphics_pipeline = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .unwrap()[0]
        };

        let mut graphics_pipelines = self.resources.render_pipelines.write();
        graphics_pipelines.insert(pipeline_handle, graphics_pipeline);
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
    swapchain_descriptor: &SwapchainDescriptor,
    old_swapchain: Option<vk::SwapchainKHR>,
) -> vk::SwapchainKHR {
    let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(swapchain_descriptor.frames_in_flight)
        .image_format(swapchain_descriptor.format)
        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .image_extent(swapchain_descriptor.extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(swapchain_descriptor.present_mode)
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
