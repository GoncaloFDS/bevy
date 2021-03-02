use std::{borrow::Cow, collections::HashMap, ffi::CString, ops::Range, sync::Arc};

use ash::{
    extensions::khr::{Surface, Swapchain, Win32Surface},
    version::DeviceV1_0,
    vk, Device, Entry, Instance,
};
use parking_lot::RwLock;

use bevy_asset::{Assets, Handle, HandleUntyped};
use bevy_render::{
    pipeline::{BindGroupDescriptorId, PipelineDescriptor},
    renderer::{
        BindGroup, BufferId, BufferInfo, BufferMapMode, RenderResourceContext, RenderResourceId,
        SamplerId, TextureId,
    },
    shader::{glsl_to_spirv, Shader, ShaderError, ShaderSource},
    texture::{SamplerDescriptor, TextureDescriptor, TextureFormat},
};
use bevy_utils::tracing::*;
use bevy_window::{Window, WindowId};
use bevy_winit::WinitWindows;

use crate::{
    renderer::{SwapchainProperties, SwapchainSupportDetails},
    vulkan_renderer::QueueFamiliesIndices,
};

pub const BIND_BUFFER_ALIGNMENT: usize = 256;
pub const TEXTURE_ALIGNMENT: usize = 256;

pub fn required_extension_names() -> Vec<*const i8> {
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}

#[derive(Clone)]
pub struct VulkanRenderResourceContext {
    entry: Arc<ash::Entry>,
    instance: Arc<ash::Instance>,
    pub device: Arc<ash::Device>,
    physical_device: Arc<vk::PhysicalDevice>,
    queue_indices: QueueFamiliesIndices,

    surface_loader: Arc<Surface>,

    window_surfaces: Arc<RwLock<HashMap<WindowId, vk::SurfaceKHR>>>,
    window_swap_chains: Arc<RwLock<HashMap<WindowId, vk::SwapchainKHR>>>,
    pub window_swap_chain_properties: Arc<RwLock<HashMap<WindowId, SwapchainProperties>>>,
    swapchain_loader: Arc<RwLock<Option<Swapchain>>>,

    swap_chain_images: Arc<RwLock<Vec<vk::Image>>>,
    pub swap_chain_image_views: Arc<RwLock<Vec<vk::ImageView>>>,

    shader_modules: Arc<RwLock<HashMap<Handle<Shader>, vk::ShaderModule>>>,

    pub render_pass: Arc<RwLock<vk::RenderPass>>,
    render_pipeline_layouts: Arc<RwLock<HashMap<Handle<PipelineDescriptor>, vk::PipelineLayout>>>,
    pub render_pipelines: Arc<RwLock<HashMap<Handle<PipelineDescriptor>, vk::Pipeline>>>,

    pub swapchain_frame_buffers: Arc<RwLock<Vec<vk::Framebuffer>>>,

    pub command_pool: Arc<RwLock<vk::CommandPool>>,
    pub command_buffers: Arc<RwLock<Vec<vk::CommandBuffer>>>,

    pub image_available_semaphore: Arc<RwLock<vk::Semaphore>>,
    pub render_finished_semaphore: Arc<RwLock<vk::Semaphore>>,

    pub asset_resources: Arc<RwLock<HashMap<(HandleUntyped, u64), RenderResourceId>>>,
}

impl VulkanRenderResourceContext {
    pub fn new(
        entry: Arc<Entry>,
        instance: Arc<Instance>,
        device: Arc<Device>,
        physical_device: Arc<vk::PhysicalDevice>,
        queue_indices: QueueFamiliesIndices,
    ) -> Self {
        let surface_loader = Surface::new(entry.as_ref(), instance.as_ref());
        let command_pool = Self::create_command_pool(device.as_ref());

        VulkanRenderResourceContext {
            entry,
            device,
            instance,
            physical_device,
            queue_indices,
            surface_loader: Arc::new(surface_loader),
            window_surfaces: Arc::new(Default::default()),
            window_swap_chains: Arc::new(Default::default()),
            window_swap_chain_properties: Arc::new(Default::default()),
            swapchain_loader: Arc::new(Default::default()),
            swap_chain_images: Arc::new(RwLock::new(vec![])),
            swap_chain_image_views: Arc::new(RwLock::new(vec![])),
            shader_modules: Arc::new(Default::default()),
            render_pass: Arc::new(Default::default()),
            render_pipeline_layouts: Arc::new(Default::default()),
            render_pipelines: Arc::new(Default::default()),
            swapchain_frame_buffers: Arc::new(Default::default()),
            command_pool: Arc::new(RwLock::new(command_pool)),
            command_buffers: Arc::new(Default::default()),
            image_available_semaphore: Arc::new(Default::default()),
            render_finished_semaphore: Arc::new(Default::default()),
            asset_resources: Arc::new(Default::default()),
        }
    }

    pub fn create_window_surface(&self, window_id: WindowId, winit_windows: &WinitWindows) {
        info!("Creating window");
        let winit_window = winit_windows.get_window(window_id).unwrap();
        let surface = unsafe {
            ash_window::create_surface(&*self.entry, &*self.instance, winit_window, None)
        };

        let mut window_surfaces = self.window_surfaces.write();
        window_surfaces.insert(window_id, surface.unwrap());
    }

    fn try_next_swap_chain_texture(&self, _window_id: bevy_window::WindowId) -> Option<TextureId> {
        // let mut window_swap_chains = self.window_swap_chains.write();
        // let mut swap_chain_outputs = self.swap_chain_images.write();
        Some(TextureId::new())
    }

    fn create_command_pool(device: &Device) -> vk::CommandPool {
        let graphics_family = 0;

        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(graphics_family)
            .flags(vk::CommandPoolCreateFlags::empty())
            .build();

        unsafe {
            device
                .create_command_pool(&command_pool_info, None)
                .unwrap()
        }
    }

    fn create_and_register_command_buffers(
        device: &Device,
        pool: vk::CommandPool,
        framebuffers: &[vk::Framebuffer],
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
        graphics_pipeline: vk::Pipeline,
    ) -> Vec<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as _)
            .build();

        info!("frame len {}", framebuffers.len());

        let buffers = unsafe { device.allocate_command_buffers(&allocate_info).unwrap() };

        buffers
            .iter()
            .zip(framebuffers.iter())
            .for_each(|(buffer, framebuffer)| {
                let buffer = *buffer;
                {
                    // Command Buffer
                    let command_buffer_info = vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)
                        .build();
                    unsafe {
                        device
                            .begin_command_buffer(buffer, &command_buffer_info)
                            .unwrap()
                    };
                }
                {
                    // Render Pass
                    let clear_values = [vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    }];
                    let render_area = vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    };
                    let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                        .render_pass(render_pass)
                        .framebuffer(*framebuffer)
                        .render_area(render_area)
                        .clear_values(&clear_values)
                        .build();

                    unsafe {
                        device.cmd_begin_render_pass(
                            buffer,
                            &render_pass_begin_info,
                            vk::SubpassContents::INLINE,
                        )
                    };
                }

                unsafe {
                    device.cmd_bind_pipeline(
                        buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        graphics_pipeline,
                    );
                    device.cmd_draw(buffer, 3, 1, 0, 0);
                    device.cmd_end_render_pass(buffer);
                    device.end_command_buffer(buffer).unwrap();
                }
            });

        buffers
    }

    fn create_semaphore(device: &Device) -> vk::Semaphore {
        let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
        unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
    }

    fn validate_physical_device_surface_support(&self, surface: &vk::SurfaceKHR) {
        let physical_device_surface_support = unsafe {
            self.surface_loader.get_physical_device_surface_support(
                *self.physical_device,
                self.queue_indices.present_index,
                *surface,
            )
        };
        if physical_device_surface_support.is_err() {
            panic!("{}", physical_device_surface_support.err().unwrap());
        }
    }

    pub fn begin_buffer(&mut self) {
        let mut command_buffers = VulkanRenderResourceContext::create_and_register_command_buffers(
            &self.device.as_ref(),
            *self.command_pool.read(),
            &self.swapchain_frame_buffers.read(),
            *self.render_pass.read(),
            vk::Extent2D {
                width: 800,
                height: 720,
            },
            *self.render_pipelines.write().iter().next().unwrap().1,
        );

        info!("command buffers {:?}", command_buffers);

        self.command_buffers.write().append(&mut command_buffers);

        *self.image_available_semaphore.write() =
            VulkanRenderResourceContext::create_semaphore(self.device.as_ref());
        *self.render_finished_semaphore.write() =
            VulkanRenderResourceContext::create_semaphore(self.device.as_ref());
    }
}

impl RenderResourceContext for VulkanRenderResourceContext {
    fn create_swap_chain(&self, window: &Window) {
        let surfaces = self.window_surfaces.read();
        let surface = surfaces
            .get(&window.id())
            .expect("No Surface found for window");

        let details = SwapchainSupportDetails::new(
            *self.physical_device,
            self.surface_loader.as_ref(),
            *surface,
        );
        let properties =
            details.get_ideal_swapchain_properties([window.height() as u32, window.width() as u32]);

        let image_count = {
            let max = details.capabilities.max_image_count;
            let mut preferred = details.capabilities.min_image_count + 1;
            if max > 0 && preferred > max {
                preferred = max;
            }
            preferred
        };

        info!("Creating swapchain {:?}", properties);

        let family_indices = [
            self.queue_indices.graphics_index,
            self.queue_indices.present_index,
        ];

        self.validate_physical_device_surface_support(surface);

        let swapchain_create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::builder()
                .surface(*surface)
                .min_image_count(image_count)
                .image_format(properties.format.format)
                .image_color_space(properties.format.color_space)
                .image_extent(properties.extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

            builder = if self.queue_indices.graphics_index != self.queue_indices.present_index {
                builder
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&family_indices)
            } else {
                builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

            builder
                .pre_transform(details.capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(properties.present_mode)
                .clipped(true)
                .build()
        };

        let swapchain_loader = Swapchain::new(&*self.instance, self.device.as_ref());
        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap()
        };

        self.window_swap_chain_properties
            .write()
            .insert(window.id(), properties);
        self.window_swap_chains
            .write()
            .insert(window.id(), swapchain);

        let mut swap_chain_images = self.swap_chain_images.write();
        let mut images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };
        swap_chain_images.clear();
        swap_chain_images.append(&mut images);
        let mut image_views = create_swapchain_image_views(
            self.device.as_ref(),
            &swap_chain_images,
            properties.format.format,
        );
        info!("images {:?}", image_views);
        let mut swap_chain_image_views = self.swap_chain_image_views.write();
        swap_chain_image_views.append(&mut image_views);
        info!("images {:?}", swap_chain_image_views);
        *self.swapchain_loader.write() = Some(swapchain_loader);
    }

    fn next_swap_chain_texture(&self, window: &Window) -> TextureId {
        if let Some(texture_id) = self.try_next_swap_chain_texture(window.id()) {
            texture_id
        } else {
            self.window_swap_chains.write().remove(&window.id());
            self.create_swap_chain(window);
            self.try_next_swap_chain_texture(window.id())
                .expect("Failed to acquire next swap chain texture!")
        }
    }

    fn drop_swap_chain_texture(&self, render_resource: TextureId) {
        unimplemented!()
    }

    fn drop_all_swap_chain_textures(&self) {
        let mut swap_chain_outputs = self.swap_chain_image_views.write();
        swap_chain_outputs.clear();
    }

    fn create_sampler(&self, _sampler_descriptor: &SamplerDescriptor) -> SamplerId {
        SamplerId::new()
    }

    fn create_texture(&self, _texture_descriptor: TextureDescriptor) -> TextureId {
        TextureId::new()
    }

    fn create_buffer(&self, _buffer_info: BufferInfo) -> BufferId {
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

    fn map_buffer(&self, _id: BufferId, _mode: BufferMapMode) {}

    fn unmap_buffer(&self, _id: BufferId) {}

    fn create_buffer_with_data(&self, _buffer_info: BufferInfo, _data: &[u8]) -> BufferId {
        BufferId::new()
    }

    fn create_shader_module(&self, shader_handle: &Handle<Shader>, shaders: &Assets<Shader>) {
        if self.shader_modules.read().get(&shader_handle).is_some() {
            return;
        }
        let shader = shaders.get(shader_handle).unwrap();
        info!("Loading shader {:?}", shader_handle.id);
        self.create_shader_module_from_source(shader_handle, shader)
    }

    fn create_shader_module_from_source(&self, shader_handle: &Handle<Shader>, shader: &Shader) {
        let mut shader_modules = self.shader_modules.write();
        let spirv: Cow<[u32]> = shader.get_spirv(None).unwrap().into();
        let create_info = vk::ShaderModuleCreateInfo::builder().code(&spirv).build();
        let shader_module = unsafe {
            self.device
                .as_ref()
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

    fn remove_buffer(&self, _buffer: BufferId) {}

    fn remove_texture(&self, _texture: TextureId) {}

    fn remove_sampler(&self, _sampler: SamplerId) {}

    fn get_buffer_info(&self, _buffer: BufferId) -> Option<BufferInfo> {
        unimplemented!()
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
        let mut asset_resources = self.asset_resources.write();
        asset_resources.insert((handle, index), render_resource);
    }

    fn get_asset_resource_untyped(
        &self,
        handle: HandleUntyped,
        index: u64,
    ) -> Option<RenderResourceId> {
        let asset_resources = self.asset_resources.read();
        asset_resources.get(&(handle, index)).cloned()
    }

    fn remove_asset_resource_untyped(&self, _handle: HandleUntyped, _index: u64) {}

    fn create_render_pipeline(
        &self,
        pipeline_handle: Handle<PipelineDescriptor>,
        pipeline_descriptor: &PipelineDescriptor,
        shaders: &Assets<Shader>,
    ) {
        info!("Creating Render Pipeline");
        // render pass
        let format = match pipeline_descriptor.color_target_states[0].format {
            TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_UNORM,
            _ => unimplemented!(),
        };
        let attachment_desc = vk::AttachmentDescription::builder()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .build();
        let attachment_descs = [attachment_desc];

        let attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let attachment_refs = [attachment_ref];

        let subpass_desc = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&attachment_refs)
            .build();
        let subpass_descs = [subpass_desc];

        let subpass_dep = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .build();
        let subpass_deps = [subpass_dep];

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment_descs)
            .subpasses(&subpass_descs)
            .dependencies(&subpass_deps)
            .build();

        let render_pass = unsafe {
            self.device
                .as_ref()
                .create_render_pass(&render_pass_info, None)
                .unwrap()
        };
        *self.render_pass.write() = render_pass;
        // pipeline

        self.create_shader_module(&pipeline_descriptor.shader_stages.vertex, shaders);
        if let Some(ref fragment_handle) = pipeline_descriptor.shader_stages.fragment {
            self.create_shader_module(fragment_handle, shaders);
        }

        let entry_point_name = CString::new("main").unwrap();
        let shader_modules = self.shader_modules.read();
        let mut shader_state_infos = vec![];

        let vertex_shader_module = shader_modules
            .get(&pipeline_descriptor.shader_stages.vertex)
            .unwrap();
        let vertex_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(*vertex_shader_module)
            .name(&entry_point_name)
            .build();
        shader_state_infos.push(vertex_shader_state_info);

        if let Some(ref fragment_handle) = pipeline_descriptor.shader_stages.fragment {
            let fragment_shader_module = shader_modules.get(fragment_handle).unwrap();
            let fragment_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(*fragment_shader_module)
                .name(&entry_point_name)
                .build();
            shader_state_infos.push(fragment_shader_state_info);
        };

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder().build();
        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false)
            .build();
        let s = self.window_swap_chain_properties.read();
        let swapchain_properties = s.iter().next();
        let extent = match swapchain_properties {
            Some((_, properties)) => properties.extent,
            None => vk::Extent2D {
                width: 800,
                height: 720,
            },
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as _,
            height: extent.height as _,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let viewports = [viewport];
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        let scissors = [scissor];
        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors)
            .build();

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0)
            .build();

        let multisampling_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .build();

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .build();
        let color_blend_attachments = [color_blend_attachment];

        let color_blending_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0])
            .build();

        let render_pipeline_layout = {
            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
                //.set_layouts
                //.push_constant_ranges
                .build();

            unsafe {
                self.device
                    .as_ref()
                    .create_pipeline_layout(&pipeline_layout_info, None)
                    .unwrap()
            }
        };
        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_state_infos)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampling_info)
            .color_blend_state(&color_blending_info)
            .layout(render_pipeline_layout)
            .render_pass(render_pass)
            .subpass(0)
            .build();
        let pipeline_infos = [pipeline_info];

        let render_pipeline = unsafe {
            self.device
                .as_ref()
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
                .unwrap()[0]
        };
        let mut render_pipelines = self.render_pipelines.write();
        render_pipelines.insert(pipeline_handle.clone(), render_pipeline);
        let mut render_pipeline_layouts = self.render_pipeline_layouts.write();
        render_pipeline_layouts.insert(pipeline_handle, render_pipeline_layout);
    }

    fn bind_group_descriptor_exists(
        &self,
        _bind_group_descriptor_id: BindGroupDescriptorId,
    ) -> bool {
        true
    }

    fn create_bind_group(
        &self,
        _bind_group_descriptor_id: BindGroupDescriptorId,
        _bind_group: &BindGroup,
    ) {
    }

    fn clear_bind_groups(&self) {}

    fn remove_stale_bind_groups(&self) {}
}

fn create_swapchain_image_views(
    device: &Device,
    swapchain_images: &[vk::Image],
    swapchain_format: vk::Format,
) -> Vec<vk::ImageView> {
    swapchain_images
        .iter()
        .map(|image| {
            let create_info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(swapchain_format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
            unsafe { device.create_image_view(&create_info, None).unwrap() }
        })
        .collect::<Vec<_>>()
}

impl Drop for VulkanRenderResourceContext {
    fn drop(&mut self) {
        unsafe {
            if Arc::strong_count(&self.render_finished_semaphore) == 1 {
                self.device
                    .destroy_semaphore(*self.render_finished_semaphore.write(), None);
                self.device
                    .destroy_semaphore(*self.image_available_semaphore.write(), None);
                self.device
                    .destroy_command_pool(*self.command_pool.write(), None);
                self.swapchain_frame_buffers
                    .write()
                    .iter()
                    .for_each(|buffer| self.device.destroy_framebuffer(*buffer, None));
                self.render_pipelines
                    .write()
                    .iter()
                    .for_each(|(_, pipeline)| self.device.destroy_pipeline(*pipeline, None));
                self.render_pipeline_layouts
                    .write()
                    .iter()
                    .for_each(|(_, layout)| self.device.destroy_pipeline_layout(*layout, None));
                self.device
                    .destroy_render_pass(*self.render_pass.write(), None);
                self.swap_chain_image_views
                    .write()
                    .iter()
                    .for_each(|image_view| self.device.destroy_image_view(*image_view, None));
                self.window_swap_chains
                    .write()
                    .iter()
                    .for_each(|(_, swapchain)| {
                        self.swapchain_loader
                            .write()
                            .as_ref()
                            .unwrap()
                            .destroy_swapchain(*swapchain, None)
                    });
                self.window_surfaces
                    .write()
                    .iter()
                    .for_each(|(_, surface)| {
                        self.surface_loader.destroy_surface(*surface, None);
                    });
            }
        }
    }
}
