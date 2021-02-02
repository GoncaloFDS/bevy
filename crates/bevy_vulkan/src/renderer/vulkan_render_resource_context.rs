use crate::{
    debug::*,
    renderer::{SwapchainProperties, SwapchainSupportDetails},
};
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain, Win32Surface},
    },
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk,
    vk::PhysicalDevice,
    Device, Entry, Instance,
};
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
use parking_lot::RwLock;
use std::{
    borrow::Cow,
    collections::HashMap,
    ffi::{CStr, CString},
    ops::Range,
    sync::Arc,
};

#[derive(Clone, Copy, Default)]
struct QueueFamiliesIndices {
    graphics_index: u32,
    present_index: u32,
}

pub const BIND_BUFFER_ALIGNMENT: usize = 256;
pub const TEXTURE_ALIGNMENT: usize = 256;

pub fn required_extension_names() -> Vec<*const i8> {
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}

#[derive(Clone)]
pub struct VulkanRenderResourceContext {
    pub entry: Arc<ash::Entry>,
    pub device: Arc<RwLock<Option<ash::Device>>>,
    pub instance: Arc<ash::Instance>,
    pub surface_loader: Arc<Surface>,

    pub physical_device: Arc<RwLock<Option<vk::PhysicalDevice>>>,

    pub debug_utils: Option<(DebugUtils, vk::DebugUtilsMessengerEXT)>,

    pub window_surfaces: Arc<RwLock<HashMap<WindowId, vk::SurfaceKHR>>>,
    pub window_swap_chains: Arc<RwLock<HashMap<WindowId, vk::SwapchainKHR>>>,
    pub window_swap_chain_properties: Arc<RwLock<HashMap<WindowId, SwapchainProperties>>>,
    pub swapchain_loader: Option<Swapchain>,
    pub graphics_queue: Arc<RwLock<Option<vk::Queue>>>,
    pub present_queue: Arc<RwLock<Option<vk::Queue>>>,

    pub swap_chain_images: Arc<RwLock<Vec<vk::Image>>>,
    pub swap_chain_image_views: Arc<RwLock<Vec<vk::ImageView>>>,

    pub shader_modules: Arc<RwLock<HashMap<Handle<Shader>, vk::ShaderModule>>>,

    pub render_pass: Arc<RwLock<vk::RenderPass>>,
    pub render_pipeline_layouts:
        Arc<RwLock<HashMap<Handle<PipelineDescriptor>, vk::PipelineLayout>>>,
    pub render_pipelines: Arc<RwLock<HashMap<Handle<PipelineDescriptor>, vk::Pipeline>>>,
    queue_family_indices: Arc<RwLock<QueueFamiliesIndices>>,
}

impl VulkanRenderResourceContext {
    pub fn new() -> Self {
        info!("Creating VulkanRenderResourceContext");
        let entry: Entry = ash::Entry::new().expect("Failed to create entry.");
        let instance = Self::create_instance(&entry);
        let surface_loader = Surface::new(&entry, &instance);

        let debug_utils = setup_debug_messenger(&entry, &instance);

        let (physical_device, device) =
            VulkanRenderResourceContext::create_default_devices(&instance);

        VulkanRenderResourceContext {
            entry: Arc::new(entry),
            device: Arc::new(RwLock::new(Some(device))),
            instance: Arc::new(instance),
            surface_loader: Arc::new(surface_loader),
            physical_device: Arc::new(RwLock::new(Some(physical_device))),
            debug_utils,
            window_surfaces: Arc::new(Default::default()),
            window_swap_chains: Arc::new(Default::default()),
            window_swap_chain_properties: Arc::new(Default::default()),
            swapchain_loader: None,
            graphics_queue: Arc::new(RwLock::new(None)),
            present_queue: Arc::new(RwLock::new(None)),
            queue_family_indices: Arc::new(RwLock::new(QueueFamiliesIndices::default())),
            swap_chain_images: Arc::new(RwLock::new(vec![])),
            swap_chain_image_views: Arc::new(RwLock::new(vec![])),
            shader_modules: Arc::new(Default::default()),
            render_pass: Arc::new(Default::default()),
            render_pipeline_layouts: Arc::new(Default::default()),
            render_pipelines: Arc::new(Default::default()),
        }
    }

    fn create_default_devices(instance: &Instance) -> (PhysicalDevice, Device) {
        let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let physical_device = physical_devices[0];
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        info!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });

        let device_extensions = Self::get_required_device_extensions();
        let device_extension_ptrs = device_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let device_features = vk::PhysicalDeviceFeatures::builder().build();

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        let queue_create_infos = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(0)
            .queue_priorities(&[1.0])
            .build()];

        let mut device_create_info_builder = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extension_ptrs)
            .enabled_features(&device_features);

        if ENABLE_VALIDATION_LAYERS {
            device_create_info_builder =
                device_create_info_builder.enabled_layer_names(&layer_names_ptrs)
        }
        let device_create_info = device_create_info_builder.build();

        let device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create logical device")
        };
        (physical_device, device)
    }

    pub fn create_window_surface(&self, window_id: WindowId, winit_windows: &WinitWindows) {
        info!("Creating window");
        let winit_window = winit_windows.get_window(window_id).unwrap();
        let surface = unsafe {
            ash_window::create_surface(&*self.entry, &*self.instance, winit_window, None)
        };

        let mut window_surfaces = self.window_surfaces.write();
        window_surfaces.insert(window_id, surface.unwrap());

        self.init_with_surface(surface.unwrap());
    }

    pub fn init_with_surface(&self, surface: vk::SurfaceKHR) {
        let (physical_device, queue_family_indices) =
            Self::pick_physical_device(&*self.instance, &*self.surface_loader, surface);
        let (device, graphics_queue, present_queue) =
            Self::create_logical_device_with_graphics_queue(
                &*self.instance,
                physical_device,
                queue_family_indices,
            );

        *self.physical_device.write() = Some(physical_device);
        *self.device.write() = Some(device);
        *self.queue_family_indices.write() = queue_family_indices;
        *self.graphics_queue.write() = Some(graphics_queue);
        *self.present_queue.write() = Some(present_queue);
    }

    fn try_next_swap_chain_texture(&self, _window_id: bevy_window::WindowId) -> Option<TextureId> {
        // let mut window_swap_chains = self.window_swap_chains.write();
        // let mut swap_chain_outputs = self.swap_chain_images.write();
        Some(TextureId::new())
    }

    fn create_instance(entry: &Entry) -> Instance {
        let app_name = CString::new("Vulkan App").unwrap();
        let engine_name = CString::new("Light Engine").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name.as_c_str())
            .application_version(vk::make_version(0, 1, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(vk::make_version(0, 1, 0))
            .api_version(vk::make_version(1, 2, 0))
            .build();

        let mut extension_names = required_extension_names();
        if ENABLE_VALIDATION_LAYERS {
            extension_names.push(DebugUtils::name().as_ptr());
        }

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        let mut instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        if ENABLE_VALIDATION_LAYERS {
            check_validation_layer_support(&entry);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_names_ptrs)
        }

        unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .expect("Failed to create Instance")
        }
    }

    fn pick_physical_device(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
    ) -> (vk::PhysicalDevice, QueueFamiliesIndices) {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let device = devices
            .into_iter()
            .find(|device| Self::is_device_suitable(instance, surface, surface_khr, *device))
            .expect("No suitable physical devices");

        let props = unsafe { instance.get_physical_device_properties(device) };
        info!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });
        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        let queue_family_indices = QueueFamiliesIndices {
            graphics_index: graphics.unwrap(),
            present_index: present.unwrap(),
        };

        (device, queue_family_indices)
    }

    fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice) -> bool {
        let required_extensions = Self::get_required_device_extensions();
        let extension_props = unsafe {
            instance
                .enumerate_device_extension_properties(device)
                .unwrap()
        };

        required_extensions.iter().all(|required| {
            extension_props.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                required == &name
            })
        })
    }

    fn is_device_suitable(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> bool {
        let (graphics, present) = Self::find_queue_families(instance, surface, surface_khr, device);
        let extension_support = Self::check_device_extension_support(instance, device);
        let is_swapchain_valid = {
            let details = SwapchainSupportDetails::new(device, surface, surface_khr);
            !details.formats.is_empty() && !details.present_modes.is_empty()
        };
        let features = unsafe { instance.get_physical_device_features(device) };
        graphics.is_some()
            && present.is_some()
            && extension_support
            && is_swapchain_valid
            && features.sampler_anisotropy == vk::TRUE
    }

    /// Find a queue family with at least one graphics queue and one with
    /// at least one presentation queue from `device`.
    ///
    /// #Returns
    ///
    /// Return a tuple (Option<graphics_family_index>, Option<present_family_index>).
    fn find_queue_families(
        instance: &Instance,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> (Option<u32>, Option<u32>) {
        let mut graphics = None;
        let mut present = None;
        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
            let index = index as u32;
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index);
            }

            let present_support =
                unsafe { surface.get_physical_device_surface_support(device, index, surface_khr) };
            if present_support.is_ok() && present.is_none() {
                present = Some(index);
            }

            if graphics.is_some() && present.is_some() {
                break;
            }
        }
        (graphics, present)
    }

    fn get_required_device_extensions() -> [&'static CStr; 1] {
        [Swapchain::name()]
    }

    /// Create the logical device to interact with `device`, a graphics queue
    /// and a presentation queue.
    ///
    /// # Returns
    ///
    /// Return a tuple containing the logical device, the graphics queue and the presentation queue.
    fn create_logical_device_with_graphics_queue(
        instance: &Instance,
        device: vk::PhysicalDevice,
        queue_families_indices: QueueFamiliesIndices,
    ) -> (Device, vk::Queue, vk::Queue) {
        let graphics_family_index = queue_families_indices.graphics_index;
        let present_family_index = queue_families_indices.present_index;
        let queue_priorities = [1.0f32];

        let queue_create_infos = {
            // Vulkan specs does not allow passing an array containing duplicated family indices.
            // And since the family for graphics and presentation could be the same we need to
            // deduplicate it.
            let mut indices = vec![graphics_family_index, present_family_index];
            indices.dedup();

            // Now we build an array of `DeviceQueueCreateInfo`.
            // One for each different family index.
            indices
                .iter()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(*index)
                        .queue_priorities(&queue_priorities)
                        .build()
                })
                .collect::<Vec<_>>()
        };

        let device_extensions = Self::get_required_device_extensions();
        let device_extension_ptrs = device_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let device_features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .build();

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        let mut device_create_info_builder = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extension_ptrs)
            .enabled_features(&device_features);

        if ENABLE_VALIDATION_LAYERS {
            device_create_info_builder =
                device_create_info_builder.enabled_layer_names(&layer_names_ptrs)
        }
        let device_create_info = device_create_info_builder.build();

        let device = unsafe {
            instance
                .create_device(device, &device_create_info, None)
                .expect("Failed to create logical device")
        };

        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_family_index, 0) };

        (device, graphics_queue, present_queue)
    }
}

impl RenderResourceContext for VulkanRenderResourceContext {
    fn create_swap_chain(&self, window: &Window) {
        let surfaces = self.window_surfaces.read();
        let surface = surfaces
            .get(&window.id())
            .expect("No Surface found for window");

        let details = SwapchainSupportDetails::new(
            self.physical_device.read().unwrap(),
            &*self.surface_loader,
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

        info!(
            "Creating swapchain\n\tFormat: {:?}\n\tColorSpace: {:?}\n\tPresentMode: {:?}\n\tExtent: {:?}\n\tImageCount: {}",
            properties.format.format,
            properties.format.color_space,
            properties.present_mode,
            properties.extent,
            image_count,
        );

        let queue_family_indices = *self.queue_family_indices.read();
        let family_indices = [
            queue_family_indices.graphics_index,
            queue_family_indices.present_index,
        ];

        let swapchain_create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::builder()
                .surface(*surface)
                .min_image_count(image_count)
                .image_format(properties.format.format)
                .image_color_space(properties.format.color_space)
                .image_extent(properties.extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

            builder = if queue_family_indices.graphics_index != queue_family_indices.present_index {
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

        let swapchain_loader = Swapchain::new(
            &*self.instance,
            self.device.read().as_ref().expect("Invalid Device"),
        );
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
        swap_chain_images.append(&mut images);
        let mut image_views = create_swapchain_image_views(
            self.device.read().as_ref().unwrap(),
            &swap_chain_images,
            properties.format.format,
        );
        let mut swap_chain_image_views = self.swap_chain_image_views.write();
        swap_chain_image_views.append(&mut image_views);
        println!("{:?}", swap_chain_images);
        println!("{:?}", swap_chain_image_views);
        println!("{:?}", image_views);
        info!("create swap chain");
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

    fn drop_swap_chain_texture(&self, _render_resource: TextureId) {}

    fn drop_all_swap_chain_textures(&self) {}

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
    }

    fn read_mapped_buffer(
        &self,
        _id: BufferId,
        _range: Range<u64>,
        _read: &dyn Fn(&[u8], &dyn RenderResourceContext),
    ) {
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
                .read()
                .as_ref()
                .unwrap()
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
        None
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
    }

    fn get_asset_resource_untyped(
        &self,
        _handle: HandleUntyped,
        _index: u64,
    ) -> Option<RenderResourceId> {
        None
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

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment_descs)
            .subpasses(&subpass_descs)
            .build();

        let render_pass = unsafe {
            self.device
                .read()
                .as_ref()
                .unwrap()
                .create_render_pass(&render_pass_info, None)
                .unwrap()
        };
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
                height: 800,
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
                    .read()
                    .as_ref()
                    .unwrap()
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
                .read()
                .as_ref()
                .unwrap()
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
        false
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
