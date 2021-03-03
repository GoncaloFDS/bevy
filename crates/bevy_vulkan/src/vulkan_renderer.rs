use std::{
    ffi::{CStr, CString},
    sync::Arc,
};

use ash::{
    Device,
    Entry,
    extensions::{ext::DebugUtils, khr::Swapchain}, Instance, version::{DeviceV1_0, EntryV1_0, InstanceV1_0}, vk,
};

use bevy_app::{ManualEventReader, prelude::*};
use bevy_ecs::{Resources, World};
use bevy_render::{
    render_graph::{DependentNodeStager, RenderGraph, RenderGraphStager},
    renderer::RenderResourceContext,
};
use bevy_utils::tracing::*;
use bevy_window::{WindowCreated, WindowResized, Windows};

use crate::{
    debug,
    debug::{
        check_validation_layer_support, ENABLE_VALIDATION_LAYERS, get_layer_names_and_pointers,
    },
    renderer::*,
};

pub struct VulkanRenderer {
    pub entry: Arc<ash::Entry>,
    pub instance: Arc<ash::Instance>,
    pub device: Arc<ash::Device>,
    pub physical_device: Arc<vk::PhysicalDevice>,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub queue_indices: QueueFamiliesIndices,

    debug_utils: Option<(DebugUtils, vk::DebugUtilsMessengerEXT)>,

    pub window_resized_event_reader: ManualEventReader<WindowResized>,
    pub window_created_event_reader: ManualEventReader<WindowCreated>,
}

impl VulkanRenderer {
    pub fn new() -> Self {
        let entry: Entry = ash::Entry::new().expect("Failed to create entry.");
        let instance = Self::create_instance(&entry);
        let debug_utils = debug::setup_debug_messenger(&entry, &instance);
        let (physical_device, queue_indices) = Self::pick_physical_device(&instance);
        let (device, graphics_queue, present_queue) =
            Self::create_logical_device_with_graphics_queue(
                &instance,
                physical_device,
                queue_indices,
            );

        VulkanRenderer {
            entry: Arc::new(entry),
            instance: Arc::new(instance),
            device: Arc::new(device),
            physical_device: Arc::new(physical_device),
            graphics_queue,
            present_queue,
            queue_indices,
            debug_utils,
            window_resized_event_reader: Default::default(),
            window_created_event_reader: Default::default(),
        }
    }

    pub fn handle_window_create_events(&mut self, resources: &Resources) {
        let mut render_resource_context = resources
            .get_mut::<Box<dyn RenderResourceContext>>()
            .unwrap();
        let render_resource_context = render_resource_context
            .downcast_mut::<VulkanRenderResourceContext>()
            .unwrap();
        let windows = resources.get::<Windows>().unwrap();
        let window_created_events = resources.get::<Events<WindowCreated>>().unwrap();
        for window_created_event in self
            .window_created_event_reader
            .iter(&window_created_events)
        {
            let window = windows
                .get(window_created_event.id)
                .expect("Received window created event for non-existing window.");
            let winit_windows = resources.get::<bevy_winit::WinitWindows>().unwrap();
            let winit_window = winit_windows.get_window(window.id()).unwrap();
            let surface = unsafe {
                ash_window::create_surface(self.entry.as_ref(), self.instance.as_ref(), winit_window, None)
            };
            render_resource_context.set_window_surface(window.id(), surface.unwrap());

            info!("handled window create event");
        }
    }

    pub fn run_graph(&mut self, world: &mut World, resources: &mut Resources) {
        let mut render_graph = resources.get_mut::<RenderGraph>().unwrap();
        //stage nodes
        let mut stager = DependentNodeStager::loose_grouping();
        let stages = stager.get_stages(&render_graph).unwrap();
        let mut borrowed = stages.borrow(&mut render_graph);

        // execute stages
        let graph_executor = VulkanRenderGraphExecutor {
            max_thread_count: 1,
        };
        graph_executor.execute(
            world,
            resources,
            self.device.clone(),
            &mut self.graphics_queue,
            &mut borrowed,
        );
    }

    pub fn update(&mut self, world: &mut World, resources: &mut Resources) {
        self.handle_window_create_events(resources);
        self.run_graph(world, resources);

        let render_resource_context = resources.get::<Box<dyn RenderResourceContext>>().unwrap();
        render_resource_context.drop_all_swap_chain_textures();
        render_resource_context.remove_stale_bind_groups();
    }

    //////////////////////////////////////////////////////////////////////////

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

    fn pick_physical_device(instance: &Instance) -> (vk::PhysicalDevice, QueueFamiliesIndices) {
        let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let physical_device = physical_devices
            .into_iter()
            .find(|physical_device| Self::is_device_suitable(instance, *physical_device))
            .expect("No suitable physical devices");

        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        info!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });
        let (graphics, present) = Self::find_queue_families(instance, physical_device);
        let queue_family_indices = QueueFamiliesIndices {
            graphics_index: graphics.unwrap(),
            present_index: present.unwrap(),
        };

        (physical_device, queue_family_indices)
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

    fn is_device_suitable(instance: &Instance, physical_device: vk::PhysicalDevice) -> bool {
        let (graphics, present) = Self::find_queue_families(instance, physical_device);
        let extension_support = Self::check_device_extension_support(instance, physical_device);
        let features = unsafe { instance.get_physical_device_features(physical_device) };

        graphics.is_some()
            && present.is_some()
            && extension_support
            && features.sampler_anisotropy == vk::TRUE
    }

    fn find_queue_families(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> (Option<u32>, Option<u32>) {
        let mut graphics = None;
        let mut present = None;
        let props =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        for (index, family) in props.iter().filter(|f| f.queue_count > 0).enumerate() {
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index as u32);
                present = Some(index as u32);
                break;
            }
        }

        (graphics, present)
    }

    fn get_required_device_extensions() -> [&'static CStr; 1] {
        [Swapchain::name()]
    }

    fn check_device_extension_support(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> bool {
        let required_extensions = Self::get_required_device_extensions();
        let extension_props = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap()
        };

        required_extensions.iter().all(|required| {
            extension_props.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                required == &name
            })
        })
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            if let Some((report, callback)) = self.debug_utils.take() {
                report.destroy_debug_utils_messenger(callback, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct QueueFamiliesIndices {
    pub graphics_index: u32,
    pub present_index: u32,
}
