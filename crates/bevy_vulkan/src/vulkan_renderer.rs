use std::{
    ffi::{CStr, CString},
    sync::Arc,
};

use ash::{
    extensions::{ext::DebugUtils, khr::Swapchain},
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk, Device, Entry, Instance,
};

use bevy_app::{Events, ManualEventReader};
use bevy_ecs::world::{Mut, World};
use bevy_render::{
    render_graph::{DependentNodeStager, RenderGraph, RenderGraphStager},
    renderer::RenderResourceContext,
};
use bevy_utils::tracing::*;
use bevy_window::{WindowCreated, WindowResized, Windows};

use crate::renderer::*;
use crate::vulkan_debug;
use crate::vulkan_debug::*;

pub struct VulkanRenderer {
    pub entry: Arc<ash::Entry>,
    pub instance: Arc<ash::Instance>,
    pub device: Arc<ash::Device>,
    pub physical_device: Arc<vk::PhysicalDevice>,

    pub graphics_queue: vk::Queue,
    pub queue_indices: QueueFamiliesIndices,

    debug_utils: Option<(DebugUtils, vk::DebugUtilsMessengerEXT)>,

    pub window_resized_event_reader: ManualEventReader<WindowResized>,
    pub window_created_event_reader: ManualEventReader<WindowCreated>,
}

impl Default for VulkanRenderer {
    fn default() -> Self {
        let entry: Entry = unsafe { ash::Entry::new().expect("Failed to create entry.") };
        let instance = create_instance(&entry);
        let debug_utils = vulkan_debug::setup_debug_messenger(&entry, &instance);

        let (physical_device, queue_indices) = pick_physical_device(&instance);
        let graphics_queue_family = queue_indices.graphics_index;
        let device = create_logical_device(&instance, physical_device, &[graphics_queue_family]);
        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };

        VulkanRenderer {
            entry: Arc::new(entry),
            instance: Arc::new(instance),
            device: Arc::new(device),
            physical_device: Arc::new(physical_device),
            graphics_queue,
            queue_indices,
            debug_utils,
            window_resized_event_reader: Default::default(),
            window_created_event_reader: Default::default(),
        }
    }
}

impl VulkanRenderer {
    pub fn handle_window_create_events(&mut self, world: &mut World) {
        let world = world.cell();
        let mut render_resource_context = world
            .get_resource_mut::<Box<dyn RenderResourceContext>>()
            .unwrap();
        let render_resource_context = render_resource_context
            .downcast_mut::<VulkanRenderResourceContext>()
            .unwrap();
        let windows = world.get_resource::<Windows>().unwrap();
        let window_created_events = world.get_resource::<Events<WindowCreated>>().unwrap();
        for window_created_event in self
            .window_created_event_reader
            .iter(&window_created_events)
        {
            let window = windows
                .get(window_created_event.id)
                .expect("Received window created event for non-existing window.");
            let winit_windows = world.get_resource::<bevy_winit::WinitWindows>().unwrap();
            let winit_window = winit_windows.get_window(window.id()).unwrap();
            let surface = unsafe {
                ash_window::create_surface(
                    self.entry.as_ref(),
                    self.instance.as_ref(),
                    winit_window,
                    None,
                )
            };
            render_resource_context.set_window_surface(window.id(), surface.unwrap());

            info!("handled window create event");
        }
    }

    pub fn run_graph(&mut self, world: &mut World) {
        world.resource_scope(|world, mut render_graph: Mut<RenderGraph>| {
            render_graph.prepare(world);
            //stage nodes
            let mut stager = DependentNodeStager::loose_grouping();
            let stages = stager.get_stages(&render_graph).unwrap();
            let mut borrowed = stages.borrow(&mut render_graph);

            //execute stages
            let graph_executor = VulkanRenderGraphExecutor {
                max_thread_count: 1,
            };
            graph_executor.execute(
                world,
                self.device.clone(),
                &mut self.graphics_queue,
                &mut borrowed,
            );
        })
    }

    pub fn update(&mut self, world: &mut World) {
        self.handle_window_create_events(world);
        self.run_graph(world);

        let render_resource_context = world
            .get_resource::<Box<dyn RenderResourceContext>>()
            .unwrap();
        render_resource_context.drop_all_swap_chain_textures();
        render_resource_context.remove_stale_bind_groups();
    }
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

fn pick_physical_device(instance: &Instance) -> (vk::PhysicalDevice, QueueFamiliesIndices) {
    let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap() };
    let physical_device = physical_devices
        .into_iter()
        .find(|physical_device| is_device_suitable(instance, *physical_device))
        .expect("No suitable physical devices");

    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    info!("Selected physical device: {:?}", unsafe {
        CStr::from_ptr(props.device_name.as_ptr())
    });
    let (graphics, present) = find_queue_families(instance, physical_device);
    let queue_family_indices = QueueFamiliesIndices {
        graphics_index: graphics.unwrap(),
        present_index: present.unwrap(),
    };

    (physical_device, queue_family_indices)
}

fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    unique_queue_families: &[u32],
) -> Device {
    let queue_priorities = [1.0];
    let mut queue_create_infos = vec![];
    for &queue_family in unique_queue_families.iter() {
        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities)
            .build();
        queue_create_infos.push(queue_create_info);
    }

    let enabled_extension_names = [Swapchain::name().as_ptr()];

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(queue_create_infos.as_slice())
        .enabled_extension_names(&enabled_extension_names);

    unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .unwrap()
    }
}

fn is_device_suitable(instance: &Instance, physical_device: vk::PhysicalDevice) -> bool {
    let (graphics, present) = find_queue_families(instance, physical_device);
    let extension_support = check_device_extension_support(instance, physical_device);
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
    let props = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
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
    let required_extensions = get_required_device_extensions();
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
