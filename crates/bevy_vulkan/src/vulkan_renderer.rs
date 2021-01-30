use crate::renderer::*;
use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk, Device, Instance,
};
use bevy_app::prelude::*;
use bevy_ecs::{Resources, World};
use bevy_render::{
    render_graph::{DependentNodeStager, RenderGraph, RenderGraphStager},
    renderer::RenderResourceContext,
};
use bevy_utils::tracing::*;
use bevy_window::{WindowCreated, WindowResized, Windows};
use std::ffi::CStr;
use bevy_app::ManualEventReader;

pub struct VulkanRenderer {
    pub window_resized_event_reader: ManualEventReader<WindowResized>,
    pub window_created_event_reader: ManualEventReader<WindowCreated>,
}

impl VulkanRenderer {
    pub fn new() -> Self {
        info!("Create Vulkan Renderer");

        VulkanRenderer {
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
            render_resource_context.create_window_surface(window.id(), &*winit_windows);

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
            max_thread_count: 2,
        };
        graph_executor.execute(world, resources, &mut borrowed);
    }

    pub fn update(&mut self, world: &mut World, resources: &mut Resources) {
        self.handle_window_create_events(resources);
        self.run_graph(world, resources);

        let render_resource_context = resources.get::<Box<dyn RenderResourceContext>>().unwrap();
        render_resource_context.drop_all_swap_chain_textures();
        render_resource_context.remove_stale_bind_groups();
    }

    //////////////////////////////////////////////////////////////////////////
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        info!("Dropping application");
        // unsafe {
        //     if let Some((report, callback)) = self.debug_report_callback.take() {
        //         report.destroy_debug_report_callback(callback, None);
        //     }
        //     self.instance.destroy_instance(None);
        // }
    }
}
