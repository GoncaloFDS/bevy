use bevy_app::{prelude::*, Plugin};
use bevy_ecs::{IntoExclusiveSystem, IntoSystem, Resources, World};
use bevy_render::renderer::{shared_buffers_update_system, RenderResourceContext, SharedBuffers};
use renderer::VulkanRenderResourceContext;
use vulkan_renderer::*;
use bevy_render::RenderStage;

mod debug;
pub mod renderer;
mod vulkan_render_pass;
mod vulkan_renderer;

#[derive(Default)]
pub struct VulkanPlugin;

impl Plugin for VulkanPlugin {
    fn build(&self, app: &mut AppBuilder) {
        let render_system = get_vulkan_render_system(app.resources_mut());
        app.add_system_to_stage(RenderStage::Render, render_system.exclusive_system())
            .add_system_to_stage(
                RenderStage::PostRender,
                shared_buffers_update_system.system(),
            );
    }
}

pub fn get_vulkan_render_system(
    resources: &mut Resources,
) -> impl FnMut(&mut World, &mut Resources) {
    let mut vulkan_renderer = VulkanRenderer::new();
    let resource_context = VulkanRenderResourceContext::new(
        vulkan_renderer.entry.clone(),
        vulkan_renderer.instance.clone(),
        vulkan_renderer.device.clone(),
        vulkan_renderer.physical_device.clone(),
        vulkan_renderer.queue_indices,
    );
    resources.insert::<Box<dyn RenderResourceContext>>(Box::new(resource_context));
    resources.insert(SharedBuffers::new(4096));

    move |world, resources| {
        vulkan_renderer.update(world, resources);
    }
}
