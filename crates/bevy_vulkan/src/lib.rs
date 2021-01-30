mod debug;
pub mod renderer;
mod vulkan_renderer;
use bevy_app::{prelude::*, Plugin};
use bevy_ecs::{IntoSystem, Resources, World};
use bevy_render::renderer::{shared_buffers_update_system, RenderResourceContext, SharedBuffers};
use renderer::VulkanRenderResourceContext;
use vulkan_renderer::*;

#[derive(Default)]
pub struct VulkanPlugin;

impl Plugin for VulkanPlugin {
    fn build(&self, app: &mut AppBuilder) {
        let render_system = get_vulkan_render_system(app.resources_mut());
        app.add_system_to_stage(bevy_render::stage::RENDER, render_system.system())
            .add_system_to_stage(
                bevy_render::stage::POST_RENDER,
                shared_buffers_update_system.system(),
            );
    }
}

pub fn get_vulkan_render_system(
    resources: &mut Resources,
) -> impl FnMut(&mut World, &mut Resources) {
    let resource_context = VulkanRenderResourceContext::new();
    resources.insert::<Box<dyn RenderResourceContext>>(Box::new(resource_context));
    resources.insert(SharedBuffers::new(4096));

    let mut vulkan_renderer = VulkanRenderer::new();
    move |world, resources| {
        vulkan_renderer.update(world, resources);
    }
}
