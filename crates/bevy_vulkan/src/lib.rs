use std::borrow::Cow;

use futures_lite::future;

use bevy_app::prelude::*;
use bevy_app::prelude::*;
use bevy_ecs::{
    system::{IntoExclusiveSystem, IntoSystem},
    world::World,
};
use bevy_render::{
    renderer::{RenderResourceContext, shared_buffers_update_system, SharedBuffers},
    RenderStage,
};
use renderer::VulkanRenderResourceContext;
pub use vulkan_render_pass::*;
pub use vulkan_renderer::*;
pub use vulkan_resources::*;

pub mod renderer;
mod vulkan_render_pass;
mod vulkan_renderer;
mod vulkan_resources;

#[derive(Default)]
pub struct VulkanPlugin;

impl Plugin for VulkanPlugin {
    fn build(&self, app: &mut AppBuilder) {
        let render_system = get_vulkan_render_system(app.world_mut());
        app.add_system_to_stage(RenderStage::Render, render_system.exclusive_system())
            .add_system_to_stage(
                RenderStage::PostRender,
                shared_buffers_update_system.system(),
            );
    }
}

pub fn get_vulkan_render_system(world: &mut World) -> impl FnMut(&mut World) {
    let mut vulkan_renderer = VulkanRenderer::new();

    let resource_context = VulkanRenderResourceContext::new(
        vulkan_renderer.entry.clone(),
        vulkan_renderer.instance.clone(),
        vulkan_renderer.device.clone(),
        vulkan_renderer.physical_device.clone(),
        vulkan_renderer.queue_indices,
    );
    world.insert_resource::<Box<dyn RenderResourceContext>>(Box::new(resource_context));
    world.insert_resource(SharedBuffers::new(4096));
    move |world| {
        vulkan_renderer.update(world);
    }
}
