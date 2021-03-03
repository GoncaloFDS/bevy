use std::sync::Arc;

use ash::{extensions::khr::Surface, vk, Device};
use parking_lot::RwLock;

use bevy_asset::{Handle, HandleUntyped};
use bevy_render::{
    pipeline::PipelineDescriptor,
    renderer::{RenderResourceId, TextureId},
    shader::Shader,
};
use bevy_utils::HashMap;
use bevy_window::WindowId;

#[derive(Clone)]
pub struct VulkanResources {
    pub device: Arc<ash::Device>,
    pub surface_loader: Arc<Surface>,

    pub window_surfaces: Arc<RwLock<HashMap<WindowId, vk::SurfaceKHR>>>,
    pub window_swap_chains: Arc<RwLock<HashMap<WindowId, vk::SwapchainKHR>>>,
    pub swap_chain_images: Arc<RwLock<HashMap<TextureId, vk::Image>>>,
    pub swap_chain_image_views: Arc<RwLock<HashMap<TextureId, vk::ImageView>>>,
    pub shader_modules: Arc<RwLock<HashMap<Handle<Shader>, vk::ShaderModule>>>,
    pub render_pipelines: Arc<RwLock<HashMap<Handle<PipelineDescriptor>, vk::Pipeline>>>,

    pub asset_resources: Arc<RwLock<HashMap<(HandleUntyped, u64), RenderResourceId>>>,
}
impl VulkanResources {
    pub fn new(device: Arc<Device>, surface_loader: Arc<Surface>) -> Self {
        VulkanResources {
            device,
            surface_loader,
            window_surfaces: Default::default(),
            window_swap_chains: Default::default(),
            swap_chain_images: Default::default(),
            swap_chain_image_views: Default::default(),
            shader_modules: Default::default(),
            render_pipelines: Default::default(),
            asset_resources: Default::default(),
        }
    }
}

impl Drop for VulkanResources {
    fn drop(&mut self) {
        if Arc::strong_count(&self.window_surfaces) <= 1 {
            self.window_surfaces
                .write()
                .iter()
                .for_each(|(_, surface)| unsafe {
                    self.surface_loader.destroy_surface(*surface, None)
                })
        }
    }
}
