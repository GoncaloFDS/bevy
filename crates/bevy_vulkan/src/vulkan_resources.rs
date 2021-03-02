use ash::vk;
use bevy_asset::{Handle, HandleUntyped};
use bevy_render::{
    pipeline::PipelineDescriptor,
    renderer::{RenderResourceId, TextureId},
    shader::Shader,
};
use bevy_utils::HashMap;
use bevy_window::WindowId;
use parking_lot::RwLock;
use std::sync::Arc;

#[derive(Default, Clone, Debug)]
pub struct VulkanResources {
    pub window_surfaces: Arc<RwLock<HashMap<WindowId, vk::SurfaceKHR>>>,
    pub window_swap_chains: Arc<RwLock<HashMap<WindowId, vk::SwapchainKHR>>>,
    pub swap_chain_images: Arc<RwLock<HashMap<TextureId, vk::Image>>>,
    pub swap_chain_image_views: Arc<RwLock<HashMap<TextureId, vk::ImageView>>>,
    pub shader_modules: Arc<RwLock<HashMap<Handle<Shader>, vk::ShaderModule>>>,
    pub render_pipelines: Arc<RwLock<HashMap<Handle<PipelineDescriptor>, vk::Pipeline>>>,

    pub asset_resources: Arc<RwLock<HashMap<(HandleUntyped, u64), RenderResourceId>>>,
}
