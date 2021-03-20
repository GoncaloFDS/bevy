use std::{ops::Range, sync::Arc};

use ash::{
    extensions::khr::{Surface, Win32Surface},
    vk, Device, Entry, Instance,
};

use bevy_asset::{Assets, Handle, HandleUntyped};
use bevy_render::{
    pipeline::{BindGroupDescriptorId, PipelineDescriptor},
    renderer::{
        BindGroup, BufferId, BufferInfo, BufferMapMode, RenderResourceContext, RenderResourceId,
        SamplerId, TextureId,
    },
    shader::{glsl_to_spirv, Shader, ShaderError, ShaderSource},
    texture::{SamplerDescriptor, TextureDescriptor},
};
use bevy_utils::tracing::*;
use bevy_window::{Window, WindowId};

use crate::{vulkan_renderer::QueueFamiliesIndices, vulkan_resources::VulkanResources};

pub const BIND_BUFFER_ALIGNMENT: usize = 256;
pub const TEXTURE_ALIGNMENT: usize = 256;

pub fn required_extension_names() -> Vec<*const i8> {
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}

#[derive(Clone)]
pub struct VulkanRenderResourceContext {
    pub entry: Arc<ash::Entry>,
    pub instance: Arc<ash::Instance>,
    pub device: Arc<ash::Device>,
    pub physical_device: Arc<vk::PhysicalDevice>,
    pub queue_indices: QueueFamiliesIndices,
    pub surface_loader: Arc<Surface>,
    pub resources: VulkanResources,
}

impl VulkanRenderResourceContext {
    pub fn new(
        entry: Arc<Entry>,
        instance: Arc<Instance>,
        device: Arc<Device>,
        physical_device: Arc<vk::PhysicalDevice>,
        queue_indices: QueueFamiliesIndices,
    ) -> Self {
        let surface_loader = Arc::new(Surface::new(entry.as_ref(), instance.as_ref()));
        let resources = VulkanResources::new(device.clone(), surface_loader.clone());

        VulkanRenderResourceContext {
            entry,
            instance,
            device,
            physical_device,
            queue_indices,
            surface_loader,
            resources,
        }
    }

    pub fn set_window_surface(&self, window_id: WindowId, surface: vk::SurfaceKHR) {
        let mut window_surfaces = self.resources.window_surfaces.write();
        window_surfaces.insert(window_id, surface);
    }
}

impl RenderResourceContext for VulkanRenderResourceContext {
    fn create_swap_chain(&self, _window: &Window) {
        warn!("create_swap_chain not implemented")
    }

    fn next_swap_chain_texture(&self, _window: &Window) -> TextureId {
        warn!("next_swap_chain_texture not implemented");
        TextureId::new()
    }

    fn drop_swap_chain_texture(&self, _render_resource: TextureId) {
        warn!("drop_swap_chain_texture not implemented")
    }

    fn drop_all_swap_chain_textures(&self) {
        warn!("drop_all_chain_textures not implemented")
    }

    fn create_sampler(&self, _sampler_descriptor: &SamplerDescriptor) -> SamplerId {
        warn!("create_sampler not implemented");
        SamplerId::new()
    }

    fn create_texture(&self, _texture_descriptor: TextureDescriptor) -> TextureId {
        warn!("create_texture not implemented");
        TextureId::new()
    }

    fn create_buffer(&self, _buffer_info: BufferInfo) -> BufferId {
        warn!("create_buffer not implemented");
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

    fn map_buffer(&self, _id: BufferId, _mode: BufferMapMode) {
        warn!("map_buffer not implemented")
    }

    fn unmap_buffer(&self, _id: BufferId) {
        warn!("unmap_buffer not implemented")
    }

    fn create_buffer_with_data(&self, _buffer_info: BufferInfo, _data: &[u8]) -> BufferId {
        warn!("create_buffer_with_data not implemented");
        BufferId::new()
    }

    fn create_shader_module(&self, _shader_handle: &Handle<Shader>, _shaders: &Assets<Shader>) {
        todo!()
    }

    fn create_shader_module_from_source(&self, _shader_handle: &Handle<Shader>, _shader: &Shader) {
        todo!()
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

    fn remove_buffer(&self, _buffer: BufferId) {
        todo!()
    }

    fn remove_texture(&self, _texture: TextureId) {
        todo!()
    }

    fn remove_sampler(&self, _sampler: SamplerId) {
        todo!()
    }

    fn get_buffer_info(&self, _buffer: BufferId) -> Option<BufferInfo> {
        todo!()
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
        warn!("set_asset_resource_untyped not implemented")
    }

    fn get_asset_resource_untyped(
        &self,
        _handle: HandleUntyped,
        _index: u64,
    ) -> Option<RenderResourceId> {
        warn!("get_asset_resource_untyped not implemented");
        None
    }

    fn remove_asset_resource_untyped(&self, _handle: HandleUntyped, _index: u64) {
        todo!()
    }

    fn create_render_pipeline(
        &self,
        _pipeline_handle: Handle<PipelineDescriptor>,
        _pipeline_descriptor: &PipelineDescriptor,
        _shaders: &Assets<Shader>,
    ) {
        warn!("create_render_pipeline not implemented");
    }

    fn bind_group_descriptor_exists(
        &self,
        _bind_group_descriptor_id: BindGroupDescriptorId,
    ) -> bool {
        warn!("bind_group_descriptor_exists not implemented");
        true
    }

    fn create_bind_group(
        &self,
        _bind_group_descriptor_id: BindGroupDescriptorId,
        _bind_group: &BindGroup,
    ) {
        warn!("create_bind_group not implemented");
    }

    fn clear_bind_groups(&self) {
        todo!()
    }

    fn remove_stale_bind_groups(&self) {
        warn!("remove_stale_bind_groups not implemented");
    }
}
