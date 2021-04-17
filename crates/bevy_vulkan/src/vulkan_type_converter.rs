use ash::vk;

use bevy_render::texture::TextureFormat;

pub trait VulkanFrom<T> {
    fn from(val: T) -> Self;
}

pub trait VulkanInto<U> {
    fn vulkan_into(self) -> U;
}

impl<T, U> VulkanInto<U> for T
    where
        U: VulkanFrom<T>,
{
    fn vulkan_into(self) -> U {
        U::from(self)
    }
}

impl VulkanFrom<TextureFormat> for vk::Format {
    fn from(val: TextureFormat) -> Self {
        match val {
            TextureFormat::R8Unorm => vk::Format::R8_UNORM,
            TextureFormat::R8Snorm => vk::Format::R8_SNORM,
            TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
            _ => panic!(),
        }
    }
}
