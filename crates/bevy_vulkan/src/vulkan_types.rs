use std::ffi::CStr;
use std::os::raw::c_char;

use ash::vk;

use bevy_render::texture::TextureFormat;

use crate::vulkan_type_converter::{VulkanFrom, VulkanInto};

// convert vk string to String
pub fn vk_to_string(raw_char_array: &[c_char]) -> String {
    let raw_string = unsafe {
        let pointer = raw_char_array.as_ptr();
        CStr::from_ptr(pointer)
    };
    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
        .to_owned()
}

pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
}

impl AllocatedBuffer {
    pub fn new(
        allocator: &vk_mem::Allocator,
        buffer_size: vk::DeviceSize,
        usage_flags: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        memory_flags: vk::MemoryPropertyFlags,
    ) -> Self {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(buffer_size)
            .usage(usage_flags);
        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: memory_usage,
            required_flags: memory_flags,
            ..Default::default()
        };
        let (buffer, allocation, _info) = allocator
            .create_buffer(&buffer_info, &allocation_info)
            .unwrap();

        AllocatedBuffer { buffer, allocation }
    }

    pub fn destroy(&mut self, allocator: &vk_mem::Allocator) {
        allocator
            .destroy_buffer(self.buffer, &self.allocation)
            .expect("failed to deallocate buffer");
    }
}

pub struct AllocatedImage {
    pub image: vk::Image,
    pub allocation: vk_mem::Allocation,
}

impl AllocatedImage {
    pub fn new(
        allocator: &vk_mem::Allocator,
        create_info: vk::ImageCreateInfo,
        allocation_info: vk_mem::AllocationCreateInfo,
    ) -> Self {
        let (image, allocation, _info) = allocator
            .create_image(&create_info, &allocation_info)
            .unwrap();

        AllocatedImage { image, allocation }
    }

    pub fn destroy(&mut self, allocator: &vk_mem::Allocator) {
        allocator
            .destroy_image(self.image, &self.allocation)
            .expect("failed to deallocate image");
    }
}

pub struct SwapchainDescriptor {
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub present_mode: vk::PresentModeKHR,
    pub frames_in_flight: u32,
}

impl VulkanFrom<&bevy_window::Window> for SwapchainDescriptor {
    fn from(window: &bevy_window::Window) -> Self {
        SwapchainDescriptor {
            format: TextureFormat::default().vulkan_into(),
            extent: vk::Extent2D {
                width: window.physical_width(),
                height: window.physical_height(),
            },
            present_mode: if window.vsync() {
                vk::PresentModeKHR::FIFO
            } else {
                vk::PresentModeKHR::IMMEDIATE
            },
            frames_in_flight: 3,
        }
    }
}
