use std::borrow::Cow;
use std::ffi::{CStr, CString};

use ash::extensions::ext::DebugUtils;
use ash::{version::EntryV1_0, vk, Entry, Instance};

use bevy_utils::tracing::*;

#[cfg(debug_assertions)]
pub const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
pub const ENABLE_VALIDATION_LAYERS: bool = false;

pub const REQUIRED_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE {
        trace!("{}", message);
    } else if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        info!("{}", message);
    } else if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("{}", message);
    } else if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("{}", message);
    }
    vk::FALSE
}

pub fn get_layer_names_and_pointers() -> (Vec<CString>, Vec<*const i8>) {
    let layer_names = REQUIRED_LAYERS
        .iter()
        .map(|name| CString::new(*name).expect("Failed to build CString"))
        .collect::<Vec<_>>();
    let layer_names_ptrs = layer_names
        .iter()
        .map(|name| name.as_ptr())
        .collect::<Vec<_>>();

    (layer_names, layer_names_ptrs)
}

pub fn check_validation_layer_support(entry: &Entry) {
    for required in REQUIRED_LAYERS.iter() {
        let found = entry
            .enumerate_instance_layer_properties()
            .unwrap()
            .iter()
            .any(|layer| {
                let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                let name = name.to_str().expect("Failed to get layer name pointer");
                required == &name
            });

        if !found {
            panic!("Validation layer not supported: {}", required);
        }
    }
}

pub fn setup_debug_messenger(
    entry: &Entry,
    instance: &Instance,
) -> Option<(DebugUtils, vk::DebugUtilsMessengerEXT)> {
    if !ENABLE_VALIDATION_LAYERS {
        return None;
    }
    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .pfn_user_callback(Some(vulkan_debug_callback));

    let debug_utils_loader = DebugUtils::new(entry, instance);
    let debug_call_back =
        unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_info, None) }.unwrap();

    Some((debug_utils_loader, debug_call_back))
}
