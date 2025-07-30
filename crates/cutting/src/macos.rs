//! macOS-specific initialization for GStreamer
//! 
//! This module handles the NSApplication requirement for GStreamer GL elements on macOS.

#[cfg(target_os = "macos")]
use cocoa::appkit::NSApplication;
#[cfg(target_os = "macos")]
use objc::runtime::Object;

/// Initialize NSApplication on macOS to prevent GStreamer GL warnings
/// 
/// This function must be called on the main thread before using any GStreamer
/// elements that interact with graphics (like playbin with video output).
/// 
/// # Returns
/// 
/// Returns `Ok(())` on success or if not running on macOS.
pub fn init_macos_app() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "macos")]
    {
        use std::sync::Once;
        
        static MACOS_INIT: Once = Once::new();
        
        MACOS_INIT.call_once(|| {
            unsafe {
                // Get or create the shared NSApplication instance
                let _app: *mut Object = NSApplication::sharedApplication(cocoa::base::nil);
                
                // Optional: Set activation policy to regular app
                // This makes the app appear in the dock if needed
                // Commented out by default to keep it as a background process
                // NSApplication::setActivationPolicy_(_app, cocoa::appkit::NSApplicationActivationPolicyRegular);
                
                println!("NSApplication initialized for GStreamer on macOS");
            }
        });
    }
    
    Ok(())
}

/// Alternative function that initializes NSApplication as a background process
/// 
/// Use this if you want the app to remain in the background without appearing in the dock.
pub fn init_macos_background_app() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "macos")]
    {
        use std::sync::Once;
        
        static MACOS_BG_INIT: Once = Once::new();
        
        MACOS_BG_INIT.call_once(|| {
            unsafe {
                let app: *mut Object = NSApplication::sharedApplication(cocoa::base::nil);
                
                // Set as background application - won't appear in dock
                let bg_policy = cocoa::appkit::NSApplicationActivationPolicyAccessory;
                NSApplication::setActivationPolicy_(app, bg_policy);
            }
        });
    }
    
    Ok(())
} 