use once_cell::sync::OnceCell;

use crate::webgpu_v2::helper::WebGpuHelper;


static WGPU_HELPER_BLAKE3: OnceCell<WebGpuHelper> = OnceCell::new();

pub fn get_wgpu_helper_blake3() -> Option<&'static WebGpuHelper> {
  WGPU_HELPER_BLAKE3.get()
}
pub async fn get_wgpu_helper_blake3_async() -> &'static WebGpuHelper {
    let wgpu_helper = WGPU_HELPER_BLAKE3.get();

    if wgpu_helper.is_none() { 
        let helper = WebGpuHelper::new_async(wgpu::include_wgsl!("blake3.wgsl")).await.unwrap();
        WGPU_HELPER_BLAKE3.set(helper).unwrap();
        WGPU_HELPER_BLAKE3.get().unwrap()
        
    }else{
        wgpu_helper.unwrap()

    }
}


