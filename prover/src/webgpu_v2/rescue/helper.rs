use once_cell::sync::OnceCell;

use crate::webgpu_v2::helper::WebGpuHelper;


static WGPU_HELPER_RPO_RPX: OnceCell<WebGpuHelper> = OnceCell::new();

pub fn get_wgpu_helper_rp() -> Option<&'static WebGpuHelper> {
    WGPU_HELPER_RPO_RPX.get()
}
pub async fn get_wgpu_helper_rp_async() -> &'static WebGpuHelper {
    let wgpu_helper = WGPU_HELPER_RPO_RPX.get();

    if wgpu_helper.is_none() { 
        let helper = WebGpuHelper::new_async(wgpu::include_wgsl!("rescue_prime.wgsl")).await.unwrap();
        WGPU_HELPER_RPO_RPX.set(helper).unwrap();
        WGPU_HELPER_RPO_RPX.get().unwrap()
        
    }else{
        wgpu_helper.unwrap()

    }
}


