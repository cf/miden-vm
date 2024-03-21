use once_cell::sync::OnceCell;

#[derive(Debug)]
pub struct WebGpuHelper {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub module: wgpu::ShaderModule,
}
const DISPATCH_MAX_PER_DIM: u64 = 32768u64;

pub fn get_dispatch_linear(size: u64) -> (u32, u32, u32) {
    if size <= DISPATCH_MAX_PER_DIM {
        return (size as u32, 1, 1);
    } else if size <= DISPATCH_MAX_PER_DIM * DISPATCH_MAX_PER_DIM {
        assert_eq!(size % DISPATCH_MAX_PER_DIM, 0);
        return (DISPATCH_MAX_PER_DIM as u32, (size / DISPATCH_MAX_PER_DIM) as u32, 1);
    } else if size <= DISPATCH_MAX_PER_DIM * DISPATCH_MAX_PER_DIM * DISPATCH_MAX_PER_DIM {
        assert_eq!(size % (DISPATCH_MAX_PER_DIM * DISPATCH_MAX_PER_DIM), 0);
        return (
            DISPATCH_MAX_PER_DIM as u32,
            DISPATCH_MAX_PER_DIM as u32,
            (size / (DISPATCH_MAX_PER_DIM * DISPATCH_MAX_PER_DIM)) as u32,
        );
    } else {
        panic!("size too large for dispatch");
    }
}

impl WebGpuHelper {
    pub async fn new_async() -> Option<Self> {
        // Instantiates instance of WebGPU
        let instance = wgpu::Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await?;

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        let cs_module = device.create_shader_module(/*wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("hsh.wgsl"))),
        }*/ wgpu::include_wgsl!("rescue_prime.wgsl"));

        Some(Self {
            device,
            queue,
            module: cs_module,
        })
    }
}

unsafe impl Send for WebGpuHelper{}
unsafe impl Sync for WebGpuHelper{}
static WGPU_HELPER: OnceCell<WebGpuHelper> = OnceCell::new();

pub fn get_wgpu_helper() -> Option<&'static WebGpuHelper> {
    WGPU_HELPER.get()
}


pub async fn init_wgpu_helper() {
    if WGPU_HELPER.get().is_none() { 
        let helper = WebGpuHelper::new_async().await.unwrap();
        WGPU_HELPER.set(helper).unwrap();
    }
}

pub async fn get_wgpu_helper_async() -> &'static WebGpuHelper {
    let wgpu_helper = WGPU_HELPER.get();

    if wgpu_helper.is_none() { 
        let helper = WebGpuHelper::new_async().await.unwrap();
        WGPU_HELPER.set(helper).unwrap();
        WGPU_HELPER.get().unwrap()
        
    }else{
        wgpu_helper.unwrap()

    }
}
