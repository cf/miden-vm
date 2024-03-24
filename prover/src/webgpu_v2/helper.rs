use once_cell::sync::OnceCell;
use wgpu::ShaderModuleDescriptor;

#[derive(Debug)]
pub struct WebGpuHelper {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub module: wgpu::ShaderModule,
}
impl WebGpuHelper {
    pub async fn new_async(module: ShaderModuleDescriptor<'static>) -> Option<Self> {
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
        }*/ module);

        Some(Self {
            device,
            queue,
            module: cs_module,
        })
    }
}

unsafe impl Send for WebGpuHelper{}
unsafe impl Sync for WebGpuHelper{}