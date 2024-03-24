use air::Felt;
use processor::crypto::Blake3Digest;
use wgpu::util::DeviceExt;


#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::webgpu_v2::{helper::WebGpuHelper, utils::get_dispatch_linear};
const BLAKE3_FLAG_CHUNK_START : u32 = 1;
const BLAKE3_FLAG_CHUNK_END : u32 = 2;
const BLAKE3_FLAG_ROOT : u32 = 8;

struct Blake3PlanItem {
  pub flags: u32,
  pub block_length: u32,
}
impl Blake3PlanItem {
  pub fn new(flags: u32, block_length: u32) -> Self {
    Self { flags, block_length }
  }
  pub fn to_slice(&self) -> [u32; 2] {
    [self.block_length, self.flags]
  }
  pub fn generate_plan(total_length_u8: usize)->Vec<Self> {
    let num_std_blocks = total_length_u8/64;
    let pad_col = total_length_u8%64;



    let mut plan: Vec<Self> = (0..num_std_blocks).map(|_|Self::new(0, 64)).collect();
    if pad_col != 0 {
        plan.push(Self::new(0, pad_col as u32));
    }
    let last_ind = plan.len()-1;
    plan[0].flags |= BLAKE3_FLAG_CHUNK_START;
    plan[last_ind].flags |= BLAKE3_FLAG_CHUNK_END | BLAKE3_FLAG_ROOT;

    


    plan
  }
}

fn plan_item_to_webgpu_buffer(helper: &WebGpuHelper, plan_item: &Blake3PlanItem) -> wgpu::Buffer {
    let slice = plan_item.to_slice();
     helper.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("absorb_params"),
        contents: bytemuck::cast_slice(&slice),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}


pub struct WebGpuBlake3_256RowMajor {
    digests_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    //node_count_buffer: wgpu::Buffer,
    encoder: wgpu::CommandEncoder,
    first_absorb_pipeline: wgpu::ComputePipeline,
    absorb_pipeline: wgpu::ComputePipeline,
    n: usize,
    current_step: usize,
    plan: Vec<Blake3PlanItem>,
}

impl WebGpuBlake3_256RowMajor {
    pub fn new(helper: &WebGpuHelper, n: usize, num_base_columns: usize) -> Self {
        let device = &helper.device;
        let row_state_size = (4 * n * core::mem::size_of::<u64>()) as wgpu::BufferAddress;
        let digests_size = (4 * n * core::mem::size_of::<u64>()) as wgpu::BufferAddress;

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: digests_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        /*let node_count_value = [0u64; 1];
        let node_count_buffer =
            helper.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("node_count"),
                contents: bytemuck::cast_slice(&node_count_value),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });*/
        let digests_buffer = helper.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Digests Buffer"),
            size: digests_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let plan = Blake3PlanItem::generate_plan(num_base_columns*8);
        let first_absorb_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &helper.module,
            entry_point: "blake3_256_absorb_row_first",
        });

        let absorb_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &helper.module,
                entry_point: "blake3_256_absorb_row",
            });

        Self {
            staging_buffer,
            encoder,
            first_absorb_pipeline: first_absorb_pipeline,
            absorb_pipeline: absorb_pipeline,
            //node_count_buffer,
            digests_buffer,
            current_step: 0,
            n,
            plan,
        }
    }
    pub fn update(&mut self, helper: &WebGpuHelper, rows: &[[Felt; 8]]) {
        let rows_ptr =
            unsafe { core::slice::from_raw_parts(rows.as_ptr() as *mut u8, rows.len() * 8 * 8) };
        let row_input_buffer =
            helper.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rows"),
                contents: rows_ptr,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });


        let pipeline = if self.current_step == 0 {
            &self.first_absorb_pipeline
        } else {
            &self.absorb_pipeline
        };

        let absorb_params_buffer = plan_item_to_webgpu_buffer(helper, &self.plan[self.current_step]);
        self.current_step += 1;

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let bind_group = helper.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.digests_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: row_input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: absorb_params_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("compute row absorb");

            let dispatch_dims = get_dispatch_linear(rows.len() as u64);

            cpass.dispatch_workgroups(dispatch_dims.0, dispatch_dims.1, dispatch_dims.2);
        }
    }
    pub async fn finish(self, helper: &WebGpuHelper) -> Option<Vec<Blake3Digest<32>>> {
        let staging_buffer = self.staging_buffer;
        let results_buffer = self.digests_buffer;
        let mut encoder = self.encoder;
        let queue = &helper.queue;
        let n = self.n;
        encoder.copy_buffer_to_buffer(&results_buffer, 0, &staging_buffer, 0, (n * 4 * 8) as u64);

        queue.submit(Some(encoder.finish()));
        let buffer_slice = staging_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.

        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        helper.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // Awaits until `buffer_future` can be read from
        if let Ok(Ok(())) = receiver.recv_async().await {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            //let result:Vec<u64> = bytemuck::cast_slice(&data).to_vec();
            let result = unsafe { core::slice::from_raw_parts(data.as_ptr() as *mut Blake3Digest<32>, n) };

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            staging_buffer.unmap(); // Unmaps buffer from memory
                                    // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                    //   delete myPointer;
                                    //   myPointer = NULL;
                                    // It effectively frees the memory

            // Returns data from buffer
            Some(result.to_vec())
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}
