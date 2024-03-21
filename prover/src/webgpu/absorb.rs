use air::Felt;
use wgpu::util::DeviceExt;

#[cfg(not(feature = "std"))]
use core::vec::Vec;

use super::helper::{get_dispatch_linear, WebGpuHelper};

pub struct WebGpuRpo256RowMajor {
    row_hash_state_buffer: wgpu::Buffer,
    digests_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    //node_count_buffer: wgpu::Buffer,
    encoder: wgpu::CommandEncoder,
    pipeline: wgpu::ComputePipeline,
    pad_pipeline: wgpu::ComputePipeline,
    n: usize,
    pad_columns: bool,
}

impl WebGpuRpo256RowMajor {
    pub fn new_rpo(helper: &WebGpuHelper, n: usize, pad_columns: bool) -> Self {
        Self::new(helper, n, pad_columns, false)
    }
    pub fn new_rpx(helper: &WebGpuHelper, n: usize, pad_columns: bool) -> Self {
        Self::new(helper, n, pad_columns, true)
    }
    pub fn new(helper: &WebGpuHelper, n: usize, pad_columns: bool, rpx: bool) -> Self {
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
        let row_hash_state_buffer = helper.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Row State Buffer"),
            size: row_state_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
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

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &helper.module,
            entry_point: if rpx { "rpx_absorb_rows" } else { "rpo_absorb_rows" },
        });

        let compute_pipeline_pad =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &helper.module,
                entry_point: if rpx {
                    "rpx_absorb_rows_pad"
                } else {
                    "rpo_absorb_rows_pad"
                },
            });

        Self {
            staging_buffer,
            encoder,
            pipeline: compute_pipeline,
            pad_pipeline: compute_pipeline_pad,
            //node_count_buffer,
            digests_buffer,
            row_hash_state_buffer,
            pad_columns,
            n,
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

        let pipeline = if self.pad_columns {
            self.pad_columns = false;
            &self.pad_pipeline
        } else {
            &self.pipeline
        };

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
                    resource: self.row_hash_state_buffer.as_entire_binding(),
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
    pub async fn finish(self, helper: &WebGpuHelper) -> Option<Vec<[Felt; 4]>> {
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
            let result = unsafe { core::slice::from_raw_parts(data.as_ptr() as *mut [Felt; 4], n) };

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
