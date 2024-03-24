use air::Felt;
use wgpu::util::DeviceExt;


#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::webgpu_v2::{helper::WebGpuHelper, utils::get_dispatch_linear};

pub fn log2(n: usize) -> u32 {
    assert!(n.is_power_of_two(), "n must be a power of two");
    n.trailing_zeros()
}

pub async fn generate_merkle_tree_webgpu_rpo(
    helper: &WebGpuHelper,
    leaves: &[[Felt; 4]],
    rpx: bool,
) -> Option<Vec<[Felt; 4]>> {
    let height = log2(leaves.len());
    let digest_size = 4 * 8usize;
    let tree_nodes_size = digest_size * leaves.len();
    let staging_buffer = helper.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: tree_nodes_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let leaf_digests_buffer = helper.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("leaf_digests"),
        contents: unsafe {
            core::slice::from_raw_parts(leaves.as_ptr() as *const u8, leaves.len() * digest_size)
        },
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let nodes_buffer = helper.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Nodes Buffer"),
        size: tree_nodes_size as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let rpo_hash_leaves_pipeline =
        helper.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &helper.module,
            entry_point: if rpx { "rpx_hash_leaves" } else { "rpo_hash_leaves" },
        });
    let mut node_count_value: [u32; 1] = [(leaves.len() / 2usize) as u32];

    let node_count_buf = helper.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("node_count"),
        contents: bytemuck::cast_slice(&node_count_value),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let mut encoder = helper
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = rpo_hash_leaves_pipeline.get_bind_group_layout(0);
    let bind_group = helper.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: nodes_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: node_count_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: leaf_digests_buffer.as_entire_binding(),
            },
        ],
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&rpo_hash_leaves_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute hash leaves");

        let dispatch_dims = get_dispatch_linear(node_count_value[0] as u64);

        cpass.dispatch_workgroups(dispatch_dims.0, dispatch_dims.1, dispatch_dims.2);
        // Number of cells to run, the (x,y,z) size of item being processed
    }

    // start merkle hashing
    let rpo_hash_level_pipeline =
        helper.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &helper.module,
            entry_point: if rpx { "rpx_hash_level" } else{ "rpo_hash_level" },
        });

    for _ in 1..height {
        node_count_value[0] >>= 1;
        let node_count_buf = helper.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("node_count"),
            contents: bytemuck::cast_slice(&node_count_value),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group_layout = rpo_hash_level_pipeline.get_bind_group_layout(0);
        let bind_group = helper.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: nodes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: node_count_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&rpo_hash_level_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("compute hash leaves");

            let dispatch_dims = get_dispatch_linear(node_count_value[0] as u64);

            cpass.dispatch_workgroups(dispatch_dims.0, dispatch_dims.1, dispatch_dims.2);
            // Number of cells to run, the (x,y,z) size of item being processed
        }
    }

    // end merkle hashing

    // get result

    encoder.copy_buffer_to_buffer(&nodes_buffer, 0, &staging_buffer, 0, tree_nodes_size as u64);

    helper.queue.submit(Some(encoder.finish()));
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
        let result =
            unsafe { core::slice::from_raw_parts(data.as_ptr() as *mut [Felt; 4], leaves.len()) };

        /*
        let result = unsafe {
          Vec::from_raw_parts(data.as_ptr() as *mut [Felt; 4], n, n)
        };*/

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
        panic!("failed to merkle hasher on web gpu!")
    }
}
