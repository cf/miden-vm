use air::Felt;
use processor::crypto::Blake3_256;
use processor::{crypto::Rpo256, ONE};
use winter_crypto::{ElementHasher, MerkleTree, RandomCoin};

use crate::crypto::RpoDigest;
use crate::webgpu_v2::blake3::helper::get_wgpu_helper_blake3_async;
use crate::{async_execution_prover::AsyncExecutionProver, hwa::{AsyncHWAProver, HWABridge, HWAProver, HWARowHasherAsync}};

use super::super::utils::SimpleVectorAllocator;
use super::absorb::WebGpuBlake3_256RowMajor;
use super::helper::get_wgpu_helper_blake3;
use super::merkle::generate_merkle_tree_webgpu_blake3_256;

pub(crate) struct WebGPUBlake3_256ExecutionProver<R>(pub AsyncExecutionProver<Blake3_256, R>)
where
    R: RandomCoin<BaseField = Felt, Hasher = Blake3_256>;


pub struct FakeWebGPUBlake3_256RowHasher{
  num_base_columns: usize,
  rows: Vec<Vec<Felt>>,
  current_column: usize,
  //row_absorb: WebGpuRpo256RowMajor,
}
impl FakeWebGPUBlake3_256RowHasher {
  fn new(domain_size: usize, num_base_columns: usize) -> Self{
    let v : Vec<Vec<Felt>> = (0..domain_size).map(|_| Vec::<Felt>::with_capacity(num_base_columns)).collect();
    Self { num_base_columns, rows: v, current_column: 0 }

  }
  pub fn update(&mut self, rows: &[[Felt; 8]]) {
    if self.current_column + 8 > self.num_base_columns {
      let mod_col = self.num_base_columns%8;
      for r in 0..rows.len() {
        for i in 0..mod_col {
          self.rows[r].push(rows[r][i]);
        }
      }
    }else{
      for r in 0..rows.len() {
        for i in 0..8 {
          self.rows[r].push(rows[r][i]);
        }
      }
    }
    self.current_column += 8;
  }
  pub fn finish(&self) -> MerkleTree<Blake3_256> {

    let rr = self.rows.iter().map(|r| {
      Blake3_256::hash_elements(r)
    }).collect();

    MerkleTree::<Blake3_256>::new(rr).unwrap()
  }
}
pub struct WebGPUBlake3_256RowHasher{
  row_absorb: WebGpuBlake3_256RowMajor,
  //row_absorb: WebGpuRpo256RowMajor,
}
impl HWARowHasherAsync<[Felt; 8], MerkleTree<Blake3_256>> for WebGPUBlake3_256RowHasher {
    async fn new(domain_size: usize, num_base_columns: usize, _requires_padding: bool) -> Self {
      //println!("hwa domain_size = {}, num_base_columns = {}", domain_size, num_base_columns);
      let helper = get_wgpu_helper_blake3_async().await;
      let row_absorb = WebGpuBlake3_256RowMajor::new(helper, domain_size, num_base_columns);

      Self { row_absorb,}
    }

    fn update(&mut self, rows: &[[Felt; 8]]) {
      self.row_absorb.update(get_wgpu_helper_blake3().unwrap(), rows);
      //self.fake_hasher.update(rows)
    }

    fn update_pad(&mut self, rows: &[[Felt; 8]]) {
      self.row_absorb.update(get_wgpu_helper_blake3().unwrap(), rows);

      //self.fake_hasher.update(rows)
    }

    async fn finish(self) -> MerkleTree<Blake3_256> {
      //self.fake_hasher.finish()
      let helper = get_wgpu_helper_blake3().unwrap();

      let leaves = self.row_absorb.finish(helper).await.unwrap();
      let nodes = generate_merkle_tree_webgpu_blake3_256(helper, &leaves).await.unwrap();

      MerkleTree::<Blake3_256>::from_raw_parts(nodes, leaves).unwrap()
      //MerkleTree::<Blake3_256>::new(leaves).unwrap()

    }
}

impl< R: RandomCoin<BaseField = Felt, Hasher = Blake3_256>> HWAProver<8> for WebGPUBlake3_256ExecutionProver<R> {
  type BaseField = Felt;
  type HashFn = Blake3_256;
  type VectorAllocator = SimpleVectorAllocator;
  fn get_padded_segment_idx(num_base_columns: usize) -> Option<usize> {
      if num_base_columns%8 == 0 {
        None
      }else{
        Some(num_base_columns / 8)
      }
  }
}

impl< R: RandomCoin<BaseField = Felt, Hasher = Blake3_256>> AsyncHWAProver<8> for WebGPUBlake3_256ExecutionProver<R> {

  type RandomCoin = R;
  type RowHasher = WebGPUBlake3_256RowHasher;
}


pub fn get_webgpu_blake3_256_prover<R: RandomCoin<BaseField = Felt, Hasher = Blake3_256>>(prover: AsyncExecutionProver<Blake3_256, R>) -> HWABridge<WebGPUBlake3_256ExecutionProver<R>>{
   HWABridge::<WebGPUBlake3_256ExecutionProver<R>>(prover)
}