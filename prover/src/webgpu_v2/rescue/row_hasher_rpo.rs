use air::Felt;
use processor::{crypto::Rpo256, ONE};
use winter_crypto::{MerkleTree, RandomCoin};

use crate::crypto::RpoDigest;
use crate::{async_execution_prover::AsyncExecutionProver, hwa::{AsyncHWAProver, HWABridge, HWAProver, HWARowHasherAsync}};

use super::super::utils::SimpleVectorAllocator;
use super::{absorb::WebGpuRpo256RowMajor, helper::{get_wgpu_helper_rp, get_wgpu_helper_rp_async}, merkle::generate_merkle_tree_webgpu_rpo};

pub(crate) struct WebGPURpoExecutionProver<R>(pub AsyncExecutionProver<Rpo256, R>)
where
    R: RandomCoin<BaseField = Felt, Hasher = Rpo256>;


pub struct WebGPURpoRowHasher{
  num_base_columns: usize,
  _domain_size: usize,
  _requires_padding: bool,
  row_absorb: WebGpuRpo256RowMajor,
}
impl HWARowHasherAsync<[Felt; 8], MerkleTree<Rpo256>> for WebGPURpoRowHasher {
    async fn new(domain_size: usize, num_base_columns: usize, requires_padding: bool) -> Self {
      let helper = get_wgpu_helper_rp_async().await;
      //println!("domain_size = {}, num_base_columns = {}", domain_size, num_base_columns);
      Self { num_base_columns, _domain_size:domain_size, _requires_padding: requires_padding, row_absorb: WebGpuRpo256RowMajor::new_rpo(helper, domain_size, requires_padding) }
    }

    fn update(&mut self, rows: &[[Felt; 8]]) {
      self.row_absorb.update(get_wgpu_helper_rp().unwrap(), rows)
    }

    fn update_pad(&mut self, rows: &[[Felt; 8]]) {
      let rpo_pad_column = self.num_base_columns % 8;
      let rpo_padded_segment: Vec<[Felt; 8]> = rows
      .iter()
      .map(|x| {
          let mut s = x.clone();
          s[rpo_pad_column] = ONE;
          s
      })
      .collect();
    self.row_absorb.update(get_wgpu_helper_rp().unwrap(), &rpo_padded_segment)
    }

    async fn finish(self) -> MerkleTree<Rpo256> {
      let helper = get_wgpu_helper_rp().unwrap();
      let row_hashes  =self.row_absorb.finish(helper).await.unwrap();
      let tree_nodes = generate_merkle_tree_webgpu_rpo(helper, &row_hashes, false).await;

      let nodes = tree_nodes.unwrap().into_iter().map(RpoDigest::new).collect();
      let leaves = row_hashes.into_iter().map(RpoDigest::new).collect();
      let commitment = MerkleTree::<Rpo256>::from_raw_parts(nodes, leaves).unwrap();
      commitment

    }
}

impl< R: RandomCoin<BaseField = Felt, Hasher = Rpo256>> HWAProver<8> for WebGPURpoExecutionProver<R> {
  type BaseField = Felt;
  type HashFn = Rpo256;
  type VectorAllocator = SimpleVectorAllocator;
  fn get_padded_segment_idx(num_base_columns: usize) -> Option<usize> {
      if num_base_columns%8 == 0 {
        None
      }else{
        Some(num_base_columns / 8)
      }
  }
}

impl< R: RandomCoin<BaseField = Felt, Hasher = Rpo256>> AsyncHWAProver<8> for WebGPURpoExecutionProver<R> {

  type RandomCoin = R;
  type RowHasher = WebGPURpoRowHasher;
}


pub fn get_webgpu_rpo_prover<R: RandomCoin<BaseField = Felt, Hasher = Rpo256>>(prover: AsyncExecutionProver<Rpo256, R>) -> HWABridge<WebGPURpoExecutionProver<R>>{
   HWABridge::<WebGPURpoExecutionProver<R>>(prover)
}