use air::{Felt, FieldElement, ProcessorAir, StarkField};
use elsa::FrozenVec;
use processor::{math::fft, ColMatrix, ExecutionTrace};
use tracing::info_span;
use winter_crypto::{ElementHasher, Hasher, MerkleTree, RandomCoin};
use winter_prover::{
    async_prover::AsyncProver, math::ExtensibleField, matrix::{get_evaluation_offsets, RowMatrix, Segment}, proof::Queries, Air, AsyncTraceLde, CompositionPoly, CompositionPolyTrace, ConstraintCommitment, ConstraintEvaluator, DefaultConstraintEvaluator, EvaluationFrame, StarkDomain, Trace, TraceInfo, TraceLayout, TraceLde, TracePolyTable
};

use crate::async_execution_prover::AsyncExecutionProver;

pub trait HWAVectorAllocator {
    fn uninit_vector<T>(length: usize) -> Vec<T>;
}
pub trait HWARowHasher<R, L> {
    fn new(domain_size: usize, num_base_columns: usize, requires_padding: bool) -> Self;
    fn update(&mut self, rows: &[R]);
    fn update_pad(&mut self, rows: &[R]);
    fn finish(self) -> L;
}
pub trait HWARowHasherAsync<R, L> {
    async fn new(domain_size: usize, num_base_columns: usize, requires_padding: bool) -> Self;
    fn update(&mut self, rows: &[R]);
    fn update_pad(&mut self, rows: &[R]);
    async fn finish(self) -> L;
}

pub trait HWAProver<const SEGMENT_BATCH_SIZE: usize>: Sized {
    type VectorAllocator: HWAVectorAllocator;
    /// Base field for the computation described by this prover.
    type BaseField: StarkField + ExtensibleField<2> + ExtensibleField<3> + Sized;

    /// Hash function to be used.
    type HashFn: ElementHasher<BaseField = Self::BaseField>;

        
    fn get_padded_segment_idx(num_base_columns: usize) -> Option<usize>;
    fn build_aligned_segement<E>(
        polys: &ColMatrix<E>,
        poly_offset: usize,
        offsets: &[Self::BaseField],
        twiddles: &[Self::BaseField],
    ) -> Segment<Self::BaseField, SEGMENT_BATCH_SIZE>
    where
        E: FieldElement<BaseField = Self::BaseField>,
    {
        let poly_size = polys.num_rows();
        let domain_size = offsets.len();
        assert!(domain_size.is_power_of_two());
        assert!(domain_size > poly_size);
        assert_eq!(poly_size, twiddles.len() * 2);
        assert!(poly_offset < polys.num_base_cols());

        // allocate memory for the segment
        let data = if polys.num_base_cols() - poly_offset >= SEGMENT_BATCH_SIZE {
            // if we will fill the entire segment, we allocate uninitialized memory
            Self::VectorAllocator::uninit_vector(domain_size)
        } else {
            // but if some columns in the segment will remain unfilled, we allocate memory initialized
            // to zeros to make sure we don't end up with memory with undefined values
            //group_vector_elements(Self::BaseField::zeroed_vector(SEGMENT_BATCH_SIZE * domain_size))
            let mut tmp = Self::VectorAllocator::uninit_vector(domain_size);
            tmp.fill([Self::BaseField::ZERO; SEGMENT_BATCH_SIZE]);
            tmp
        };

        Segment::new_with_buffer(data, polys, poly_offset, offsets, twiddles)
    }
    fn build_aligned_segements<E>(
        polys: &ColMatrix<E>,
        twiddles: &[Self::BaseField],
        offsets: &[Self::BaseField],
    ) -> Vec<Segment<Self::BaseField, SEGMENT_BATCH_SIZE>>
    where
        E: FieldElement<BaseField = Self::BaseField>,
    {
        assert!(SEGMENT_BATCH_SIZE > 0, "batch size N must be greater than zero");
        debug_assert_eq!(polys.num_rows(), twiddles.len() * 2);
        debug_assert_eq!(offsets.len() % polys.num_rows(), 0);

        let num_segments = if polys.num_base_cols() % SEGMENT_BATCH_SIZE == 0 {
            polys.num_base_cols() / SEGMENT_BATCH_SIZE
        } else {
            polys.num_base_cols() / SEGMENT_BATCH_SIZE + 1
        };

        (0..num_segments)
            .map(|i| Self::build_aligned_segement(polys, i * SEGMENT_BATCH_SIZE, offsets, twiddles))
            .collect()
    }

}

pub trait AsyncHWAProver<const SEGMENT_BATCH_SIZE: usize> : HWAProver<SEGMENT_BATCH_SIZE> {
    type RowHasher: HWARowHasherAsync<[Self::BaseField; SEGMENT_BATCH_SIZE], MerkleTree<Self::HashFn>>;

    /// Algebraic intermediate representation (AIR) for the computation described by this prover.
    //type Air: Air<BaseField = Self::BaseField>;

    /// Execution trace of the computation described by this prover.
    //type Trace: Trace<BaseField = Self::BaseField>;
    
    /// PRNG to be used for generating random field elements.
    type RandomCoin: RandomCoin<BaseField = Self::BaseField, Hasher = Self::HashFn>;

    /// Trace low-degree extension for building the LDEs of trace segments and their commitments.
  /*   type TraceLde<E>: AsyncTraceLde<E, HashFn = Self::HashFn>
    where
        E: FieldElement<BaseField = Self::BaseField>;
*/
    /// Constraints evaluator used to evaluate AIR constraints over the extended execution trace.


        async fn build_constraint_commitment<E: FieldElement<BaseField = Self::BaseField>>(
          composition_poly_trace: CompositionPolyTrace<E>,
          num_trace_poly_columns: usize,
          domain: &StarkDomain<Self::BaseField>,
      ) -> (ConstraintCommitment<E, Self::HashFn>, CompositionPoly<E>) {
          // evaluate composition polynomial columns over the LDE domain
          let composition_poly =
              CompositionPoly::new(composition_poly_trace, domain, num_trace_poly_columns);
          let blowup = domain.trace_to_lde_blowup();
          let offsets =
              get_evaluation_offsets::<E>(composition_poly.column_len(), blowup, domain.offset());
          let segments = Self::build_aligned_segements(composition_poly.data(), domain.trace_twiddles(), &offsets);
  
          // build constraint evaluation commitment
          //let now = Instant::now();
          let lde_domain_size = domain.lde_domain_size();
          let num_base_columns =
              composition_poly.num_columns() * <E as FieldElement>::EXTENSION_DEGREE;
              let padded_segment_idx = Self::get_padded_segment_idx(num_base_columns);
              let requires_padding = padded_segment_idx.is_some();
              let mut row_hasher = Self::RowHasher::new(
                  lde_domain_size,
                  num_base_columns,
                  requires_padding,
              ).await;
          for (segment_idx, segment) in segments.iter().enumerate() {
              // check if the segment requires padding
              if padded_segment_idx.map_or(false, |pad_idx| pad_idx == segment_idx) {
                  // duplicate and modify the last segment with Rpo256's padding
                  // rule ("1" followed by "0"s). Our segments are already
                  // padded with "0"s we only need to add the "1"s.
                  row_hasher.update_pad(segment);
                  break;
              }
              row_hasher.update(segment);
          }
          let commitment = row_hasher.finish().await;
          let composed_evaluations = RowMatrix::<E>::from_segments(segments, num_base_columns);
          let constraint_commitment = ConstraintCommitment::new(composed_evaluations, commitment);
  
          (constraint_commitment, composition_poly)
      }
/// Computes a low-degree extension (LDE) of the provided execution trace over the specified
/// domain and builds a commitment to the extended trace.
///
/// The extension is performed by interpolating each column of the execution trace into a
/// polynomial of degree = trace_length - 1, and then evaluating the polynomial over the LDE
/// domain.
///
/// Trace commitment is computed by hashing each row of the extended execution trace, and then
/// building a Merkle tree from the resulting hashes.
///
/// Interpolations and evaluations are computed on the CPU while hashes are simultaneously
/// computed on the GPU:
///
/// ```text
///        ──────────────────────────────────────────────────────
///               ┌───┐   ┌────┐   ┌───┐   ┌────┐   ┌───┐
///  CPU:   ... ──┤fft├─┬─┤ifft├───┤fft├─┬─┤ifft├───┤fft├─┬─ ...
///               └───┘ │ └────┘   └───┘ │ └────┘   └───┘ │
///        ╴╴╴╴╴╴╴╴╴╴╴╴╴┼╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴┼╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴╴┼╴╴╴╴╴╴
///                     │ ┌──────────┐   │ ┌──────────┐   │
///  GPU:               └─┤   hash   │   └─┤   hash   │   └─ ...
///                       └──────────┘     └──────────┘
///        ────┼────────┼────────┼────────┼────────┼────────┼────
///           t=n     t=n+1    t=n+2     t=n+3   t=n+4    t=n+5
/// ```
async fn build_trace_commitment<
E: FieldElement<BaseField = Self::BaseField>,
F: FieldElement<BaseField = Self::BaseField>>(
  trace: &ColMatrix<F>,
  domain: &StarkDomain<Self::BaseField>,
) -> (RowMatrix<F>, MerkleTree<Self::HashFn>, ColMatrix<F>) {
  // interpolate the execution trace
  let inv_twiddles = fft::get_inv_twiddles::<Self::BaseField>(trace.num_rows());
  let trace_polys = trace.columns().map(|col| {
      let mut poly = col.to_vec();
      fft::interpolate_poly(&mut poly, &inv_twiddles);
      poly
  });

  // extend the execution trace and generate hashes on the gpu
  let lde_segments = FrozenVec::new();
  let lde_domain_size: usize = domain.lde_domain_size();
  let num_base_columns = trace.num_base_cols();
  let padded_segment_idx = Self::get_padded_segment_idx(num_base_columns);
  let requires_padding = padded_segment_idx.is_some();
  let mut row_hasher = Self::RowHasher::new(
      lde_domain_size,
      num_base_columns,
      requires_padding,
  ).await;
  let mut lde_segment_generator  = SegmentGenerator::<Self, F, _, SEGMENT_BATCH_SIZE>::new(trace_polys, domain);
  let mut lde_segment_iter = lde_segment_generator.gen_segment_iter().enumerate();
  for (segment_idx, segment) in &mut lde_segment_iter {
      let segment = lde_segments.push_get(Box::new(segment));
      // check if the segment requires padding
      if padded_segment_idx.map_or(false, |pad_idx| pad_idx == segment_idx) {
          row_hasher.update_pad(segment);
          assert!(lde_segment_iter.next().is_none(), "padded segment should be the last");
          break;
      }
      row_hasher.update(segment);
  }
  let trace_tree = row_hasher.finish().await;
  // aggregate segments at the same time as the GPU generates the merkle tree nodes
  let lde_segments = lde_segments.into_vec().into_iter().map(|p| *p).collect();
  let trace_lde = RowMatrix::from_segments(lde_segments, num_base_columns);
  let trace_polys = lde_segment_generator.into_polys().unwrap();

  (trace_lde, trace_tree, trace_polys)
}



}
fn build_trace_commitment_sync<E, F, H, const DEFAULT_SEGMENT_WIDTH: usize>(
    trace: &ColMatrix<F>,
    domain: &StarkDomain<E::BaseField>,
) -> (RowMatrix<F>, MerkleTree<H>, ColMatrix<F>)
where
    E: FieldElement,
    F: FieldElement<BaseField = E::BaseField>,
    H: ElementHasher<BaseField = E::BaseField>,
{
    // extend the execution trace
    let (trace_lde, trace_polys) = {
        let span = info_span!(
            "extend_execution_trace",
            num_cols = trace.num_cols(),
            blowup = domain.trace_to_lde_blowup()
        )
        .entered();
        let trace_polys = trace.interpolate_columns();
        let trace_lde =
            RowMatrix::evaluate_polys_over::<DEFAULT_SEGMENT_WIDTH>(&trace_polys, domain);
        drop(span);

        (trace_lde, trace_polys)
    };
    assert_eq!(trace_lde.num_cols(), trace.num_cols());
    assert_eq!(trace_polys.num_rows(), trace.num_rows());
    assert_eq!(trace_lde.num_rows(), domain.lde_domain_size());

    // build trace commitment
    let tree_depth = trace_lde.num_rows().ilog2() as usize;
    let trace_tree = info_span!("compute_execution_trace_commitment", tree_depth)
        .in_scope(|| trace_lde.commit_to_rows());
    assert_eq!(trace_tree.depth(), tree_depth);

    (trace_lde, trace_tree, trace_polys)
}

// TRACE LOW DEGREE EXTENSION (METAL)
// ================================================================================================

/// Contains all segments of the extended execution trace, the commitments to these segments, the
/// LDE blowup factor, and the [TraceInfo].
///
/// Segments are stored in two groups:
/// - Main segment: this is the first trace segment generated by the prover. Values in this segment
///   will always be elements in the base field (even when an extension field is used).
/// - Auxiliary segments: a list of 0 or more segments for traces generated after the prover
///   commits to the first trace segment. Currently, at most 1 auxiliary segment is possible.
pub struct AsyncHWATraceLde<const SEGMENT_BATCH_SIZE: usize, H: HWAProver<SEGMENT_BATCH_SIZE>, E: FieldElement<BaseField = H::BaseField>> {
    // low-degree extension of the main segment of the trace
    main_segment_lde: RowMatrix<E::BaseField>,
    // commitment to the main segment of the trace
    main_segment_tree: MerkleTree<H::HashFn>,
    // low-degree extensions of the auxiliary segments of the trace
    aux_segment_ldes: Vec<RowMatrix<E>>,
    // commitment to the auxiliary segments of the trace
    aux_segment_trees: Vec<MerkleTree<H::HashFn>>,
    blowup: usize,
    trace_info: TraceInfo,
}

fn build_segment_queries<const SEGMENT_BATCH_SIZE: usize, HWA: HWAProver<SEGMENT_BATCH_SIZE>, E: FieldElement<BaseField = HWA::BaseField>>(
    segment_lde: &RowMatrix<E>,
    segment_tree: &MerkleTree<HWA::HashFn>,
    positions: &[usize],
) -> Queries {
    // for each position, get the corresponding row from the trace segment LDE and put all these
    // rows into a single vector
    let trace_states =
        positions.iter().map(|&pos| segment_lde.row(pos).to_vec()).collect::<Vec<_>>();

    // build Merkle authentication paths to the leaves specified by positions
    let trace_proof = segment_tree
        .prove_batch(positions)
        .expect("failed to generate a Merkle proof for trace queries");

    Queries::new(trace_proof, trace_states)
}

impl<const SEGMENT_BATCH_SIZE: usize, HWA: AsyncHWAProver<SEGMENT_BATCH_SIZE>, E: FieldElement> AsyncHWATraceLde<SEGMENT_BATCH_SIZE, HWA, E> where E: FieldElement<BaseField = HWA::BaseField> {
    /// Takes the main trace segment columns as input, interpolates them into polynomials in
    /// coefficient form, evaluates the polynomials over the LDE domain, commits to the
    /// polynomial evaluations, and creates a new [DefaultTraceLde] with the LDE of the main trace
    /// segment and the commitment.
    ///
    /// Returns a tuple containing a [TracePolyTable] with the trace polynomials for the main trace
    /// segment and the new [DefaultTraceLde].
    pub async fn new(
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<E::BaseField>,
        domain: &StarkDomain<E::BaseField>,
    ) -> (Self, TracePolyTable<E>) {
        // extend the main execution trace and build a Merkle tree from the extended trace
        let (main_segment_lde, main_segment_tree, main_segment_polys) =
        HWA::build_trace_commitment::<E, E::BaseField>(main_trace, domain).await;

        let trace_poly_table = TracePolyTable::new(main_segment_polys);
        let trace_lde = Self {
            main_segment_lde,
            main_segment_tree,
            aux_segment_ldes: Vec::new(),
            aux_segment_trees: Vec::new(),
            blowup: domain.trace_to_lde_blowup(),
            trace_info: trace_info.clone(),
        };

        (trace_lde, trace_poly_table)
    }

    // TEST HELPERS
    // --------------------------------------------------------------------------------------------
}
impl<const SEGMENT_BATCH_SIZE: usize, HWA: AsyncHWAProver<SEGMENT_BATCH_SIZE>, E: FieldElement<BaseField = HWA::BaseField>> TraceLde<E> for AsyncHWATraceLde<SEGMENT_BATCH_SIZE, HWA, E> {
    type HashFn = HWA::HashFn;


    /// Returns the commitment to the low-degree extension of the main trace segment.
    fn get_main_trace_commitment(&self) -> <Self::HashFn as Hasher>::Digest {
        let root_hash = self.main_segment_tree.root();
        *root_hash
    }

    /// Takes auxiliary trace segment columns as input, interpolates them into polynomials in
    /// coefficient form, evaluates the polynomials over the LDE domain, and commits to the
    /// polynomial evaluations.
    ///
    /// Returns a tuple containing the column polynomials in coefficient from and the commitment
    /// to the polynomial evaluations over the LDE domain.
    ///
    /// # Panics
    ///
    /// This function will panic if any of the following are true:
    /// - the number of rows in the provided `aux_trace` does not match the main trace.
    /// - this segment would exceed the number of segments specified by the trace layout.
    fn add_aux_segment(
        &mut self,
        aux_trace: &ColMatrix<E>,
        domain: &StarkDomain<HWA::BaseField>,
    ) -> (ColMatrix<E>, <Self::HashFn as Hasher>::Digest) {
        // extend the auxiliary trace segment and build a Merkle tree from the extended trace
        let (aux_segment_lde, aux_segment_tree, aux_segment_polys) =
        build_trace_commitment_sync::<E, E, Self::HashFn, SEGMENT_BATCH_SIZE>(aux_trace, domain);

        // check errors
        assert!(
            self.aux_segment_ldes.len() < self.trace_info.layout().num_aux_segments(),
            "the specified number of auxiliary segments has already been added"
        );
        assert_eq!(
            self.main_segment_lde.num_rows(),
            aux_segment_lde.num_rows(),
            "the number of rows in the auxiliary segment must be the same as in the main segment"
        );

        // save the lde and commitment
        self.aux_segment_ldes.push(aux_segment_lde);
        let root_hash = *aux_segment_tree.root();
        self.aux_segment_trees.push(aux_segment_tree);

        (aux_segment_polys, root_hash)
    }

    /// Reads current and next rows from the main trace segment into the specified frame.
    fn read_main_trace_frame_into(&self, lde_step: usize, frame: &mut EvaluationFrame<HWA::BaseField>) {
        // at the end of the trace, next state wraps around and we read the first step again
        let next_lde_step = (lde_step + self.blowup()) % self.trace_len();

        // copy main trace segment values into the frame
        frame.current_mut().copy_from_slice(self.main_segment_lde.row(lde_step));
        frame.next_mut().copy_from_slice(self.main_segment_lde.row(next_lde_step));
    }

    /// Reads current and next rows from the auxiliary trace segment into the specified frame.
    ///
    /// # Panics
    /// This currently assumes that there is exactly one auxiliary trace segment, and will panic
    /// otherwise.
    fn read_aux_trace_frame_into(&self, lde_step: usize, frame: &mut EvaluationFrame<E>) {
        // at the end of the trace, next state wraps around and we read the first step again
        let next_lde_step = (lde_step + self.blowup()) % self.trace_len();

        // copy auxiliary trace segment values into the frame
        let segment = &self.aux_segment_ldes[0];
        frame.current_mut().copy_from_slice(segment.row(lde_step));
        frame.next_mut().copy_from_slice(segment.row(next_lde_step));
    }

    /// Returns trace table rows at the specified positions along with Merkle authentication paths
    /// from the commitment root to these rows.
    fn query(&self, positions: &[usize]) -> Vec<Queries> {
        // build queries for the main trace segment
        let mut result = vec![build_segment_queries::<SEGMENT_BATCH_SIZE, HWA, E::BaseField>(
            &self.main_segment_lde,
            &self.main_segment_tree,
            positions,
        )];

        // build queries for auxiliary trace segments
        for (i, segment_tree) in self.aux_segment_trees.iter().enumerate() {
            let segment_lde = &self.aux_segment_ldes[i];
            result.push(build_segment_queries::<SEGMENT_BATCH_SIZE, HWA, _>(segment_lde, segment_tree, positions));
        }

        result
    }

    /// Returns the number of rows in the execution trace.
    fn trace_len(&self) -> usize {
        self.main_segment_lde.num_rows()
    }

    /// Returns blowup factor which was used to extend original execution trace into trace LDE.
    fn blowup(&self) -> usize {
        self.blowup
    }

    /// Returns the trace layout of the execution trace.
    fn trace_layout(&self) -> &TraceLayout {
        self.trace_info.layout()
    }
}
impl<const SEGMENT_BATCH_SIZE: usize, HWA: AsyncHWAProver<SEGMENT_BATCH_SIZE>, E: FieldElement<BaseField = HWA::BaseField>> AsyncTraceLde<E> for AsyncHWATraceLde<SEGMENT_BATCH_SIZE, HWA, E> {
    async fn add_aux_segment_async(
        &mut self,
        aux_trace: &ColMatrix<E>,
        domain: &StarkDomain<<E as FieldElement>::BaseField>,
    ) -> (ColMatrix<E>, <Self::HashFn as Hasher>::Digest) {
        // extend the auxiliary trace segment and build a Merkle tree from the extended trace
        let (aux_segment_lde, aux_segment_tree, aux_segment_polys) =
            HWA::build_trace_commitment::<E,E>(aux_trace, domain).await;

        // check errors
        assert!(
            self.aux_segment_ldes.len() < self.trace_info.layout().num_aux_segments(),
            "the specified number of auxiliary segments has already been added"
        );
        assert_eq!(
            self.main_segment_lde.num_rows(),
            aux_segment_lde.num_rows(),
            "the number of rows in the auxiliary segment must be the same as in the main segment"
        );

        // save the lde and commitment
        self.aux_segment_ldes.push(aux_segment_lde);
        let root_hash = *aux_segment_tree.root();
        self.aux_segment_trees.push(aux_segment_tree);

        (aux_segment_polys, root_hash)
    }
}


struct SegmentGenerator<'a, HWA: HWAProver<SEGMENT_BATCH_SIZE>, E, I, const SEGMENT_BATCH_SIZE: usize>
where
    E: FieldElement<BaseField = HWA::BaseField>,
    I: IntoIterator<Item = Vec<E>>,
{
    poly_iter: I::IntoIter,
    polys: Option<ColMatrix<E>>,
    poly_offset: usize,
    offsets: Vec<HWA::BaseField>,
    domain: &'a StarkDomain<HWA::BaseField>,
}

impl<'a, HWA: HWAProver<SEGMENT_BATCH_SIZE>, E, I, const SEGMENT_BATCH_SIZE: usize> SegmentGenerator<'a, HWA, E, I, SEGMENT_BATCH_SIZE>
where
    E: FieldElement<BaseField = HWA::BaseField>,
    I: IntoIterator<Item = Vec<E>>,
{
    fn new(polys: I, domain: &'a StarkDomain<HWA::BaseField>) -> Self {
        assert!(SEGMENT_BATCH_SIZE > 0, "batch size N must be greater than zero");
        let poly_size = domain.trace_length();
        let lde_blowup = domain.trace_to_lde_blowup();
        let offsets = get_evaluation_offsets::<E>(poly_size, lde_blowup, domain.offset());
        Self {
            poly_iter: polys.into_iter(),
            polys: None,
            poly_offset: 0,
            offsets,
            domain,
        }
    }

    /// Returns the matrix of polynomials used to generate segments.
    fn into_polys(self) -> Option<ColMatrix<E>> {
        self.polys
    }

    /// Returns a segment generating iterator.
    fn gen_segment_iter(&mut self) -> SegmentIterator<'a, '_, HWA, E, I, SEGMENT_BATCH_SIZE> {
        SegmentIterator(self)
    }

    /// Generates the next segment if it exists otherwise returns None.
    fn gen_next_segment(&mut self) -> Option<Segment<HWA::BaseField, SEGMENT_BATCH_SIZE>> {
        // initialize our col matrix
        if self.polys.is_none() {
            self.polys = Some(ColMatrix::new(vec![self.poly_iter.next()?]));
        }

        let offset = self.poly_offset;
        let polys = self.polys.as_mut().unwrap();
        while polys.num_base_cols() < offset + SEGMENT_BATCH_SIZE {
            if let Some(poly) = self.poly_iter.next() {
                polys.merge_column(poly)
            } else {
                break;
            }
        }

        // terminate if there are no more segments to create
        if polys.num_base_cols() <= offset {
            return None;
        }

        let domain_size = self.domain.lde_domain_size();
        let mut data: Vec<[HWA::BaseField; SEGMENT_BATCH_SIZE]> = HWA::VectorAllocator::uninit_vector(domain_size); //Vec::with_capacity(domain_size);//unsafe { page_aligned_uninit_vector(domain_size) };

        if polys.num_base_cols() < offset + SEGMENT_BATCH_SIZE {
            // the segment will remain unfilled so we pad it with zeros
            data.fill([HWA::BaseField::ZERO; SEGMENT_BATCH_SIZE]);
        }

        let twiddles = self.domain.trace_twiddles();
        let segment = Segment::new_with_buffer(data, &*polys, offset, &self.offsets, twiddles);
        self.poly_offset += SEGMENT_BATCH_SIZE;
        Some(segment)
    }
}
/* 
fn build_segment_queries<
    const N: usize,
    HWA: HWAProver<N>,
    E: FieldElement<BaseField = HWA::BaseField>,
>(
    segment_lde: &RowMatrix<E>,
    segment_tree: &MerkleTree<HWA::HashFn>,
    positions: &[usize],
) -> Queries {
    // for each position, get the corresponding row from the trace segment LDE and put all these
    // rows into a single vector
    let trace_states =
        positions.iter().map(|&pos| segment_lde.row(pos).to_vec()).collect::<Vec<_>>();

    // build Merkle authentication paths to the leaves specified by positions
    let trace_proof = segment_tree
        .prove_batch(positions)
        .expect("failed to generate a Merkle proof for trace queries");

    Queries::new(trace_proof, trace_states)
}
*/
struct SegmentIterator<'a, 'b, HWA: HWAProver<N>, E, I, const N: usize>(
    &'b mut SegmentGenerator<'a, HWA, E, I, N>,
)
where
    E: FieldElement<BaseField = HWA::BaseField>,
    I: IntoIterator<Item = Vec<E>>;

impl<'a, 'b, HWA: HWAProver<N>, E, I, const N: usize> Iterator
    for SegmentIterator<'a, 'b, HWA, E, I, N>
where
    E: FieldElement<BaseField = HWA::BaseField>,
    I: IntoIterator<Item = Vec<E>>,
{
    type Item = Segment<HWA::BaseField, N>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.gen_next_segment()
    }
}


pub struct HWABridge<HWA: AsyncHWAProver<8, BaseField = Felt>> (pub AsyncExecutionProver<HWA::HashFn, HWA::RandomCoin>);

impl<HWA: AsyncHWAProver<8, BaseField = Felt>> AsyncProver for HWABridge<HWA> {
    type BaseField = HWA::BaseField;

    type Air = ProcessorAir;
    type Trace = ExecutionTrace;

    type HashFn = HWA::HashFn;

    type RandomCoin = HWA::RandomCoin;

    type TraceLde<E: FieldElement<BaseField = HWA::BaseField>> = AsyncHWATraceLde<8, HWA, E>;

    type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
        DefaultConstraintEvaluator<'a, ProcessorAir, E>;

    fn get_pub_inputs(&self, trace: &Self::Trace) -> <<Self as AsyncProver>::Air as Air>::PublicInputs {
        self.0.get_pub_inputs(trace)
    }

    fn options(&self) -> &winter_prover::ProofOptions {
        self.0.options()
    }

    async fn new_trace_lde<E>(
        &self,
        trace_info: &TraceInfo,
        main_trace: &ColMatrix<Self::BaseField>,
        domain: &StarkDomain<Self::BaseField>,
    ) -> (Self::TraceLde<E>, TracePolyTable<E>)
    where
        E: FieldElement<BaseField = Self::BaseField> {
       AsyncHWATraceLde::<8, HWA, E>::new(trace_info, main_trace, domain).await
    }

    async fn new_evaluator<'a, E>(
        &self,
        air: &'a Self::Air,
        aux_rand_elements: winter_prover::AuxTraceRandElements<E>,
        composition_coefficients: winter_prover::ConstraintCompositionCoefficients<E>,
    ) -> Self::ConstraintEvaluator<'a, E>
    where
        E: FieldElement<BaseField = Self::BaseField> {
            self.0.new_evaluator(air, aux_rand_elements, composition_coefficients).await
        }
    
        async fn build_constraint_commitment<E>(
            &self,
            composition_poly_trace: CompositionPolyTrace<E>,
            num_constraint_composition_columns: usize,
            domain: &StarkDomain<Self::BaseField>,
        ) -> (ConstraintCommitment<E, Self::HashFn>, CompositionPoly<E>)
        where
            E: FieldElement<BaseField = Self::BaseField>,
        {
            HWA::build_constraint_commitment(composition_poly_trace, num_constraint_composition_columns, domain).await
        }
}