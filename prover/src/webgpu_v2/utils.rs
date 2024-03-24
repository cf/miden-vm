use crate::hwa::HWAVectorAllocator;


#[allow(clippy::uninit_vec)]
pub unsafe fn uninit_vector_real<T>(length: usize) -> Vec<T> {
    let mut vector = Vec::with_capacity(length);
    vector.set_len(length);
    vector
}


pub struct SimpleVectorAllocator{}
impl HWAVectorAllocator for SimpleVectorAllocator {
    fn uninit_vector<T>(length: usize) -> Vec<T> {
        unsafe {uninit_vector_real::<T>(length)}
    }
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
