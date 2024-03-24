/*
MIT License

Copyright (c) 2024 QED Protocol (Zero Knowledge Labs Limited)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


fn add_w(a: vec2<u32>, b: vec2<u32>) -> vec2<u32>{
  let result = a+b;
  return vec2<u32>(result.x, result.y + select(0u, 1u, result.x<a.x));
}
fn sub_w(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let result = a-b;
  return vec2<u32>(result.x, result.y - select(0u, 1u, result.x>a.x));
}

fn sub_w_overflowing(a: vec2<u32>, b: vec2<u32>) -> vec3<u32>{
  let result = sub_w(a, b);
  return vec3<u32>(result.x, result.y, select(0u, 1u, result.y > a.y || (result.y == a.y && result.x > a.x)));
}
fn add_w_overflowing(a: vec2<u32>, b: vec2<u32>) -> vec3<u32>{
  let result = add_w(a, b);
  return vec3<u32>(result.x, result.y, select(0u, 1u, result.y < a.y));
}
fn mont_to_fp(x_0: u32, x_1: u32) -> vec2<u32>{

  let a_e = add_w_overflowing(vec2<u32>(x_0, x_1), vec2<u32>(0u, x_0));
  let b = sub_w(sub_w(a_e.xy, vec2<u32>(a_e.y, 0u)), vec2<u32>(a_e.z, 0u));
  let r_c  = sub_w_overflowing(vec2<u32>(0u, 0u), b);
  let result = sub_w(r_c.xy, vec2<u32>(r_c.z*0xffffffffu, 0u));
  return result;
}


alias Blake3BlockWords = array<u32,16>;
alias Blake3HashOut = array<u32,8>;

// precomputed permutation indicies
const MSG_PERMUTATION_2D = array<array<i32,16>,7>(
  array<i32,16>(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),
  array<i32,16>(2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8),
  array<i32,16>(3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1),
  array<i32,16>(10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6),
  array<i32,16>(12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4),
  array<i32,16>(9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7),
  array<i32,16>(11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13)
);

// using private varaible so we don't have to keep copying around the permutation state
var<private> state: Blake3BlockWords;
var<private> blockWords: Blake3BlockWords;

fn rot_right_16(x: u32) -> u32{
  return (x>>16)|(x<<16);
}

fn rot_right_12(x: u32) -> u32{
  return (x>>12)|(x<<20);
}

fn rot_right_8(x: u32) -> u32{
  return (x>>8)|(x<<24);
}

fn rot_right_7(x: u32) -> u32{
  return (x>>7)|(x<<25);
}

fn blake3_g(a: i32, b: i32, c: i32, d: i32, mx: u32, my: u32){
  state[a] = state[a] + state[b] + mx;

  state[d] = rot_right_16(state[d] ^ state[a]);
  state[c] = state[c] + state[d];
  state[b] = rot_right_12(state[b] ^ state[c]);
  state[a] = state[a] + state[b] + my;
  state[d] = rot_right_8(state[d] ^ state[a]);
  state[c] = state[c] + state[d];
  state[b] = rot_right_7(state[b] ^ state[c]);

}
/*

fn blake3_round_fn(round: i32){
  blake3_g(0, 4, 8, 12, blockWords[MSG_PERMUTATION_2D[round][0]], blockWords[MSG_PERMUTATION_2D[round][1]]);
  blake3_g(1, 5, 9, 13, blockWords[MSG_PERMUTATION_2D[round][2]], blockWords[MSG_PERMUTATION_2D[round][3]]);
  blake3_g(2, 6, 10, 14, blockWords[MSG_PERMUTATION_2D[round][4]], blockWords[MSG_PERMUTATION_2D[round][5]]);
  blake3_g(3, 7, 11, 15, blockWords[MSG_PERMUTATION_2D[round][6]], blockWords[MSG_PERMUTATION_2D[round][7]]);

  // Mix diagonals
  blake3_g(0, 5, 10, 15, blockWords[MSG_PERMUTATION_2D[round][8]], blockWords[MSG_PERMUTATION_2D[round][9]]);
  blake3_g(1, 6, 11, 12, blockWords[MSG_PERMUTATION_2D[round][10]], blockWords[MSG_PERMUTATION_2D[round][11]]);
  blake3_g(2, 7, 8, 13, blockWords[MSG_PERMUTATION_2D[round][12]], blockWords[MSG_PERMUTATION_2D[round][13]]);
  blake3_g(3, 4, 9, 14, blockWords[MSG_PERMUTATION_2D[round][14]], blockWords[MSG_PERMUTATION_2D[round][15]]);
}
*/


fn blake3_round_fn_0(){
  blake3_g(0, 4, 8, 12, blockWords[MSG_PERMUTATION_2D[0][0]], blockWords[MSG_PERMUTATION_2D[0][1]]);
  blake3_g(1, 5, 9, 13, blockWords[MSG_PERMUTATION_2D[0][2]], blockWords[MSG_PERMUTATION_2D[0][3]]);
  blake3_g(2, 6, 10, 14, blockWords[MSG_PERMUTATION_2D[0][4]], blockWords[MSG_PERMUTATION_2D[0][5]]);
  blake3_g(3, 7, 11, 15, blockWords[MSG_PERMUTATION_2D[0][6]], blockWords[MSG_PERMUTATION_2D[0][7]]);

  // Mix diagonals
  blake3_g(0, 5, 10, 15, blockWords[MSG_PERMUTATION_2D[0][8]], blockWords[MSG_PERMUTATION_2D[0][9]]);
  blake3_g(1, 6, 11, 12, blockWords[MSG_PERMUTATION_2D[0][10]], blockWords[MSG_PERMUTATION_2D[0][11]]);
  blake3_g(2, 7, 8, 13, blockWords[MSG_PERMUTATION_2D[0][12]], blockWords[MSG_PERMUTATION_2D[0][13]]);
  blake3_g(3, 4, 9, 14, blockWords[MSG_PERMUTATION_2D[0][14]], blockWords[MSG_PERMUTATION_2D[0][15]]);
}

fn blake3_round_fn_1(){
  blake3_g(0, 4, 8, 12, blockWords[MSG_PERMUTATION_2D[1][0]], blockWords[MSG_PERMUTATION_2D[1][1]]);
  blake3_g(1, 5, 9, 13, blockWords[MSG_PERMUTATION_2D[1][2]], blockWords[MSG_PERMUTATION_2D[1][3]]);
  blake3_g(2, 6, 10, 14, blockWords[MSG_PERMUTATION_2D[1][4]], blockWords[MSG_PERMUTATION_2D[1][5]]);
  blake3_g(3, 7, 11, 15, blockWords[MSG_PERMUTATION_2D[1][6]], blockWords[MSG_PERMUTATION_2D[1][7]]);

  // Mix diagonals
  blake3_g(0, 5, 10, 15, blockWords[MSG_PERMUTATION_2D[1][8]], blockWords[MSG_PERMUTATION_2D[1][9]]);
  blake3_g(1, 6, 11, 12, blockWords[MSG_PERMUTATION_2D[1][10]], blockWords[MSG_PERMUTATION_2D[1][11]]);
  blake3_g(2, 7, 8, 13, blockWords[MSG_PERMUTATION_2D[1][12]], blockWords[MSG_PERMUTATION_2D[1][13]]);
  blake3_g(3, 4, 9, 14, blockWords[MSG_PERMUTATION_2D[1][14]], blockWords[MSG_PERMUTATION_2D[1][15]]);
}

fn blake3_round_fn_2(){
  blake3_g(0, 4, 8, 12, blockWords[MSG_PERMUTATION_2D[2][0]], blockWords[MSG_PERMUTATION_2D[2][1]]);
  blake3_g(1, 5, 9, 13, blockWords[MSG_PERMUTATION_2D[2][2]], blockWords[MSG_PERMUTATION_2D[2][3]]);
  blake3_g(2, 6, 10, 14, blockWords[MSG_PERMUTATION_2D[2][4]], blockWords[MSG_PERMUTATION_2D[2][5]]);
  blake3_g(3, 7, 11, 15, blockWords[MSG_PERMUTATION_2D[2][6]], blockWords[MSG_PERMUTATION_2D[2][7]]);

  // Mix diagonals
  blake3_g(0, 5, 10, 15, blockWords[MSG_PERMUTATION_2D[2][8]], blockWords[MSG_PERMUTATION_2D[2][9]]);
  blake3_g(1, 6, 11, 12, blockWords[MSG_PERMUTATION_2D[2][10]], blockWords[MSG_PERMUTATION_2D[2][11]]);
  blake3_g(2, 7, 8, 13, blockWords[MSG_PERMUTATION_2D[2][12]], blockWords[MSG_PERMUTATION_2D[2][13]]);
  blake3_g(3, 4, 9, 14, blockWords[MSG_PERMUTATION_2D[2][14]], blockWords[MSG_PERMUTATION_2D[2][15]]);
}

fn blake3_round_fn_3(){
  blake3_g(0, 4, 8, 12, blockWords[MSG_PERMUTATION_2D[3][0]], blockWords[MSG_PERMUTATION_2D[3][1]]);
  blake3_g(1, 5, 9, 13, blockWords[MSG_PERMUTATION_2D[3][2]], blockWords[MSG_PERMUTATION_2D[3][3]]);
  blake3_g(2, 6, 10, 14, blockWords[MSG_PERMUTATION_2D[3][4]], blockWords[MSG_PERMUTATION_2D[3][5]]);
  blake3_g(3, 7, 11, 15, blockWords[MSG_PERMUTATION_2D[3][6]], blockWords[MSG_PERMUTATION_2D[3][7]]);

  // Mix diagonals
  blake3_g(0, 5, 10, 15, blockWords[MSG_PERMUTATION_2D[3][8]], blockWords[MSG_PERMUTATION_2D[3][9]]);
  blake3_g(1, 6, 11, 12, blockWords[MSG_PERMUTATION_2D[3][10]], blockWords[MSG_PERMUTATION_2D[3][11]]);
  blake3_g(2, 7, 8, 13, blockWords[MSG_PERMUTATION_2D[3][12]], blockWords[MSG_PERMUTATION_2D[3][13]]);
  blake3_g(3, 4, 9, 14, blockWords[MSG_PERMUTATION_2D[3][14]], blockWords[MSG_PERMUTATION_2D[3][15]]);
}

fn blake3_round_fn_4(){
  blake3_g(0, 4, 8, 12, blockWords[MSG_PERMUTATION_2D[4][0]], blockWords[MSG_PERMUTATION_2D[4][1]]);
  blake3_g(1, 5, 9, 13, blockWords[MSG_PERMUTATION_2D[4][2]], blockWords[MSG_PERMUTATION_2D[4][3]]);
  blake3_g(2, 6, 10, 14, blockWords[MSG_PERMUTATION_2D[4][4]], blockWords[MSG_PERMUTATION_2D[4][5]]);
  blake3_g(3, 7, 11, 15, blockWords[MSG_PERMUTATION_2D[4][6]], blockWords[MSG_PERMUTATION_2D[4][7]]);

  // Mix diagonals
  blake3_g(0, 5, 10, 15, blockWords[MSG_PERMUTATION_2D[4][8]], blockWords[MSG_PERMUTATION_2D[4][9]]);
  blake3_g(1, 6, 11, 12, blockWords[MSG_PERMUTATION_2D[4][10]], blockWords[MSG_PERMUTATION_2D[4][11]]);
  blake3_g(2, 7, 8, 13, blockWords[MSG_PERMUTATION_2D[4][12]], blockWords[MSG_PERMUTATION_2D[4][13]]);
  blake3_g(3, 4, 9, 14, blockWords[MSG_PERMUTATION_2D[4][14]], blockWords[MSG_PERMUTATION_2D[4][15]]);
}

fn blake3_round_fn_5(){
  blake3_g(0, 4, 8, 12, blockWords[MSG_PERMUTATION_2D[5][0]], blockWords[MSG_PERMUTATION_2D[5][1]]);
  blake3_g(1, 5, 9, 13, blockWords[MSG_PERMUTATION_2D[5][2]], blockWords[MSG_PERMUTATION_2D[5][3]]);
  blake3_g(2, 6, 10, 14, blockWords[MSG_PERMUTATION_2D[5][4]], blockWords[MSG_PERMUTATION_2D[5][5]]);
  blake3_g(3, 7, 11, 15, blockWords[MSG_PERMUTATION_2D[5][6]], blockWords[MSG_PERMUTATION_2D[5][7]]);

  // Mix diagonals
  blake3_g(0, 5, 10, 15, blockWords[MSG_PERMUTATION_2D[5][8]], blockWords[MSG_PERMUTATION_2D[5][9]]);
  blake3_g(1, 6, 11, 12, blockWords[MSG_PERMUTATION_2D[5][10]], blockWords[MSG_PERMUTATION_2D[5][11]]);
  blake3_g(2, 7, 8, 13, blockWords[MSG_PERMUTATION_2D[5][12]], blockWords[MSG_PERMUTATION_2D[5][13]]);
  blake3_g(3, 4, 9, 14, blockWords[MSG_PERMUTATION_2D[5][14]], blockWords[MSG_PERMUTATION_2D[5][15]]);
}

fn blake3_round_fn_6(){
  blake3_g(0, 4, 8, 12, blockWords[MSG_PERMUTATION_2D[6][0]], blockWords[MSG_PERMUTATION_2D[6][1]]);
  blake3_g(1, 5, 9, 13, blockWords[MSG_PERMUTATION_2D[6][2]], blockWords[MSG_PERMUTATION_2D[6][3]]);
  blake3_g(2, 6, 10, 14, blockWords[MSG_PERMUTATION_2D[6][4]], blockWords[MSG_PERMUTATION_2D[6][5]]);
  blake3_g(3, 7, 11, 15, blockWords[MSG_PERMUTATION_2D[6][6]], blockWords[MSG_PERMUTATION_2D[6][7]]);

  // Mix diagonals
  blake3_g(0, 5, 10, 15, blockWords[MSG_PERMUTATION_2D[6][8]], blockWords[MSG_PERMUTATION_2D[6][9]]);
  blake3_g(1, 6, 11, 12, blockWords[MSG_PERMUTATION_2D[6][10]], blockWords[MSG_PERMUTATION_2D[6][11]]);
  blake3_g(2, 7, 8, 13, blockWords[MSG_PERMUTATION_2D[6][12]], blockWords[MSG_PERMUTATION_2D[6][13]]);
  blake3_g(3, 4, 9, 14, blockWords[MSG_PERMUTATION_2D[6][14]], blockWords[MSG_PERMUTATION_2D[6][15]]);
}

fn blake3_two_to_one() {

  blake3_round_fn_0();
  blake3_round_fn_1();
  blake3_round_fn_2();
  blake3_round_fn_3();
  blake3_round_fn_4();
  blake3_round_fn_5();
  blake3_round_fn_6();
  /* 
  blake3_round_fn(0);
  blake3_round_fn(1);
  blake3_round_fn(2);
  blake3_round_fn(3);
  blake3_round_fn(4);
  blake3_round_fn(5);
  blake3_round_fn(6);*/

  state[0] ^= state[8];
  state[8] ^= 1779033703u;
  state[1] ^= state[9];
  state[9] ^= 3144134277u;
  state[2] ^= state[10];
  state[10] ^= 1013904242u;
  state[3] ^= state[11];
  state[11] ^= 2773480762u;
  state[4] ^= state[12];
  state[12] ^= 1359893119u;
  state[5] ^= state[13];
  state[13] ^= 2600822924u;
  state[6] ^= state[14];
  state[14] ^= 528734635u;
  state[7] ^= state[15];
  state[15] ^= 1541459225u;
}


fn blake3_hash() {
  let chainingValue = array<u32,8>(state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]);

  blake3_round_fn_0();
  blake3_round_fn_1();
  blake3_round_fn_2();
  blake3_round_fn_3();
  blake3_round_fn_4();
  blake3_round_fn_5();
  blake3_round_fn_6();
  /* 
  blake3_round_fn(0);
  blake3_round_fn(1);
  blake3_round_fn(2);
  blake3_round_fn(3);
  blake3_round_fn(4);
  blake3_round_fn(5);
  blake3_round_fn(6);*/

  state[0] ^= state[8];
  state[8] ^= chainingValue[0];
  state[1] ^= state[9];
  state[9] ^= chainingValue[1];
  state[2] ^= state[10];
  state[10] ^= chainingValue[2];
  state[3] ^= state[11];
  state[11] ^= chainingValue[3];
  state[4] ^= state[12];
  state[12] ^= chainingValue[4];
  state[5] ^= state[13];
  state[13] ^= chainingValue[5];
  state[6] ^= state[14];
  state[14] ^= chainingValue[6];
  state[7] ^= state[15];
  state[15] ^= chainingValue[7];
}

fn copyToBlockWordsFromArrayTwoToOne_fp(a: Blake3HashOut, b: Blake3HashOut){
  var fp = mont_to_fp(a[0], a[1]);

  blockWords[0] = fp.x;
  blockWords[1] = fp.y;

  fp = mont_to_fp(a[2], a[3]);
  blockWords[2] = fp.x;
  blockWords[3] = fp.y;

  fp = mont_to_fp(a[4], a[5]);
  blockWords[4] = fp.x;
  blockWords[5] = fp.y;

  fp = mont_to_fp(a[6], a[7]);
  blockWords[6] = fp.x;
  blockWords[7] = fp.y;



  fp = mont_to_fp(b[0], b[1]);
  blockWords[8] = fp.x;
  blockWords[9] = fp.y;


  fp = mont_to_fp(b[2], b[3]);
  blockWords[10] = fp.x;
  blockWords[11] = fp.y;

  fp = mont_to_fp(b[4], b[5]);
  blockWords[12] =fp.x;
  blockWords[13] = fp.y;


  fp = mont_to_fp(b[6], b[7]);
  blockWords[14] = fp.x;
  blockWords[15] = fp.y;

}
fn copyToBlockWordsFromArrayTwoToOne(a: Blake3HashOut, b: Blake3HashOut) {
  blockWords[0] = a[0];
  blockWords[1] = a[1];
  blockWords[2] = a[2];
  blockWords[3] = a[3];

  blockWords[4] = a[4];
  blockWords[5] = a[5];
  blockWords[6] = a[6];
  blockWords[7] = a[7];

  blockWords[8] = b[0];
  blockWords[9] = b[1];
  blockWords[10] = b[2];
  blockWords[11] = b[3];

  blockWords[12] = b[4];
  blockWords[13] = b[5];
  blockWords[14] = b[6];
  blockWords[15] = b[7];
}

/* 
fn getFinalizedHashOut() -> Blake3HashOut {
  return array<u32,8>(
    state[0],
    state[1],
    state[2],
    state[3],
    state[4],
    state[5],
    state[6],
    state[7],
  );
}*/

@group(0) @binding(0) var<storage, read_write>digests: array<Blake3HashOut>;
@group(0) @binding(1) var<uniform> node_count: u32;

@group(0) @binding(2) var<storage, read> row_or_leaf_inputs: array<Blake3HashOut>;
@group(0) @binding(3) var<uniform> absorb_params: vec2<u32>;

//@group(0) @binding(3) var<storage, read_write>row_hash_state: array<Blake3HashOut>;



@compute @workgroup_size(1) fn blake3_256_absorb_row_first(
  @builtin(global_invocation_id) id: vec3<u32>
) {
  let i = id.x+id.y*32768u; // parent node index
  let base = i * 2u;
  state[0] = 1779033703u;
  state[1] = 3144134277u;
  state[2] = 1013904242u;
  state[3] = 2773480762u;
  state[4] = 1359893119u;
  state[5] = 2600822924u;
  state[6] = 528734635u;
  state[7] = 1541459225u;
  state[8] = 1779033703u;
  state[9] = 3144134277u;
  state[10] = 1013904242u;
  state[11] = 2773480762u;
  state[12] = 0u;
  state[13] = 0u;
  state[14] = absorb_params.x; // block length
  state[15] = absorb_params.y; // flags
  copyToBlockWordsFromArrayTwoToOne_fp(row_or_leaf_inputs[base], row_or_leaf_inputs[base + 1]);
  blake3_two_to_one();
  digests[i][0] = state[0];
  digests[i][1] = state[1];
  digests[i][2] = state[2];
  digests[i][3] = state[3];
  digests[i][4] = state[4];
  digests[i][5] = state[5];
  digests[i][6] = state[6];
  digests[i][7] = state[7];
  
  //digests[i] = getFinalizedHashOut();
}


@compute @workgroup_size(1) fn blake3_256_absorb_row(
  @builtin(global_invocation_id) id: vec3<u32>
) {
  let i = id.x+id.y*32768u; // parent node index
  let base = i * 2u;
  let dgst = digests[i];
  state[0] = dgst[0];
  state[1] = dgst[1];
  state[2] = dgst[2];
  state[3] = dgst[3];
  state[4] = dgst[4];
  state[5] = dgst[5];
  state[6] = dgst[6];
  state[7] = dgst[7];
  state[8] = 1779033703u;
  state[9] = 3144134277u;
  state[10] = 1013904242u;
  state[11] = 2773480762u;
  state[12] = 0u;
  state[13] = 0u;
  state[14] = absorb_params.x; // block length
  state[15] = absorb_params.y; // flags
  copyToBlockWordsFromArrayTwoToOne_fp(row_or_leaf_inputs[base], row_or_leaf_inputs[base + 1]);
  blake3_hash();
  digests[i][0] = state[0];
  digests[i][1] = state[1];
  digests[i][2] = state[2];
  digests[i][3] = state[3];
  digests[i][4] = state[4];
  digests[i][5] = state[5];
  digests[i][6] = state[6];
  digests[i][7] = state[7];
  
  //digests[i] = getFinalizedHashOut();
}


@compute @workgroup_size(1) fn blake3_256_hash_leaves(
  @builtin(global_invocation_id) id: vec3<u32>
) {
  let i = id.x+id.y*32768u; // parent node index
  let base = i * 2u;
  state[0] = 1779033703u;
  state[1] = 3144134277u;
  state[2] = 1013904242u;
  state[3] = 2773480762u;
  state[4] = 1359893119u;
  state[5] = 2600822924u;
  state[6] = 528734635u;
  state[7] = 1541459225u;
  state[8] = 1779033703u;
  state[9] = 3144134277u;
  state[10] = 1013904242u;
  state[11] = 2773480762u;
  state[12] = 0u;
  state[13] = 0u;
  state[14] = 64u;
  state[15] = 11u;
  copyToBlockWordsFromArrayTwoToOne(row_or_leaf_inputs[base], row_or_leaf_inputs[base + 1]);
  blake3_two_to_one();
  let node_offset = i+node_count;
  digests[node_offset][0] = state[0];
  digests[node_offset][1] = state[1];
  digests[node_offset][2] = state[2];
  digests[node_offset][3] = state[3];
  digests[node_offset][4] = state[4];
  digests[node_offset][5] = state[5];
  digests[node_offset][6] = state[6];
  digests[node_offset][7] = state[7];
  
  //digests[i] = getFinalizedHashOut();
}


@compute @workgroup_size(1) fn blake3_256_hash_level(
  @builtin(global_invocation_id) id: vec3<u32>
) {
  let i = id.x+id.y*32768u; // parent node index
  let child_nodes_offset = node_count*2+i*2;
  state[0] = 1779033703u;
  state[1] = 3144134277u;
  state[2] = 1013904242u;
  state[3] = 2773480762u;
  state[4] = 1359893119u;
  state[5] = 2600822924u;
  state[6] = 528734635u;
  state[7] = 1541459225u;
  state[8] = 1779033703u;
  state[9] = 3144134277u;
  state[10] = 1013904242u;
  state[11] = 2773480762u;
  state[12] = 0u;
  state[13] = 0u;
  state[14] = 64u;
  state[15] = 11u;
  copyToBlockWordsFromArrayTwoToOne(digests[child_nodes_offset], digests[child_nodes_offset + 1]);
  blake3_two_to_one();
  let node_offset = i+node_count;
  digests[node_offset][0] = state[0];
  digests[node_offset][1] = state[1];
  digests[node_offset][2] = state[2];
  digests[node_offset][3] = state[3];
  digests[node_offset][4] = state[4];
  digests[node_offset][5] = state[5];
  digests[node_offset][6] = state[6];
  digests[node_offset][7] = state[7];
  
  //digests[i] = getFinalizedHashOut();
}