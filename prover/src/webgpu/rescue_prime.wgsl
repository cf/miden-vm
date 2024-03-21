
alias Felt = vec2<u32>;
alias RPXState = array<Felt, 12>;
alias RPXHashOut = array<Felt, 4>;
const GoldilocksPrime = Felt(1u, 4294967295u);

// start RPX constants
const ARK1 = array<array<Felt, 12>, 7>(
  array<Felt, 12>(
    Felt(1331982684u, 1614950527u),
    Felt(1176470353u, 1599843990u),
    Felt(3459224179u, 984021449u),
    Felt(3743538108u, 526482491u),
    Felt(3188951565u, 3913433156u),
    Felt(1906237837u, 2994712265u),
    Felt(1891987107u, 80494398u),
    Felt(1158423702u, 2357728787u),
    Felt(1780433750u, 195729014u),
    Felt(1345089187u, 4227180175u),
    Felt(2494369337u, 3859712537u),
    Felt(1510167429u, 3201380850u)
  ),
  array<Felt, 12>(
    Felt(1829659575u, 3736459171u),
    Felt(1703959935u, 2438745984u),
    Felt(3883146292u, 3672634983u),
    Felt(1700835u, 3353047920u),
    Felt(1588809343u, 1399171365u),
    Felt(725308178u, 490941908u),
    Felt(369917172u, 3217766228u),
    Felt(987461314u, 3717982761u),
    Felt(600559126u, 1753214218u),
    Felt(1348124943u, 2195405311u),
    Felt(3991433165u, 2710126136u),
    Felt(4251893370u, 3669333383u)
  ),
  array<Felt, 12>(
    Felt(934359705u, 3447676605u),
    Felt(3127906046u, 4018251751u),
    Felt(2253163445u, 2219838233u),
    Felt(1743551786u, 83520855u),
    Felt(2409482763u, 1658473753u),
    Felt(2199856050u, 170942672u),
    Felt(2783383087u, 3462933216u),
    Felt(3899213960u, 4182886323u),
    Felt(4170310375u, 2948762261u),
    Felt(342217289u, 3626451611u),
    Felt(545009027u, 1681381813u),
    Felt(3841153895u, 1706405882u)
  ),
  array<Felt, 12>(
    Felt(3755662572u, 3513031408u),
    Felt(3346111551u, 3902931605u),
    Felt(1786286980u, 3557224135u),
    Felt(2522999907u, 1457006049u),
    Felt(2645799376u, 1960190275u),
    Felt(2542367324u, 1518865061u),
    Felt(4088198891u, 3537381297u),
    Felt(2036627120u, 367426831u),
    Felt(2736322608u, 2524013031u),
    Felt(3661923946u, 1735383971u),
    Felt(3422276516u, 1204449097u),
    Felt(2074557216u, 764730433u)
  ),
  array<Felt, 12>(
    Felt(2755445318u, 401536633u),
    Felt(3694728775u, 4190400650u),
    Felt(1761787606u, 299146706u),
    Felt(1619951767u, 223641838u),
    Felt(901788741u, 1562037926u),
    Felt(2417753732u, 2089792654u),
    Felt(2936597615u, 12227174u),
    Felt(2295714421u, 2330492956u),
    Felt(3549702550u, 719732678u),
    Felt(2380449367u, 247698796u),
    Felt(2816114216u, 790820609u),
    Felt(3355952610u, 3503183319u)
  ),
  array<Felt, 12>(
    Felt(3659436100u, 1133294911u),
    Felt(2584265397u, 3217011672u),
    Felt(2428433805u, 3243610685u),
    Felt(2515614149u, 2676182592u),
    Felt(2975254823u, 3916928950u),
    Felt(1436857112u, 1574114716u),
    Felt(715783963u, 1244885266u),
    Felt(3370246269u, 3521469435u),
    Felt(1486769285u, 1241584699u),
    Felt(3008338110u, 4107223196u),
    Felt(2227255588u, 3271227362u),
    Felt(1621259290u, 2068166979u)
  ),
  array<Felt, 12>(
    Felt(189928787u, 2446568213u),
    Felt(1569994391u, 2484178161u),
    Felt(2892222465u, 3900766099u),
    Felt(572813793u, 3019221273u),
    Felt(943694311u, 696120579u),
    Felt(528872197u, 1372460853u),
    Felt(913193488u, 3303274209u),
    Felt(2040613025u, 3444897795u),
    Felt(3733849933u, 4018789878u),
    Felt(2975446048u, 3727580470u),
    Felt(487218973u, 467778236u),
    Felt(2099739061u, 2523522374u)
  )
);
const ARK2 = array<array<Felt, 12>, 7>(
array<Felt, 12>(
    Felt(798739039u, 2081301822u),
    Felt(1608352816u, 3424483637u),
    Felt(3122161457u, 4220094687u),
    Felt(3108378249u, 2167611716u),
    Felt(4280926982u, 1097672419u),
    Felt(1287425336u, 303257728u),
    Felt(824418480u, 8109485u),
    Felt(619625962u, 1312520746u),
    Felt(3204260869u, 52880374u),
    Felt(75666391u, 589288058u),
    Felt(273171151u, 1678747078u),
    Felt(766187825u, 222565973u)
  ),
  array<Felt, 12>(
    Felt(1814723554u, 1036007259u),
    Felt(1946388766u, 2524739057u),
    Felt(384818995u, 3073121770u),
    Felt(1520599029u, 2687290550u),
    Felt(1454615313u, 2552635148u),
    Felt(4183461085u, 1106621264u),
    Felt(3428170867u, 702051727u),
    Felt(1186501262u, 3805450749u),
    Felt(3822121358u, 421674913u),
    Felt(1462438804u, 2965253070u),
    Felt(3116781634u, 1344688080u),
    Felt(3672374074u, 4243018422u)
  ),
  array<Felt, 12>(
    Felt(745265721u, 1681614110u),
    Felt(4112763032u, 981524372u),
    Felt(1812292203u, 1438313228u),
    Felt(574077583u, 812896080u),
    Felt(3825698566u, 2552239534u),
    Felt(3366110356u, 450622485u),
    Felt(2444393806u, 3253620264u),
    Felt(17973656u, 4194108042u),
    Felt(421386701u, 1761521427u),
    Felt(3460204046u, 2157368022u),
    Felt(311801468u, 1876984674u),
    Felt(3993559444u, 1584843096u)
  ),
  array<Felt, 12>(
    Felt(2430239846u, 1878115010u),
    Felt(1715943950u, 2978329577u),
    Felt(1103101654u, 2158048751u),
    Felt(509620894u, 4073659624u),
    Felt(3392365298u, 3162746910u),
    Felt(1390564890u, 3241068333u),
    Felt(228248579u, 915755830u),
    Felt(2622437550u, 1550199862u),
    Felt(720281290u, 3914433074u),
    Felt(3787288385u, 3529806124u),
    Felt(1389908094u, 3821009013u),
    Felt(950405748u, 789488301u)
  ),
  array<Felt, 12>(
    Felt(2678884064u, 4285358623u),
    Felt(2087099076u, 3227973257u),
    Felt(348955916u, 115517539u),
    Felt(352990304u, 2279766758u),
    Felt(2073625869u, 123070166u),
    Felt(1492303021u, 3693737407u),
    Felt(3811228150u, 158983890u),
    Felt(3134942171u, 80829214u),
    Felt(1611942137u, 3202067555u),
    Felt(3691936594u, 2387613753u),
    Felt(653873076u, 2047547745u),
    Felt(1983696925u, 577500118u)
  ),
  array<Felt, 12>(
    Felt(4119788819u, 3600106006u),
    Felt(2593482635u, 1566942956u),
    Felt(583595801u, 2153734876u),
    Felt(2555543715u, 2801289778u),
    Felt(1861755704u, 3385078902u),
    Felt(1547661682u, 944231923u),
    Felt(1395674829u, 3437520857u),
    Felt(956480934u, 476787616u),
    Felt(3691695488u, 2090321642u),
    Felt(3230995191u, 1527474021u),
    Felt(907810747u, 2322774643u),
    Felt(422724091u, 4162432134u)
  ),
  array<Felt, 12>(
    Felt(3774991629u, 826461355u),
    Felt(3183570523u, 990375395u),
    Felt(1388173503u, 665709349u),
    Felt(3654100769u, 253228405u),
    Felt(677455766u, 1841112417u),
    Felt(851019676u, 2632477040u),
    Felt(3951039724u, 2367424300u),
    Felt(2909894070u, 2136789281u),
    Felt(1691246059u, 3066155318u),
    Felt(3335376689u, 3009146593u),
    Felt(1201962425u, 2252172226u),
    Felt(4165172641u, 174117338u)
  )
);
// end RPX constants

// start misc math


fn hadd(a:u32, b:u32) -> u32 {
  return (a >> 1u) + (b >> 1u) + ((a & b) & 1u);
}

fn mul64(a:u32, b:u32) -> vec2<u32>{
  // Split into 16 bit parts
  var a0 = (a << 16u) >> 16u;
  var a1 = a >> 16u;
  var b0 = (b << 16u) >> 16u;
  var b1 = b >> 16u;

  // Compute 32 bit half products
  // Each of these is at most 0xfffe0001
  var a0b0 = a0 * b0;
  var a0b1 = a0 * b1;
  var a1b0 = a1 * b0;
  var a1b1 = a1 * b1;

  // Sum the half products
  var r: vec2<u32>;
  r.x = a0b0 + (a1b0 << 16u) + (a0b1 << 16u);
  r.y = a1b1 + (hadd((a0b0 >> 16u) + a0b1, a1b0) >> 15u);
  return r;
}

fn mul128(a: vec2<u32>, b: vec2<u32>) -> vec4 <u32>{
  // Compute 64 bit half products
  // Each of these is at most 0xfffffffe00000001
  
  var a0b0 = mul64(a.x, b.x);
  var a0b1 = mul64(a.x, b.y);
  var a1b0 = mul64(a.y, b.x);
  var a1b1 = mul64(a.y, b.y);

  var r = vec4 <u32>(a0b0, a1b1);

  // Add a0b1
  r.y += a0b1.x;
  if (r.y<a0b1.x) {
    a0b1.y += 1u; // Can not overflow
  }
  r.z += a0b1.y;
  if (r.z<a0b1.y) {
    r.w += 1u;
  }

  // Add a1b0
  r.y += a1b0.x;
  if (r.y<a1b0.x) {
    a1b0.y += 1u; // Can not overflow
  }
  r.z += a1b0.y;
  if (r.z<a1b0.y) {
    r.w += 1u;
  }

  return r;
}
//end misc math

// start goldilocks fp math

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
fn neg_w_gl(x: vec2<u32>) -> vec2<u32> {
  let result = GoldilocksPrime-x;
  return sub_w(GoldilocksPrime, x);
}
fn gl_fp_add(a: Felt, b: Felt) -> Felt {

  // We compute a + b = a - (p - b).


  // We compute a + b = a - (p - b).
  let x1 = sub_w_overflowing(a, sub_w(GoldilocksPrime, b));
  let adj = vec2<u32>(0xffffffffu*x1.z,0u);
  return sub_w(x1.xy, adj);

  
}

fn gl_fp_sub(a: Felt, b: Felt) -> Felt {


  let result = sub_w(a, b);
  let overflow =select(0u,0xffffffffu, result.y> a.y || (result.y == a.y && result.x > a.x));
  

  return sub_w(result.xy, vec2<u32>(overflow, 0u));
}
fn gl_fp_double(x: Felt) -> Felt{
  return gl_fp_add(x,x);
}

fn gl_fp_mul(a: Felt, b: Felt) -> Felt {
  return mul_mont(a,b);
}
fn gl_fp_square(a: Felt) -> Felt {
  return gl_fp_mul(a,a);
}
fn gl_fp_pow7(a: Felt) -> Felt {
  let t2 = gl_fp_square(a);
  return gl_fp_mul(gl_fp_mul(gl_fp_square(t2), t2), a);
}

// start cubic extension


fn gl_ext_mul(a: array<Felt,3>, b: array<Felt,3>) -> array<Felt, 3> {
  let a0b0 = gl_fp_mul(a[0] , b[0]);
  let a1b1 = gl_fp_mul(a[1] , b[1]);
  let a2b2 = gl_fp_mul(a[2] , b[2]);

  let a0b0_a0b1_a1b0_a1b1 = gl_fp_mul(gl_fp_add(a[0] , a[1]) , gl_fp_add(b[0] , b[1]));
  let a0b0_a0b2_a2b0_a2b2 = gl_fp_mul(gl_fp_add(a[0] , a[2]) , gl_fp_add(b[0] , b[2]));
  let a1b1_a1b2_a2b1_a2b2 = gl_fp_mul(gl_fp_add(a[1] , a[2]) , gl_fp_add(b[1] , b[2]));

  let a0b0_minus_a1b1 = gl_fp_sub(a0b0 , a1b1);

  let a0b0_a1b2_a2b1 = gl_fp_sub(gl_fp_add(a1b1_a1b2_a2b1_a2b2 , a0b0_minus_a1b1) , a2b2);


  //let a0b1_a1b0_a1b2_a2b1_a2b2 = gl_fp_sub(gl_fp_sub(gl_fp_add(a0b0_a0b1_a1b0_a1b1 , a1b1_a1b2_a2b1_a2b2) , gl_fp_double(a1b1)) , a0b0);
  let t0 = gl_fp_add(a0b0_a0b1_a1b0_a1b1 , a1b1_a1b2_a2b1_a2b2);
  let t1 = gl_fp_add(gl_fp_add(a1b1, a1b1), a0b0);

  let t3 = gl_fp_sub(t0, t1);


  let a0b2_a1b1_a2b0_a2b2 = gl_fp_sub(a0b0_a0b2_a2b0_a2b2 , a0b0_minus_a1b1);
  return array<Felt, 3>(a0b0_a1b2_a2b1, t3, a0b2_a1b1_a2b0_a2b2);
}
fn gl_ext_square(a0: Felt, a1: Felt, a2: Felt) -> array<Felt, 3> {

  let a2_sq = gl_fp_square(a2);
  let a1_a2 = gl_fp_mul(a1, a2);

  let out0 = gl_fp_add(gl_fp_square(a0), gl_fp_double(a1_a2));
  let out1 = gl_fp_add(gl_fp_double(gl_fp_add(gl_fp_mul(a0 , a1) , a1_a2)) , a2_sq);
  let out2 = gl_fp_add(gl_fp_add(gl_fp_double(gl_fp_mul(a0 , a2)) , gl_fp_square(a1)) , a2_sq);

  return array<Felt, 3>(out0, out1, out2);
}

fn gl_ext_exp7(x: array<Felt,3>) -> array<Felt, 3> {

  let x2 = gl_ext_square(x[0],x[1],x[2]);
  let x3 = gl_ext_mul(x2 , x);
  let x4 = gl_ext_square(x2[0],x2[1],x2[2]);
  return gl_ext_mul(x3 , x4);
}
fn exp_acc_3(value: Felt, tail: Felt) -> Felt {
  var x = gl_fp_mul(value,value);
  x = gl_fp_mul(x,x);
  x = gl_fp_mul(x,x);
  return gl_fp_mul(x, tail);
}
fn exp_acc_6(value: Felt, tail: Felt) -> Felt {
  var x = gl_fp_mul(value,value);
  x = gl_fp_mul(x,x);
  x = gl_fp_mul(x,x);
  
  x = gl_fp_mul(x,x);
  x = gl_fp_mul(x,x);
  x = gl_fp_mul(x,x);

  return gl_fp_mul(x, tail);
}
fn exp_acc_12(value: Felt, tail: Felt) -> Felt {
  var x = gl_fp_mul(value,value);
  x = gl_fp_mul(x,x);
  x = gl_fp_mul(x,x);
  
  x = gl_fp_mul(x,x);
  x = gl_fp_mul(x,x);
  x = gl_fp_mul(x,x);


  x = gl_fp_mul(x,x);
  x = gl_fp_mul(x,x);
  x = gl_fp_mul(x,x);


  x = gl_fp_mul(x,x);
  x = gl_fp_mul(x,x);
  x = gl_fp_mul(x,x);

  return gl_fp_mul(x, tail);
}
fn exp_acc_31(value: Felt, tail: Felt) -> Felt {
  var x = gl_fp_mul(value,value);

  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);

  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);

  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);
  x = gl_fp_square(x);

  return gl_fp_mul(x, tail);
}

fn mul_mont(x: vec2<u32>, y: vec2<u32>) -> vec2<u32> {
  let combo = mul128(x,y);
  let xl = combo.xy;
  let xh = combo.zw;
  let tmp = vec2<u32>(0u, xl.x);
  let a_s = add_w_overflowing(xl, tmp);
  let a_overflow = a_s.z;
  let a = a_s.xy;
  let alt_shf_a = add_w(vec2<u32>(a_overflow, 0u), vec2<u32>(a.y,0u));

  let b = sub_w(a, alt_shf_a);
  let r_s = sub_w_overflowing(xh, b);
  let r = r_s.xy;
  let adj = vec2<u32>(r_s.z*0xffffffffu, 0u);
  return sub_w(r, adj);


}
fn inv_sbox(x: Felt) -> Felt {


  let t1 = gl_fp_mul(x, x);
  let t2 = gl_fp_mul(t1, t1);
  let t3 = exp_acc_3(t2, t2);

  let t4 = exp_acc_6(t3, t3);
  let t5 = exp_acc_12(t4, t4);
  let t6 = exp_acc_6(t5, t3);
  let t7 = exp_acc_31(t6, t6);
  let a = gl_fp_square(gl_fp_square(gl_fp_mul(gl_fp_square(t7), t6)));
  let b = gl_fp_mul(gl_fp_mul(t1, t2), x);
  return gl_fp_mul(a,b);
}

// end goldilocks fp math


fn add_w_overflowing(a: vec2<u32>, b: vec2<u32>) -> vec3<u32>{
  let result = add_w(a, b);
  return vec3<u32>(result.x, result.y, select(0u, 1u, result.y < a.y));
}
fn u32_overflowing_add(a: u32, b: u32) -> vec2<u32> {
    let result = a + b;
    return vec2<u32>(result, select(0u, 1u, result < a));
}

fn u32_overflowing_add_wf(a: u32, b: u32) -> vec2<u32> {
    let result = a + b;
    return vec2<u32>(result, select(0u, 1u, result < a));
}


fn recombine_u32_v6(a: u32, b: u32, c: u32, d: u32) -> vec2<u32> {
    let b_low_16 = b & 0xFFFFu;
    let b_high_32 = b >> 16u;
    let d_low_48 = d&0xffffu;
    let d_high_64 = d>>16u;

    // without wfelt

    let v_0 = u32_overflowing_add_wf(a, b_low_16<<16u);

    let v_1 = u32_overflowing_add_wf(b_high_32, v_0.y);
    let v_2= u32_overflowing_add_wf(v_1.x, c);
    let v_3 = u32_overflowing_add_wf(v_2.x, d_low_48<<16u);

    let high_bits = d_high_64+v_1.y+v_2.y+v_3.y;

    let z_lo = 0u-high_bits;
    let z_hi = high_bits-(select(0u, 1u,high_bits != 0u));


    let v_res_lo = u32_overflowing_add_wf(v_0.x, z_lo);
    let v_res_hi_0 = u32_overflowing_add_wf(v_res_lo.y, z_hi);
    let v_res_hi_1 = u32_overflowing_add_wf(v_res_hi_0.x, v_3.x);
    let adjust_lo = (0u-1u)*(v_res_hi_0.y+v_res_hi_1.y);//select(0, 0u32.wrapping_sub(1u32), res_hi_0_carry||res_hi_1_carry);
    let v_res_lo_final = u32_overflowing_add_wf(v_res_lo.x, adjust_lo);
    let res_hi_final = v_res_hi_1.x+v_res_lo_final.y;

    return vec2<u32>(v_res_lo_final.x, res_hi_final);
}
const MDS_FREQ_BLOCK_ONE: vec3<i32> = vec3<i32>(16, 8, 16);
const MDS_FREQ_BLOCK_TWO: array<vec2<i32>, 3> = array<vec2<i32>, 3>(vec2<i32>(-1, 2), vec2<i32>(-1, 1), vec2<i32>(4, 8));
const MDS_FREQ_BLOCK_THREE: vec3<i32> = vec3<i32>(-8, 1, 1);

fn fft2_real(x: vec2<u32>) -> vec2<i32> {
    let x0 = i32(x.x);
    let x1 = i32(x.y);
    return vec2<i32>(x0 + x1, x0 - x1);
}

fn ifft2_real(y: vec2<i32>) -> vec2<u32> {
    let y0 = y.x;
    let y1 = y.y;
    return vec2<u32>(u32(y0 + y1), u32(y0 - y1));
}

fn fft4_real(x: vec4<u32>) -> vec4<i32> {
    let a = fft2_real(vec2<u32>(x.x, x.z));
    let b = fft2_real(vec2<u32>(x.y, x.w));
    let y0 = a.x + b.x;
    let y1 = vec2<i32>(a.y, -b.y);
    let y2 = a.x - b.x;
    return vec4<i32>(y0, y1.x, y1.y, y2);
}

fn ifft4_real(y: vec4<i32>) -> vec4<u32> {
    // In calculating 'z0' and 'z1', division by 2 is avoided by appropriately scaling
    // the MDS matrix constants.
    let z0 = y.x + y.w;
    let z1 = y.x - y.w;
    let z2 = y.y;
    let z3 = -y.z;
    let a = ifft2_real(vec2<i32>(z0, z2));
    let b = ifft2_real(vec2<i32>(z1, z3));
    return vec4<u32>(a.x, b.x, a.y, b.y);
}

fn block1(x: vec3<i32>, y: vec3<i32>) -> vec3<i32> {
    let x0 = x.x;
    let x1 = x.y;
    let x2 = x.z;
    let y0 = y.x;
    let y1 = y.y;
    let y2 = y.z;
    let z0 = x0 * y0 + x1 * y2 + x2 * y1;
    let z1 = x0 * y1 + x1 * y0 + x2 * y2;
    let z2 = x0 * y2 + x1 * y1 + x2 * y0;
    return vec3<i32>(z0, z1, z2);
}

fn block2(x: array<vec2<i32>, 3>, y: array<vec2<i32>, 3>) -> array<vec2<i32>, 3> {
    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    let y0 = y[0];
    let y1 = y[1];
    let y2 = y[2];
    let x0s = x0.x + x0.y;
    let x1s = x1.x + x1.y;
    let x2s = x2.x + x2.y;
    let y0s = y0.x + y0.y;
    let y1s = y1.x + y1.y;
    let y2s = y2.x + y2.y;

    // Compute x0​y0 ​− ix1​y2​ − ix2​y1​ using Karatsuba for complex numbers multiplication
    var m0 = vec2<i32>(x0.x * y0.x, x0.y * y0.y);
    var m1 = vec2<i32>(x1.x * y2.x, x1.y * y2.y);
    var m2 = vec2<i32>(x2.x * y1.x, x2.y * y1.y);
    let z0r = (m0.x - m0.y) + (x1s * y2s - m1.x - m1.y) + (x2s * y1s - m2.x - m2.y);
    let z0i = (x0s * y0s - m0.x - m0.y) + (-m1.x + m1.y) + (-m2.x + m2.y);

    // Compute x0​y1​ + x1​y0​ − ix2​y2 using Karatsuba for complex numbers multiplication
    m0 = vec2<i32>(x0.x * y1.x, x0.y * y1.y);
    m1 = vec2<i32>(x1.x * y0.x, x1.y * y0.y);
    m2 = vec2<i32>(x2.x * y2.x, x2.y * y2.y);
    let z1r = (m0.x - m0.y) + (m1.x - m1.y) + (x2s * y2s - m2.x - m2.y);
    let z1i = (x0s * y1s - m0.x - m0.y) + (x1s * y0s - m1.x - m1.y) + (-m2.x + m2.y);

    // Compute x0​y2​ + x1​y1 ​+ x2​y0​ using Karatsuba for complex numbers multiplication
    m0 = vec2<i32>(x0.x * y2.x, x0.y * y2.y);
    m1 = vec2<i32>(x1.x * y1.x, x1.y * y1.y);
    m2 = vec2<i32>(x2.x * y0.x, x2.y * y0.y);
    let z2r = (m0.x - m0.y) + (m1.x - m1.y) + (m2.x - m2.y);
    let z2i = (x0s * y2s - m0.x - m0.y) + (x1s * y1s - m1.x - m1.y) + (x2s * y0s - m2.x - m2.y);
    
    return array<vec2<i32>, 3>(vec2<i32>(z0r, z0i), vec2<i32>(z1r, z1i), vec2<i32>(z2r, z2i));
}

fn block3(x: vec3<i32>, y: vec3<i32>) -> vec3<i32> {
    let x0 = x.x;
    let x1 = x.y;
    let x2 = x.z;
    let y0 = y.x;
    let y1 = y.y;
    let y2 = y.z;
    let z0 = x0 * y0 - x1 * y2 - x2 * y1;
    let z1 = x0 * y1 + x1 * y0 - x2 * y2;
    let z2 = x0 * y2 + x1 * y1 + x2 * y0;
    return vec3<i32>(z0, z1, z2);
}

fn mds_multiply_freq_u32(state: array<u32, 12>) -> array<u32, 12> {
    let s0 = state[0];
    let s1 = state[1];
    let s2 = state[2];
    let s3 = state[3];
    let s4 = state[4];
    let s5 = state[5];
    let s6 = state[6];
    let s7 = state[7];
    let s8 = state[8];
    let s9 = state[9];
    let s10 = state[10];
    let s11 = state[11];

    let u0_u1_u2 = fft4_real(vec4<u32>(s0, s3, s6, s9));
    let u4_u5_u6 = fft4_real(vec4<u32>(s1, s4, s7, s10));
    let u8_u9_u10 = fft4_real(vec4<u32>(s2, s5, s8, s11));

    let u0 = u0_u1_u2.x;
    let u1 = vec2<i32>(u0_u1_u2.y, u0_u1_u2.z);
    let u2 = u0_u1_u2.w;
    let u4 = u4_u5_u6.x;
    let u5 = vec2<i32>(u4_u5_u6.y, u4_u5_u6.z);
    let u6 = u4_u5_u6.w;
    let u8 = u8_u9_u10.x;
    let u9 = vec2<i32>(u8_u9_u10.y, u8_u9_u10.z);
    let u10 = u8_u9_u10.w;

    let v0_v4_v8 = block1(vec3<i32>(u0, u4, u8), MDS_FREQ_BLOCK_ONE);
    let v1_v5_v9 = block2(array<vec2<i32>, 3>(u1, u5, u9), MDS_FREQ_BLOCK_TWO);
    let v2_v6_v10 = block3(vec3<i32>(u2, u6, u10), MDS_FREQ_BLOCK_THREE);

    let v0 = v0_v4_v8.x;
    let v1 = v1_v5_v9[0];
    let v2 = v2_v6_v10.x;
    let v4 = v0_v4_v8.y;
    let v5 = v1_v5_v9[1];
    let v6 = v2_v6_v10.y;
    let v8 = v0_v4_v8.z;
    let v9 = v1_v5_v9[2];
    let v10 = v2_v6_v10.z;

    let s0_s3_s6_s9 = ifft4_real(vec4<i32>(v0, v1.x, v1.y, v2));
    let s1_s4_s7_s10 = ifft4_real(vec4<i32>(v4, v5.x, v5.y, v6));
    let s2_s5_s8_s11 = ifft4_real(vec4<i32>(v8, v9.x, v9.y, v10));

    return array<u32, 12>(
        s0_s3_s6_s9.x, s1_s4_s7_s10.x, s2_s5_s8_s11.x,
        s0_s3_s6_s9.y, s1_s4_s7_s10.y, s2_s5_s8_s11.y,
        s0_s3_s6_s9.z, s1_s4_s7_s10.z, s2_s5_s8_s11.z,
        s0_s3_s6_s9.w, s1_s4_s7_s10.w, s2_s5_s8_s11.w
    );
}




const ZERO = Felt(0u, 0u);
const ONE = Felt(4294967295u, 0u);
const TWO = Felt(2u, 0u);
const FOUR = Felt(4u, 0u);

var<private> input: RPXState;


fn apply_mds(){
  var state_low_low = array<u32, 12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var state_low_high = array<u32, 12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var state_high_low = array<u32, 12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  var state_high_high = array<u32, 12>(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u);
  for(var i: u32 = 0u; i < 12u; i = i + 1u){
    state_low_low[i] = input[i].x&0xffffu;
    state_low_high[i] = ((input[i].x)>>16u);
    state_high_low[i] = input[i].y&0xffffu;
    state_high_high[i] = (input[i].y>>16u)&0xffffu;
  }
  let r_state_low_low = mds_multiply_freq_u32(state_low_low);
  let r_state_low_high = mds_multiply_freq_u32(state_low_high);
  let r_state_high_low = mds_multiply_freq_u32(state_high_low);
  let r_state_high_high = mds_multiply_freq_u32(state_high_high);
  input[0] = recombine_u32_v6(r_state_low_low[0], r_state_low_high[0], r_state_high_low[0], r_state_high_high[0]);
  input[1] = recombine_u32_v6(r_state_low_low[1], r_state_low_high[1], r_state_high_low[1], r_state_high_high[1]);
  input[2] = recombine_u32_v6(r_state_low_low[2], r_state_low_high[2], r_state_high_low[2], r_state_high_high[2]);
  input[3] = recombine_u32_v6(r_state_low_low[3], r_state_low_high[3], r_state_high_low[3], r_state_high_high[3]);
  input[4] = recombine_u32_v6(r_state_low_low[4], r_state_low_high[4], r_state_high_low[4], r_state_high_high[4]);
  input[5] = recombine_u32_v6(r_state_low_low[5], r_state_low_high[5], r_state_high_low[5], r_state_high_high[5]);
  input[6] = recombine_u32_v6(r_state_low_low[6], r_state_low_high[6], r_state_high_low[6], r_state_high_high[6]);
  input[7] = recombine_u32_v6(r_state_low_low[7], r_state_low_high[7], r_state_high_low[7], r_state_high_high[7]);
  input[8] = recombine_u32_v6(r_state_low_low[8], r_state_low_high[8], r_state_high_low[8], r_state_high_high[8]);
  input[9] = recombine_u32_v6(r_state_low_low[9], r_state_low_high[9], r_state_high_low[9], r_state_high_high[9]);
  input[10] = recombine_u32_v6(r_state_low_low[10], r_state_low_high[10], r_state_high_low[10], r_state_high_high[10]);
  input[11] = recombine_u32_v6(r_state_low_low[11], r_state_low_high[11], r_state_high_low[11], r_state_high_high[11]);
}

fn apply_sbox() {
  input[0] = gl_fp_pow7(input[0]);
  input[1] = gl_fp_pow7(input[1]);
  input[2] = gl_fp_pow7(input[2]);
  input[3] = gl_fp_pow7(input[3]);
  input[4] = gl_fp_pow7(input[4]);
  input[5] = gl_fp_pow7(input[5]);
  input[6] = gl_fp_pow7(input[6]);
  input[7] = gl_fp_pow7(input[7]);
  input[8] = gl_fp_pow7(input[8]);
  input[9] = gl_fp_pow7(input[9]);
  input[10] = gl_fp_pow7(input[10]);
  input[11] = gl_fp_pow7(input[11]);
}


fn apply_inv_sbox() {
  input[0] = inv_sbox(input[0]);
  input[1] = inv_sbox(input[1]);
  input[2] = inv_sbox(input[2]);
  input[3] = inv_sbox(input[3]);
  input[4] = inv_sbox(input[4]);
  input[5] = inv_sbox(input[5]);
  input[6] = inv_sbox(input[6]);
  input[7] = inv_sbox(input[7]);
  input[8] = inv_sbox(input[8]);
  input[9] = inv_sbox(input[9]);
  input[10] = inv_sbox(input[10]);
  input[11] = inv_sbox(input[11]);
}

fn apply_ext() {

  let ext0 = gl_ext_exp7(array<Felt, 3>(input[0], input[1], input[2]));
  let ext1 = gl_ext_exp7(array<Felt, 3>(input[3], input[4], input[5]));
  let ext2 = gl_ext_exp7(array<Felt, 3>(input[6], input[7], input[8]));
  let ext3 = gl_ext_exp7(array<Felt, 3>(input[9], input[10], input[11]));

  input[0] = ext0[0];
  input[1] = ext0[1];
  input[2] = ext0[2];

  input[3] = ext1[0];
  input[4] = ext1[1];
  input[5] = ext1[2];

  input[6] = ext2[0];
  input[7] = ext2[1];
  input[8] = ext2[2];

  input[9] = ext3[0];
  input[10] = ext3[1];
  input[11] = ext3[2];

}

fn add_constants_ark1_0(){
  input[0] = gl_fp_add(input[0], ARK1[0][0]);
  input[1] = gl_fp_add(input[1], ARK1[0][1]);
  input[2] = gl_fp_add(input[2], ARK1[0][2]);
  input[3] = gl_fp_add(input[3], ARK1[0][3]);
  input[4] = gl_fp_add(input[4], ARK1[0][4]);
  input[5] = gl_fp_add(input[5], ARK1[0][5]);
  input[6] = gl_fp_add(input[6], ARK1[0][6]);
  input[7] = gl_fp_add(input[7], ARK1[0][7]);
  input[8] = gl_fp_add(input[8], ARK1[0][8]);
  input[9] = gl_fp_add(input[9], ARK1[0][9]);
  input[10] = gl_fp_add(input[10], ARK1[0][10]);
  input[11] = gl_fp_add(input[11], ARK1[0][11]);
}
fn add_constants_ark2_0(){
  input[0] = gl_fp_add(input[0], ARK2[0][0]);
  input[1] = gl_fp_add(input[1], ARK2[0][1]);
  input[2] = gl_fp_add(input[2], ARK2[0][2]);
  input[3] = gl_fp_add(input[3], ARK2[0][3]);
  input[4] = gl_fp_add(input[4], ARK2[0][4]);
  input[5] = gl_fp_add(input[5], ARK2[0][5]);
  input[6] = gl_fp_add(input[6], ARK2[0][6]);
  input[7] = gl_fp_add(input[7], ARK2[0][7]);
  input[8] = gl_fp_add(input[8], ARK2[0][8]);
  input[9] = gl_fp_add(input[9], ARK2[0][9]);
  input[10] = gl_fp_add(input[10], ARK2[0][10]);
  input[11] = gl_fp_add(input[11], ARK2[0][11]);
}


fn add_constants_ark1_1(){
  input[0] = gl_fp_add(input[0], ARK1[1][0]);
  input[1] = gl_fp_add(input[1], ARK1[1][1]);
  input[2] = gl_fp_add(input[2], ARK1[1][2]);
  input[3] = gl_fp_add(input[3], ARK1[1][3]);
  input[4] = gl_fp_add(input[4], ARK1[1][4]);
  input[5] = gl_fp_add(input[5], ARK1[1][5]);
  input[6] = gl_fp_add(input[6], ARK1[1][6]);
  input[7] = gl_fp_add(input[7], ARK1[1][7]);
  input[8] = gl_fp_add(input[8], ARK1[1][8]);
  input[9] = gl_fp_add(input[9], ARK1[1][9]);
  input[10] = gl_fp_add(input[10], ARK1[1][10]);
  input[11] = gl_fp_add(input[11], ARK1[1][11]);
}
fn add_constants_ark2_1(){
  input[0] = gl_fp_add(input[0], ARK2[1][0]);
  input[1] = gl_fp_add(input[1], ARK2[1][1]);
  input[2] = gl_fp_add(input[2], ARK2[1][2]);
  input[3] = gl_fp_add(input[3], ARK2[1][3]);
  input[4] = gl_fp_add(input[4], ARK2[1][4]);
  input[5] = gl_fp_add(input[5], ARK2[1][5]);
  input[6] = gl_fp_add(input[6], ARK2[1][6]);
  input[7] = gl_fp_add(input[7], ARK2[1][7]);
  input[8] = gl_fp_add(input[8], ARK2[1][8]);
  input[9] = gl_fp_add(input[9], ARK2[1][9]);
  input[10] = gl_fp_add(input[10], ARK2[1][10]);
  input[11] = gl_fp_add(input[11], ARK2[1][11]);
}


fn add_constants_ark1_2(){
  input[0] = gl_fp_add(input[0], ARK1[2][0]);
  input[1] = gl_fp_add(input[1], ARK1[2][1]);
  input[2] = gl_fp_add(input[2], ARK1[2][2]);
  input[3] = gl_fp_add(input[3], ARK1[2][3]);
  input[4] = gl_fp_add(input[4], ARK1[2][4]);
  input[5] = gl_fp_add(input[5], ARK1[2][5]);
  input[6] = gl_fp_add(input[6], ARK1[2][6]);
  input[7] = gl_fp_add(input[7], ARK1[2][7]);
  input[8] = gl_fp_add(input[8], ARK1[2][8]);
  input[9] = gl_fp_add(input[9], ARK1[2][9]);
  input[10] = gl_fp_add(input[10], ARK1[2][10]);
  input[11] = gl_fp_add(input[11], ARK1[2][11]);
}
fn add_constants_ark2_2(){
  input[0] = gl_fp_add(input[0], ARK2[2][0]);
  input[1] = gl_fp_add(input[1], ARK2[2][1]);
  input[2] = gl_fp_add(input[2], ARK2[2][2]);
  input[3] = gl_fp_add(input[3], ARK2[2][3]);
  input[4] = gl_fp_add(input[4], ARK2[2][4]);
  input[5] = gl_fp_add(input[5], ARK2[2][5]);
  input[6] = gl_fp_add(input[6], ARK2[2][6]);
  input[7] = gl_fp_add(input[7], ARK2[2][7]);
  input[8] = gl_fp_add(input[8], ARK2[2][8]);
  input[9] = gl_fp_add(input[9], ARK2[2][9]);
  input[10] = gl_fp_add(input[10], ARK2[2][10]);
  input[11] = gl_fp_add(input[11], ARK2[2][11]);
}


fn add_constants_ark1_3(){
  input[0] = gl_fp_add(input[0], ARK1[3][0]);
  input[1] = gl_fp_add(input[1], ARK1[3][1]);
  input[2] = gl_fp_add(input[2], ARK1[3][2]);
  input[3] = gl_fp_add(input[3], ARK1[3][3]);
  input[4] = gl_fp_add(input[4], ARK1[3][4]);
  input[5] = gl_fp_add(input[5], ARK1[3][5]);
  input[6] = gl_fp_add(input[6], ARK1[3][6]);
  input[7] = gl_fp_add(input[7], ARK1[3][7]);
  input[8] = gl_fp_add(input[8], ARK1[3][8]);
  input[9] = gl_fp_add(input[9], ARK1[3][9]);
  input[10] = gl_fp_add(input[10], ARK1[3][10]);
  input[11] = gl_fp_add(input[11], ARK1[3][11]);
}
fn add_constants_ark2_3(){
  input[0] = gl_fp_add(input[0], ARK2[3][0]);
  input[1] = gl_fp_add(input[1], ARK2[3][1]);
  input[2] = gl_fp_add(input[2], ARK2[3][2]);
  input[3] = gl_fp_add(input[3], ARK2[3][3]);
  input[4] = gl_fp_add(input[4], ARK2[3][4]);
  input[5] = gl_fp_add(input[5], ARK2[3][5]);
  input[6] = gl_fp_add(input[6], ARK2[3][6]);
  input[7] = gl_fp_add(input[7], ARK2[3][7]);
  input[8] = gl_fp_add(input[8], ARK2[3][8]);
  input[9] = gl_fp_add(input[9], ARK2[3][9]);
  input[10] = gl_fp_add(input[10], ARK2[3][10]);
  input[11] = gl_fp_add(input[11], ARK2[3][11]);
}


fn add_constants_ark1_4(){
  input[0] = gl_fp_add(input[0], ARK1[4][0]);
  input[1] = gl_fp_add(input[1], ARK1[4][1]);
  input[2] = gl_fp_add(input[2], ARK1[4][2]);
  input[3] = gl_fp_add(input[3], ARK1[4][3]);
  input[4] = gl_fp_add(input[4], ARK1[4][4]);
  input[5] = gl_fp_add(input[5], ARK1[4][5]);
  input[6] = gl_fp_add(input[6], ARK1[4][6]);
  input[7] = gl_fp_add(input[7], ARK1[4][7]);
  input[8] = gl_fp_add(input[8], ARK1[4][8]);
  input[9] = gl_fp_add(input[9], ARK1[4][9]);
  input[10] = gl_fp_add(input[10], ARK1[4][10]);
  input[11] = gl_fp_add(input[11], ARK1[4][11]);
}
fn add_constants_ark2_4(){
  input[0] = gl_fp_add(input[0], ARK2[4][0]);
  input[1] = gl_fp_add(input[1], ARK2[4][1]);
  input[2] = gl_fp_add(input[2], ARK2[4][2]);
  input[3] = gl_fp_add(input[3], ARK2[4][3]);
  input[4] = gl_fp_add(input[4], ARK2[4][4]);
  input[5] = gl_fp_add(input[5], ARK2[4][5]);
  input[6] = gl_fp_add(input[6], ARK2[4][6]);
  input[7] = gl_fp_add(input[7], ARK2[4][7]);
  input[8] = gl_fp_add(input[8], ARK2[4][8]);
  input[9] = gl_fp_add(input[9], ARK2[4][9]);
  input[10] = gl_fp_add(input[10], ARK2[4][10]);
  input[11] = gl_fp_add(input[11], ARK2[4][11]);
}


fn add_constants_ark1_5(){
  input[0] = gl_fp_add(input[0], ARK1[5][0]);
  input[1] = gl_fp_add(input[1], ARK1[5][1]);
  input[2] = gl_fp_add(input[2], ARK1[5][2]);
  input[3] = gl_fp_add(input[3], ARK1[5][3]);
  input[4] = gl_fp_add(input[4], ARK1[5][4]);
  input[5] = gl_fp_add(input[5], ARK1[5][5]);
  input[6] = gl_fp_add(input[6], ARK1[5][6]);
  input[7] = gl_fp_add(input[7], ARK1[5][7]);
  input[8] = gl_fp_add(input[8], ARK1[5][8]);
  input[9] = gl_fp_add(input[9], ARK1[5][9]);
  input[10] = gl_fp_add(input[10], ARK1[5][10]);
  input[11] = gl_fp_add(input[11], ARK1[5][11]);
}
fn add_constants_ark2_5(){
  input[0] = gl_fp_add(input[0], ARK2[5][0]);
  input[1] = gl_fp_add(input[1], ARK2[5][1]);
  input[2] = gl_fp_add(input[2], ARK2[5][2]);
  input[3] = gl_fp_add(input[3], ARK2[5][3]);
  input[4] = gl_fp_add(input[4], ARK2[5][4]);
  input[5] = gl_fp_add(input[5], ARK2[5][5]);
  input[6] = gl_fp_add(input[6], ARK2[5][6]);
  input[7] = gl_fp_add(input[7], ARK2[5][7]);
  input[8] = gl_fp_add(input[8], ARK2[5][8]);
  input[9] = gl_fp_add(input[9], ARK2[5][9]);
  input[10] = gl_fp_add(input[10], ARK2[5][10]);
  input[11] = gl_fp_add(input[11], ARK2[5][11]);
}


fn add_constants_ark1_6(){
  input[0] = gl_fp_add(input[0], ARK1[6][0]);
  input[1] = gl_fp_add(input[1], ARK1[6][1]);
  input[2] = gl_fp_add(input[2], ARK1[6][2]);
  input[3] = gl_fp_add(input[3], ARK1[6][3]);
  input[4] = gl_fp_add(input[4], ARK1[6][4]);
  input[5] = gl_fp_add(input[5], ARK1[6][5]);
  input[6] = gl_fp_add(input[6], ARK1[6][6]);
  input[7] = gl_fp_add(input[7], ARK1[6][7]);
  input[8] = gl_fp_add(input[8], ARK1[6][8]);
  input[9] = gl_fp_add(input[9], ARK1[6][9]);
  input[10] = gl_fp_add(input[10], ARK1[6][10]);
  input[11] = gl_fp_add(input[11], ARK1[6][11]);
}
fn add_constants_ark2_6(){
  input[0] = gl_fp_add(input[0], ARK2[6][0]);
  input[1] = gl_fp_add(input[1], ARK2[6][1]);
  input[2] = gl_fp_add(input[2], ARK2[6][2]);
  input[3] = gl_fp_add(input[3], ARK2[6][3]);
  input[4] = gl_fp_add(input[4], ARK2[6][4]);
  input[5] = gl_fp_add(input[5], ARK2[6][5]);
  input[6] = gl_fp_add(input[6], ARK2[6][6]);
  input[7] = gl_fp_add(input[7], ARK2[6][7]);
  input[8] = gl_fp_add(input[8], ARK2[6][8]);
  input[9] = gl_fp_add(input[9], ARK2[6][9]);
  input[10] = gl_fp_add(input[10], ARK2[6][10]);
  input[11] = gl_fp_add(input[11], ARK2[6][11]);
}


fn apply_ext_round_0(){

  add_constants_ark1_0();
  apply_ext();
}
fn apply_fb_round_0(){
  apply_mds();  
  add_constants_ark1_0();

  apply_sbox();

  apply_mds();
  add_constants_ark2_0();
  
  apply_inv_sbox();
}


fn apply_ext_round_1(){

  add_constants_ark1_1();
  apply_ext();
}
fn apply_fb_round_1(){
  apply_mds();  
  add_constants_ark1_1();

  apply_sbox();

  apply_mds();
  add_constants_ark2_1();
  
  apply_inv_sbox();
}


fn apply_ext_round_2(){

  add_constants_ark1_2();
  apply_ext();
}
fn apply_fb_round_2(){
  apply_mds();  
  add_constants_ark1_2();

  apply_sbox();

  apply_mds();
  add_constants_ark2_2();
  
  apply_inv_sbox();
}


fn apply_ext_round_3(){

  add_constants_ark1_3();
  apply_ext();
}
fn apply_fb_round_3(){
  apply_mds();  
  add_constants_ark1_3();

  apply_sbox();

  apply_mds();
  add_constants_ark2_3();
  
  apply_inv_sbox();
}


fn apply_ext_round_4(){

  add_constants_ark1_4();
  apply_ext();
}
fn apply_fb_round_4(){
  apply_mds();  
  add_constants_ark1_4();

  apply_sbox();

  apply_mds();
  add_constants_ark2_4();
  
  apply_inv_sbox();
}


fn apply_ext_round_5(){

  add_constants_ark1_5();
  apply_ext();
}
fn apply_fb_round_5(){
  apply_mds();  
  add_constants_ark1_5();

  apply_sbox();

  apply_mds();
  add_constants_ark2_5();
  
  apply_inv_sbox();
}


fn apply_ext_round_6(){

  add_constants_ark1_6();
  apply_ext();
}
fn apply_fb_round_6(){
  apply_mds();  
  add_constants_ark1_6();

  apply_sbox();

  apply_mds();
  add_constants_ark2_6();
  
  apply_inv_sbox();
}


fn apply_final_round() {
  apply_mds();
  add_constants_ark1_6();
}
fn rpx_permute() {
  apply_fb_round_0();
  apply_ext_round_1();
  apply_fb_round_2();
  apply_ext_round_3();
  apply_fb_round_4();
  apply_ext_round_5();
  apply_final_round();
}
fn rpo_permute(){
  apply_fb_round_0();
  apply_fb_round_1();
  apply_fb_round_2();
  apply_fb_round_3();
  apply_fb_round_4();
  apply_fb_round_5();
  apply_fb_round_6();
}


@group(0) @binding(0) var<storage, read_write>digests: array<RPXHashOut>;
@group(0) @binding(1) var<uniform> node_count: u32;

@group(0) @binding(2) var<storage, read> row_or_leaf_inputs: array<RPXHashOut>;
@group(0) @binding(3) var<storage, read_write>row_hash_state: array<RPXHashOut>;


@compute @workgroup_size(1) fn rpo_absorb_rows(
  @builtin(global_invocation_id) id: vec3 <u32 >
) {
  
  let i = id.x+id.y*32768u; // parent node index

  let row_input_index = i*2u;

  input[0] = row_hash_state[i][0];
  input[1] = row_hash_state[i][1];
  input[2] = row_hash_state[i][2];
  input[3] = row_hash_state[i][3];
  input[4] = row_or_leaf_inputs[row_input_index][0];
  input[5] = row_or_leaf_inputs[row_input_index][1];
  input[6] = row_or_leaf_inputs[row_input_index][2];
  input[7] = row_or_leaf_inputs[row_input_index][3];
  input[8] = row_or_leaf_inputs[row_input_index+1][0];
  input[9] = row_or_leaf_inputs[row_input_index+1][1];
  input[10] = row_or_leaf_inputs[row_input_index+1][2];
  input[11] = row_or_leaf_inputs[row_input_index+1][3];
  rpo_permute();
  row_hash_state[i][0] = input[0];
  row_hash_state[i][1] = input[1];
  row_hash_state[i][2] = input[2];
  row_hash_state[i][3] = input[3];


  digests[i][0] = input[4];
  digests[i][1] = input[5];
  digests[i][2] = input[6];
  digests[i][3] = input[7];
}



@compute @workgroup_size(1) fn rpo_absorb_rows_pad(
  @builtin(global_invocation_id) id: vec3 <u32 >
) {
  
  let i = id.x+id.y*32768u; // parent node index

  let row_input_index = i*2u;

  input[0] = ONE;
  input[1] = row_hash_state[i][1];
  input[2] = row_hash_state[i][2];
  input[3] = row_hash_state[i][3];
  input[4] = row_or_leaf_inputs[row_input_index][0];
  input[5] = row_or_leaf_inputs[row_input_index][1];
  input[6] = row_or_leaf_inputs[row_input_index][2];
  input[7] = row_or_leaf_inputs[row_input_index][3];
  input[8] = row_or_leaf_inputs[row_input_index+1][0];
  input[9] = row_or_leaf_inputs[row_input_index+1][1];
  input[10] = row_or_leaf_inputs[row_input_index+1][2];
  input[11] = row_or_leaf_inputs[row_input_index+1][3];
  rpo_permute();
  row_hash_state[i][0] = input[0];
  row_hash_state[i][1] = input[1];
  row_hash_state[i][2] = input[2];
  row_hash_state[i][3] = input[3];


  digests[i][0] = input[4];
  digests[i][1] = input[5];
  digests[i][2] = input[6];
  digests[i][3] = input[7];
}



@compute @workgroup_size(1) fn rpo_hash_leaves(
  @builtin(global_invocation_id) id: vec3 <u32 >
) {
  
  let i = id.x+id.y*32768u; // parent node index

  let leaf_offset = i*2;


  input[0] = ZERO;
  input[1] = ZERO;
  input[2] = ZERO;
  input[3] = ZERO;
  input[4] = row_or_leaf_inputs[leaf_offset][0];
  input[5] = row_or_leaf_inputs[leaf_offset][1];
  input[6] = row_or_leaf_inputs[leaf_offset][2];
  input[7] = row_or_leaf_inputs[leaf_offset][3];


  input[8] = row_or_leaf_inputs[leaf_offset+1][0];
  input[9] = row_or_leaf_inputs[leaf_offset+1][1];
  input[10] = row_or_leaf_inputs[leaf_offset+1][2];
  input[11] = row_or_leaf_inputs[leaf_offset+1][3];
  rpo_permute();
  let node_offset = i+node_count;
  digests[node_offset][0] = input[4];
  digests[node_offset][1] = input[5];
  digests[node_offset][2] = input[6];
  digests[node_offset][3] = input[7];
}



@compute @workgroup_size(1) fn rpo_hash_level(
  @builtin(global_invocation_id) id: vec3 <u32 >
) {
  
  let i = id.x+id.y*32768u;

  let child_nodes_offset = node_count*2+i*2;


  input[0] = ZERO;
  input[1] = ZERO;
  input[2] = ZERO;
  input[3] = ZERO;
  input[4] = digests[child_nodes_offset][0];
  input[5] = digests[child_nodes_offset][1];
  input[6] = digests[child_nodes_offset][2];
  input[7] = digests[child_nodes_offset][3];


  input[8] = digests[child_nodes_offset+1][0];
  input[9] = digests[child_nodes_offset+1][1];
  input[10] = digests[child_nodes_offset+1][2];
  input[11] = digests[child_nodes_offset+1][3];
  rpo_permute();
  let node_offset = i+node_count;
  digests[node_offset][0] = input[4];
  digests[node_offset][1] = input[5];
  digests[node_offset][2] = input[6];
  digests[node_offset][3] = input[7];
}




@compute @workgroup_size(1) fn rpx_absorb_rows(
  @builtin(global_invocation_id) id: vec3 <u32 >
) {
  
  let i = id.x+id.y*32768u; // parent node index

  let row_input_index = i*2u;

  input[0] = row_hash_state[i][0];
  input[1] = row_hash_state[i][1];
  input[2] = row_hash_state[i][2];
  input[3] = row_hash_state[i][3];
  input[4] = row_or_leaf_inputs[row_input_index][0];
  input[5] = row_or_leaf_inputs[row_input_index][1];
  input[6] = row_or_leaf_inputs[row_input_index][2];
  input[7] = row_or_leaf_inputs[row_input_index][3];
  input[8] = row_or_leaf_inputs[row_input_index+1][0];
  input[9] = row_or_leaf_inputs[row_input_index+1][1];
  input[10] = row_or_leaf_inputs[row_input_index+1][2];
  input[11] = row_or_leaf_inputs[row_input_index+1][3];
  rpo_permute();
  row_hash_state[i][0] = input[0];
  row_hash_state[i][1] = input[1];
  row_hash_state[i][2] = input[2];
  row_hash_state[i][3] = input[3];


  digests[i][0] = input[4];
  digests[i][1] = input[5];
  digests[i][2] = input[6];
  digests[i][3] = input[7];
}



@compute @workgroup_size(1) fn rpx_absorb_rows_pad(
  @builtin(global_invocation_id) id: vec3 <u32 >
) {
  
  let i = id.x+id.y*32768u; // parent node index

  let row_input_index = i*2u;

  input[0] = ONE;
  input[1] = row_hash_state[i][1];
  input[2] = row_hash_state[i][2];
  input[3] = row_hash_state[i][3];
  input[4] = row_or_leaf_inputs[row_input_index][0];
  input[5] = row_or_leaf_inputs[row_input_index][1];
  input[6] = row_or_leaf_inputs[row_input_index][2];
  input[7] = row_or_leaf_inputs[row_input_index][3];
  input[8] = row_or_leaf_inputs[row_input_index+1][0];
  input[9] = row_or_leaf_inputs[row_input_index+1][1];
  input[10] = row_or_leaf_inputs[row_input_index+1][2];
  input[11] = row_or_leaf_inputs[row_input_index+1][3];
  rpx_permute();
  row_hash_state[i][0] = input[0];
  row_hash_state[i][1] = input[1];
  row_hash_state[i][2] = input[2];
  row_hash_state[i][3] = input[3];


  digests[i][0] = input[4];
  digests[i][1] = input[5];
  digests[i][2] = input[6];
  digests[i][3] = input[7];
}



@compute @workgroup_size(1) fn rpx_hash_leaves(
  @builtin(global_invocation_id) id: vec3 <u32 >
) {
  
  let i = id.x+id.y*32768u; // parent node index

  let leaf_offset = i*2;


  input[0] = ZERO;
  input[1] = ZERO;
  input[2] = ZERO;
  input[3] = ZERO;
  input[4] = row_or_leaf_inputs[leaf_offset][0];
  input[5] = row_or_leaf_inputs[leaf_offset][1];
  input[6] = row_or_leaf_inputs[leaf_offset][2];
  input[7] = row_or_leaf_inputs[leaf_offset][3];


  input[8] = row_or_leaf_inputs[leaf_offset+1][0];
  input[9] = row_or_leaf_inputs[leaf_offset+1][1];
  input[10] = row_or_leaf_inputs[leaf_offset+1][2];
  input[11] = row_or_leaf_inputs[leaf_offset+1][3];
  rpx_permute();
  let node_offset = i+node_count;
  digests[node_offset][0] = input[4];
  digests[node_offset][1] = input[5];
  digests[node_offset][2] = input[6];
  digests[node_offset][3] = input[7];
}



@compute @workgroup_size(1) fn rpx_hash_level(
  @builtin(global_invocation_id) id: vec3 <u32 >
) {
  
  let i = id.x+id.y*32768u;

  let child_nodes_offset = node_count*2+i*2;


  input[0] = ZERO;
  input[1] = ZERO;
  input[2] = ZERO;
  input[3] = ZERO;
  input[4] = digests[child_nodes_offset][0];
  input[5] = digests[child_nodes_offset][1];
  input[6] = digests[child_nodes_offset][2];
  input[7] = digests[child_nodes_offset][3];


  input[8] = digests[child_nodes_offset+1][0];
  input[9] = digests[child_nodes_offset+1][1];
  input[10] = digests[child_nodes_offset+1][2];
  input[11] = digests[child_nodes_offset+1][3];
  rpx_permute();
  let node_offset = i+node_count;
  digests[node_offset][0] = input[4];
  digests[node_offset][1] = input[5];
  digests[node_offset][2] = input[6];
  digests[node_offset][3] = input[7];
}
