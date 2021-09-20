use crate::{
    BASE_CYCLE_LENGTH as CYCLE_LENGTH, HASH_DIGEST_SIZE as DIGEST_SIZE,
    HASH_NUM_ROUNDS as NUM_ROUNDS, HASH_STATE_RATE as STATE_RATE, HASH_STATE_WIDTH as STATE_WIDTH,
};
use winterfell::math::{fields::f128::BaseElement, FieldElement};

// HASHER FUNCTIONS
// ================================================================================================
pub fn digest(values: &[BaseElement]) -> Vec<BaseElement> {
    assert!(
        values.len() <= STATE_RATE,
        "expected no more than {}, but received {}",
        STATE_RATE,
        values.len()
    );

    let mut state = [BaseElement::ZERO; STATE_WIDTH];
    state[..values.len()].copy_from_slice(values);
    state.reverse();

    for i in 0..NUM_ROUNDS {
        apply_round(&mut state, i);
    }

    state.reverse();
    state[..DIGEST_SIZE].to_vec()
}

pub fn apply_round(state: &mut [BaseElement], step: usize) {
    let ark_idx = step % CYCLE_LENGTH;

    // apply Rescue round
    add_constants(state, ark_idx, 0);
    apply_sbox(state);
    apply_mds(state);

    add_constants(state, ark_idx, STATE_WIDTH);
    apply_inv_sbox(state);
    apply_mds(state);
}

pub fn add_constants(state: &mut [BaseElement], idx: usize, offset: usize) {
    for i in 0..STATE_WIDTH {
        state[i] = state[i] + ARK[offset + i][idx];
    }
}

pub fn apply_sbox<E: FieldElement>(state: &mut [E]) {
    for i in 0..STATE_WIDTH {
        state[i] = state[i].exp(ALPHA.into());
    }
}

pub fn apply_inv_sbox(state: &mut [BaseElement]) {
    // TODO: optimize
    for i in 0..STATE_WIDTH {
        state[i] = state[i].exp(INV_ALPHA);
    }
}

pub fn apply_mds<E: FieldElement<BaseField = BaseElement>>(state: &mut [E]) {
    let mut result = [E::ZERO; STATE_WIDTH];
    let mut temp = [E::ZERO; STATE_WIDTH];
    for i in 0..STATE_WIDTH {
        for j in 0..STATE_WIDTH {
            temp[j] = E::from(MDS[i * STATE_WIDTH + j]) * state[j];
        }

        for j in 0..STATE_WIDTH {
            result[i] += temp[j];
        }
    }
    state.copy_from_slice(&result);
}

pub fn apply_inv_mds<E: FieldElement<BaseField = BaseElement>>(state: &mut [E]) {
    let mut result = [E::ZERO; STATE_WIDTH];
    let mut temp = [E::ZERO; STATE_WIDTH];
    for i in 0..STATE_WIDTH {
        for j in 0..STATE_WIDTH {
            temp[j] = E::from(INV_MDS[i * STATE_WIDTH + j]) * state[j];
        }

        for j in 0..STATE_WIDTH {
            result[i] = result[i] * temp[j];
        }
    }
    state.copy_from_slice(&result);
}

// 128-BIT RESCUE CONSTANTS
// ================================================================================================

const ALPHA: u32 = 3;
const INV_ALPHA: u128 = 226854911280625642308916371969163307691;

const MDS: [BaseElement; STATE_WIDTH * STATE_WIDTH] = [
    BaseElement::new(34702391375697798808541201166389247321),
    BaseElement::new(292720401120629668097050277338444166479),
    BaseElement::new(252221686506898646925660607780980529565),
    BaseElement::new(1545301432720594930091500405440765270),
    BaseElement::new(249229091188143033873076468277345141138),
    BaseElement::new(220001593723324427188563221285612032538),
    BaseElement::new(223274184432289781839114239013770504955),
    BaseElement::new(330042960751289206923775620692185805456),
    BaseElement::new(68147806084648525660922442535124284349),
    BaseElement::new(170632587193822854126540173326689266153),
    BaseElement::new(250033902372207462477717017592730125263),
    BaseElement::new(241770281110130121239200125437407593586),
    BaseElement::new(59697037488579951129595490016876870776),
    BaseElement::new(173037025415440639734730939871096987969),
    BaseElement::new(244520331803890388707106378055030145592),
    BaseElement::new(34432552219210978837375640622811234255),
    BaseElement::new(224883744083395074894597669527169800639),
    BaseElement::new(118987613174044827738657284387435362389),
    BaseElement::new(178405816334148045547444196947551204988),
    BaseElement::new(329492269239016599078865693624718656026),
    BaseElement::new(5932101030068686798137276449370802175),
    BaseElement::new(82061764821777249869371835777715849101),
    BaseElement::new(306668779179264388848571277248016578389),
    BaseElement::new(111826390250505084847238806440816480816),
    BaseElement::new(311483585658925440282415980040674425324),
    BaseElement::new(145759142525852648614462741190071048842),
    BaseElement::new(298009719049064449565063658657087558281),
    BaseElement::new(20897600766797241015108657845791963643),
    BaseElement::new(10708575082009935910638872278310543860),
    BaseElement::new(1925891850054765549789187338759995098),
    BaseElement::new(215280917571128372543200644166029121446),
    BaseElement::new(95409967251054914374823434415713274752),
    BaseElement::new(47264968375673684314231886553980659586),
    BaseElement::new(324414896710549426218067045352609729751),
    BaseElement::new(134533192639212680415562336758249126966),
    BaseElement::new(113819576569856286031671903516923963713),
];

const INV_MDS: [BaseElement; STATE_WIDTH * STATE_WIDTH] = [
    BaseElement::new(262838870629088612431704202279372804365),
    BaseElement::new(222534852307924763539037405488664736598),
    BaseElement::new(81646375972783381860153911140700582070),
    BaseElement::new(199701217011341155249764385939633347172),
    BaseElement::new(332690827334260982039678112409792234722),
    BaseElement::new(89516833564756742553232000561318423105),
    BaseElement::new(268971102785434627130755364883988908171),
    BaseElement::new(281455956826013141038766773877631706708),
    BaseElement::new(260031388727053904917456703760391819397),
    BaseElement::new(129628250235822118843226947626941528745),
    BaseElement::new(4752366532746298637564749858084292362),
    BaseElement::new(41511161206792419767881138471159674575),
    BaseElement::new(182137736677911549469583431255398613256),
    BaseElement::new(81590632530312527113593374981163544706),
    BaseElement::new(187388592404931394753634116375361630728),
    BaseElement::new(33930556286904149060086783748137598339),
    BaseElement::new(128856701558652053153053295206366630120),
    BaseElement::new(142358475095617426664163236851553505832),
    BaseElement::new(227338232821163582236298610629943480668),
    BaseElement::new(331216968806018107216634963151792902695),
    BaseElement::new(26461805327234621438733072541631103386),
    BaseElement::new(256026201749452500286052447974298083810),
    BaseElement::new(225021687632788174205391941469031171903),
    BaseElement::new(142987865655544874008196227497386450956),
    BaseElement::new(23528566895021558524617283665890082663),
    BaseElement::new(236073665414937855350933025880234841420),
    BaseElement::new(333046083413429707379387119916134573299),
    BaseElement::new(306789363216075136579531522242680359507),
    BaseElement::new(28410362227208796996681096912722982356),
    BaseElement::new(315464216130582396582421428659045349544),
    BaseElement::new(83197014978047724714246600214591095407),
    BaseElement::new(26699280864421082988027855316107178960),
    BaseElement::new(198532124113408992589453646783315962129),
    BaseElement::new(33447687267235554595784456310784250249),
    BaseElement::new(230423441211289196836867098686067243907),
    BaseElement::new(6651208139349692977552460975523401420),
];

pub const ARK: [[BaseElement; CYCLE_LENGTH]; STATE_WIDTH * 2] = [
    [
        BaseElement::new(73742662193393629993182617210984534396),
        BaseElement::new(53265540956785335308970867946461681393),
        BaseElement::new(14395595548581550072136442264588359269),
        BaseElement::new(122001776241989922016768881111033630021),
        BaseElement::new(60517382118002481956993039132628798754),
        BaseElement::new(242872884766759335785324964049644229294),
        BaseElement::new(4363347423120340347647334422662129280),
        BaseElement::new(36224510031696203479366212612960872957),
        BaseElement::new(48405253030503584410290697712994785780),
        BaseElement::new(81691558114273932307586556761543100315),
        BaseElement::new(315851285839738308287329276161693313425),
        BaseElement::new(326468515245013538774703881972225680443),
        BaseElement::new(43697512293048123577843997788308773455),
        BaseElement::new(311182552853825261047305944842224924215),
        BaseElement::new(23833044413239455428827669432473543240),
        BaseElement::new(7640791703119561504971867271087353186),
    ],
    [
        BaseElement::new(294241649061853322876594266104693176711),
        BaseElement::new(37163237225742447359704121711857363416),
        BaseElement::new(122453723578185362799857252115182955415),
        BaseElement::new(45955200056324872841369110391855073949),
        BaseElement::new(118224404177203231307646344308524770691),
        BaseElement::new(334905318181122708043147970432770442813),
        BaseElement::new(151456178618798089303785904835852898400),
        BaseElement::new(158324780313294970656577210958221752332),
        BaseElement::new(94987431711345870355583825474329298047),
        BaseElement::new(314293870425266938862923101612602484635),
        BaseElement::new(153975056764703018977481562856167540343),
        BaseElement::new(321383880935903155966493388921501530915),
        BaseElement::new(50057060110310193394516504439805601601),
        BaseElement::new(101740347373933108709122003348416870840),
        BaseElement::new(80845608757236703492016225128275757615),
        BaseElement::new(209519938465996994070512842713405349097),
    ],
    [
        BaseElement::new(158538539401072639862099558319550076686),
        BaseElement::new(221096166077280180974764042888991644280),
        BaseElement::new(58496669788416466040038464653643977917),
        BaseElement::new(59235259390239124162891762278360245334),
        BaseElement::new(337725857612570850944445340416668827103),
        BaseElement::new(232074846252364869196809445831737773796),
        BaseElement::new(50018412546799023168899671792323407156),
        BaseElement::new(166545598284411242433605578379265360252),
        BaseElement::new(41491163124497803255407972080635378902),
        BaseElement::new(302719082300742526890675313445319567341),
        BaseElement::new(193135973972933518870828237886863798021),
        BaseElement::new(230635877078223923040415038811686445073),
        BaseElement::new(138405289600908304802269329797084135857),
        BaseElement::new(185089342855265166563915858025522983409),
        BaseElement::new(43421407022492486112194101527865465264),
        BaseElement::new(62365388436267647533064120634464266870),
    ],
    [
        BaseElement::new(116358578177298194933445426886059838431),
        BaseElement::new(161426242690584918941198733450953748769),
        BaseElement::new(228752352631998151610775212885524543283),
        BaseElement::new(182846621472767704751329603405195985261),
        BaseElement::new(61911644581679112386499312030413349074),
        BaseElement::new(191090374127295994022314014407997806335),
        BaseElement::new(59079983632109588980783021622461415033),
        BaseElement::new(193859304217638223479173371326185841274),
        BaseElement::new(280938106646730498467301259432184740730),
        BaseElement::new(679464766810703810097965767355062873),
        BaseElement::new(150345637803188209699415557320545415720),
        BaseElement::new(139823638104054506965247243102295231737),
        BaseElement::new(53655583013674525883209345165753178194),
        BaseElement::new(126806292806004264446745284405742612689),
        BaseElement::new(9602891270757320013616862490986026227),
        BaseElement::new(160806286415414414379046661006476545066),
    ],
    [
        BaseElement::new(82429262549299942290847183493004485261),
        BaseElement::new(135862987622353414661673448620033990934),
        BaseElement::new(189653807408664613044858917026657980625),
        BaseElement::new(89333775516890774827962437297764936547),
        BaseElement::new(151495710594170316099539790651453416361),
        BaseElement::new(287288998844960276649854461883880913666),
        BaseElement::new(78065099645540746831460653583134588104),
        BaseElement::new(55063854082489962294956447144901184837),
        BaseElement::new(331958862978706756999748973740992156929),
        BaseElement::new(8168599451814692118441936734435571667),
        BaseElement::new(166002344081927954304873771936867289851),
        BaseElement::new(225556280578098393163620719229418290860),
        BaseElement::new(234470815157983004947611441850027217492),
        BaseElement::new(188323096432976273265052369652285099186),
        BaseElement::new(77086595049850596660690999278719011720),
        BaseElement::new(219177966622498447376602481443936826442),
    ],
    [
        BaseElement::new(153964862116746563988492365899737226989),
        BaseElement::new(171502007855719014010389694111716628578),
        BaseElement::new(69476260396969790693146402021744933499),
        BaseElement::new(154737674033120948700227987365296907637),
        BaseElement::new(321756280164272289841871040803703440350),
        BaseElement::new(131528659800906379821588177210148124019),
        BaseElement::new(80761083418432627134481526210881542477),
        BaseElement::new(250524460337537482950569449224813700391),
        BaseElement::new(230491494516843960542668710050516211970),
        BaseElement::new(45314269339319734203127091458645722622),
        BaseElement::new(2780762044206215421580528389917005833),
        BaseElement::new(165769058045453677221497711462583950139),
        BaseElement::new(259395388889719671782653113655841647262),
        BaseElement::new(219135320838134930923443959673366229256),
        BaseElement::new(286172465494565666647121151879345971089),
        BaseElement::new(147904845125332345734618546117273133070),
    ],
    [
        BaseElement::new(310538827479436149892724250590698914519),
        BaseElement::new(158907965876520949616863328303176330572),
        BaseElement::new(230609671293877243511889006223284127479),
        BaseElement::new(32424193637360906576442956294452323288),
        BaseElement::new(250107916917535224528378129994943394294),
        BaseElement::new(138628264101912804813977210615833233437),
        BaseElement::new(265168486075436613449458788630803272512),
        BaseElement::new(69216162599706897556278776240900218374),
        BaseElement::new(189445283838085809052254029811407633258),
        BaseElement::new(233141108584353453034002234415979233911),
        BaseElement::new(214406010671246827947835794343033790693),
        BaseElement::new(11153792801390339798262783617007369172),
        BaseElement::new(118114082982223329826045602989947510129),
        BaseElement::new(263157893448998999306850171729945394432),
        BaseElement::new(284751400376525550861233017183497639371),
        BaseElement::new(267697496976874759865163284761384997437),
    ],
    [
        BaseElement::new(93028746118909893246237533845189074002),
        BaseElement::new(185875552438512768131393810027777987752),
        BaseElement::new(185941035889835671403097747661105595079),
        BaseElement::new(253746202714926890236889780444552427226),
        BaseElement::new(101396399019525872501663112616210307683),
        BaseElement::new(215901816653704881294214215068798346060),
        BaseElement::new(201416315867789883891889554246377963162),
        BaseElement::new(251801358233276697762579413801911171612),
        BaseElement::new(192826288785653777157020265215517987662),
        BaseElement::new(15885268928012076989988458786521561031),
        BaseElement::new(3463161311202689884181131747640413605),
        BaseElement::new(79003969367131068546865741459306118251),
        BaseElement::new(69521903572951337445452502748565566496),
        BaseElement::new(301962999029915705994021697766389081208),
        BaseElement::new(9094956230559373758985913855137693312),
        BaseElement::new(144516981820451119929097195276243798745),
    ],
    [
        BaseElement::new(190338348930091047298074165559397264378),
        BaseElement::new(274633988293091071340356635555807179190),
        BaseElement::new(178527953570703982986577498890483203023),
        BaseElement::new(62033181748425106711292370817969454146),
        BaseElement::new(300207915911051460908298688414919921093),
        BaseElement::new(279904213525927510712810228623902377237),
        BaseElement::new(169061616865327291005064664270275836534),
        BaseElement::new(109862755442048596627938134642975399668),
        BaseElement::new(26236509279945457822369793146288866403),
        BaseElement::new(12234569840122312404615178877814773825),
        BaseElement::new(170640173476284978302806154399958141555),
        BaseElement::new(209040028248304238735923683513240525194),
        BaseElement::new(334227022756371766478760448625337379424),
        BaseElement::new(25509259982013669682461356932775370545),
        BaseElement::new(258001239441974384951891541079242930440),
        BaseElement::new(170024582242541755392979256646565617273),
    ],
    [
        BaseElement::new(179443458614881887600494128053111694648),
        BaseElement::new(108165142884901978856319583750672324489),
        BaseElement::new(97063282200318501142854934314343169049),
        BaseElement::new(261286087759526359216271155361018330507),
        BaseElement::new(67833038363599207475373040930824843019),
        BaseElement::new(56878992720628535103195481580617360771),
        BaseElement::new(198852036109370286966576164360266278255),
        BaseElement::new(174521831193496100673067735908873646985),
        BaseElement::new(251654188127562510403516067236333482372),
        BaseElement::new(48056343894932757577046683797067209079),
        BaseElement::new(306942787541210815164178987028698818659),
        BaseElement::new(156642260202818413362503062578539720517),
        BaseElement::new(251616653853928459967283575542057535293),
        BaseElement::new(188741644029927191719040650968720800409),
        BaseElement::new(281428110117091114144446350524650424481),
        BaseElement::new(64627937813848943279040280988334503406),
    ],
    [
        BaseElement::new(289278996656706117461857789813498821934),
        BaseElement::new(274604860873273636237081114376077113475),
        BaseElement::new(126000924558481152083098962591383883438),
        BaseElement::new(129877116445533126989528570413807277693),
        BaseElement::new(172066229584406173063202914726937339958),
        BaseElement::new(298530663250990395227144225232608384365),
        BaseElement::new(16989575615175240495557720305287640349),
        BaseElement::new(102835474498154050313290986853294842906),
        BaseElement::new(297928660776980173370496618733852490961),
        BaseElement::new(96037481352786813748421760769380383926),
        BaseElement::new(2818165229115014774032882127170013258),
        BaseElement::new(293027053537479076557105009345927645442),
        BaseElement::new(249369722351358137898587699909312963803),
        BaseElement::new(300544292992993952360719000252205715076),
        BaseElement::new(323117003802246814764810890058143344905),
        BaseElement::new(243579355010018669877160932197352017974),
    ],
    [
        BaseElement::new(339223760157195739332845857285008200423),
        BaseElement::new(208632865147351209340449219082125897333),
        BaseElement::new(96675618862527967726114378655626650641),
        BaseElement::new(162892536327655189685235410342890574896),
        BaseElement::new(196910153233132861881308509456401645140),
        BaseElement::new(281841826874183647567546019531929972702),
        BaseElement::new(155276073049009029667373106803046514344),
        BaseElement::new(152642017050116048509158960350000858013),
        BaseElement::new(286456894851095755022390967246767421000),
        BaseElement::new(215531716255970146473658338852472046173),
        BaseElement::new(324452408864695917006896030536225525119),
        BaseElement::new(314094406162389098987684450322979120529),
        BaseElement::new(114910730596486251472791631840513265074),
        BaseElement::new(81795345404219176616297063519210464031),
        BaseElement::new(22603524397731600512825466576357638930),
        BaseElement::new(63900149356112496372337283043133097338),
    ],
];
