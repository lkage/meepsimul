#!/usr/bin/env python3
"""단일 나노로드를 이용한 THz "pinch harmonics" 나노공진기의 3D Meep FDTD 재현.

원 논문:
  H.-R. Park et al., "Terahertz pinch harmonics enabled by single nano
  rods," Opt. Express 19(24), 24775-24781 (2011).

[이 스크립트가 재현하는 것]
논문 Sec. 2의 FDTD 파트("To simulate pinch harmonics by Pt nano rods,
we perform three-dimensional finite-difference time-domain (FDTD)
analysis..."). 즉:
  - 20 nm 두께 금박에 뚫린 길이 l1=10 um, 폭 w1=20 nm짜리 직사각형 슬롯
  - 슬롯의 짧은 축(폭) 방향으로 편광된 p-편광 THz 펄스가 수직 입사
  - 슬롯 위, 특정 위치("1:2" 지점 또는 중앙 "1:1" 지점)에 놓인 Pt 나노로드
    (폭 s=60 nm "large" 또는 s=20 nm "small")

[두 가지 모드]
  spectrum  -- 광대역(broadband) 플럭스(flux) 계산 실행.
               투과율(transmittance) T(f) = P_trans(f) / P_inc(f) 를 구함
               (논문 Fig. 2(b) / Fig. 3(b) 재현용)
  field     -- 특정 몇 개 주파수에서만 DFT(이산 푸리에 변환) 필드를 뽑아
               |Ex|^2 공간 분포(맵)를 얻음. 공진기 출구 바로 뒤 평면에서
               측정 (논문 Fig. 2(c) / Fig. 3(c) 재현용)

실행 전에 README.md를 꼭 읽을 것: 기본 해상도(resolution)는 스크립트가
"제대로 도는지"만 빠르게 확인하는 용도의 저해상도 값이며, 논문 수준의
정밀한 재현을 위한 값이 아니다. 20 nm급 구조를 충분히 잘 표현하려면
해상도를 수백(1/um) 수준까지 올려야 하고, 그러면 3D 계산의 메모리/시간
비용이 매우 커지므로 MPI 병렬(mpirun)로 클러스터에서 돌리는 것을 전제로
한다.

사용 예 (Fig. 2(b), 즉 "1:2" 위치를 재현하는 경우):
    mpirun -np 4 python3 simulate.py --mode spectrum --empty \
        --outdir results/pos12
    mpirun -np 4 python3 simulate.py --mode spectrum --rod none \
        --outdir results/pos12
    mpirun -np 4 python3 simulate.py --mode spectrum --rod large \
        --position 1:2 --outdir results/pos12
    mpirun -np 4 python3 simulate.py --mode spectrum --rod small \
        --position 1:2 --outdir results/pos12
    python3 analyze.py spectrum --indir results/pos12 --outfile fig2b.png

Fig. 2와 Fig. 3 전체를 재현하는 데 필요한 실행 목록은 run_all.sh 참고.
모델링 가정과 한계는 README.md에 정리되어 있다.
"""

import argparse
import os

import numpy as np

import meep as mp

from geometry import ResonatorParams, RodParams, build_geometry, rod_y_center

# 논문에서 쓰는 두 가지 로드 폭 (um 단위). CLI의 --rod large/small 선택지에 대응.
ROD_WIDTHS = {"large": 0.060, "small": 0.020}  # 60 nm / 20 nm


def parse_args():
    """명령행 인자를 정의하고 파싱한다.

    각 옵션이 물리적으로 무엇을 뜻하는지는 geometry.py의 ResonatorParams /
    RodParams 도크스트링과 아래 build_sim() 주석을 함께 보면 이해하기 쉽다.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- 실행 모드 및 구조 선택 ---
    p.add_argument("--mode", choices=["spectrum", "field"], required=True)
    p.add_argument("--rod", choices=["none", "large", "small"], default="none")
    p.add_argument("--position", choices=["1:2", "1:1"], default="1:2",
                    help="슬롯을 따라 로드가 놓이는 (d1:d2) 위치. --rod none이면 무시됨")
    p.add_argument("--film-metal", default="Au", choices=["Au", "Pt"])
    p.add_argument("--rod-metal", default="Pt", choices=["Au", "Pt"])
    p.add_argument("--custom-drude-film", type=str, default=None,
                    help="'f_p,gamma' (1/um 단위). 지정하면 meep.materials.Au/Pt 대신 "
                         "이 값으로 만든 단일극 Drude 매질을 금박 재질로 사용 "
                         "(예: Ordal et al. 표에서 직접 읽은 값)")
    p.add_argument("--custom-drude-rod", type=str, default=None,
                    help="위와 동일하되 나노로드 재질에 대해 적용")

    # --- 구조 치수 (기본값은 논문의 스케일-축소 FDTD 모델 값) ---
    p.add_argument("--l1", type=float, default=10.0, help="슬롯 길이, um")
    p.add_argument("--w1", type=float, default=0.020, help="슬롯 폭, um")
    p.add_argument("--t1", type=float, default=0.020, help="금박 두께, um")
    p.add_argument("--l2", type=float, default=0.30, help="로드 길이(슬롯을 가로지르는 방향), um")
    p.add_argument("--t2", type=float, default=0.020, help="로드 두께, um")

    # --- 격자 해상도 및 시뮬레이션 셀 여백(padding) ---
    p.add_argument("--resolution", type=float, default=60,
                    help="1 um 당 격자점 수. 신뢰할 만한 결과를 얻으려면 250 이상 권장; "
                         "기본값은 빠르게 스모크테스트만 해보는 저해상도 값이다.")
    p.add_argument("--dpml", type=float, default=1.0, help="PML(흡수 경계층) 두께, um")
    p.add_argument("--pad-x", type=float, default=0.30, help="로드 바깥쪽 진공 여백, x, um")
    p.add_argument("--pad-y", type=float, default=1.0, help="슬롯 바깥쪽 진공 여백, y, um")
    p.add_argument("--pad-z", type=float, default=2.0, help="금박 위/아래 진공 여백, z, um")

    # --- 소스(입사 펄스) 및 주파수 스펙트럼 설정 ---
    p.add_argument("--fcen", type=float, default=0.09, help="소스 중심 주파수 (1/um)")
    p.add_argument("--df", type=float, default=0.16, help="소스 주파수 폭 (1/um)")
    p.add_argument("--nfreq", type=int, default=400, help="스펙트럼 모드에서 플럭스를 기록할 주파수 점 개수")
    p.add_argument("--freqs", type=str, default=None,
                    help="필드 모드에서 볼 주파수들, 콤마로 구분 (1/um), 예: '0.05,0.15'")
    p.add_argument("--field-z-offset", type=float, default=0.02,
                    help="필드 맵을 뽑을 평면이 금박 아랫면에서 얼마나 떨어져 있는지, um")

    # --- 기타 실행 옵션 ---
    p.add_argument("--empty", action="store_true",
                    help="금박/슬롯/로드를 전혀 넣지 않고(진공만 있는 상태로) 실행한다. "
                         "스펙트럼 모드에서 투과율을 정규화할 때 쓰는 입사 플럭스 "
                         "P_inc(f) 기준값을 얻기 위한 '레퍼런스 실행'용 옵션이다.")
    p.add_argument("--no-symmetry", action="store_true", help="거울 대칭 최적화를 끈다")
    p.add_argument("--decay-tol", type=float, default=1e-4)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--tag", type=str, default=None, help="자동 생성되는 실행 태그(파일명 접미사)를 직접 지정")
    return p.parse_args()


def make_run_tag(args):
    """이 실행을 구분하는 짧은 태그 문자열을 만든다 (출력 파일명에 쓰임).

    예: "reference"(레퍼런스), "norod"(로드 없음),
        "large_pos12"(큰 로드, 1:2 위치) 등.
    """
    if args.tag:
        return args.tag
    if args.empty:
        return "reference"
    if args.rod == "none":
        return "norod"
    pos = args.position.replace(":", "")
    return f"{args.rod}_pos{pos}"


def _parse_drude_pair(s):
    """'f_p,gamma' 형태의 문자열을 (float, float) 튜플로 변환. None이면 None 그대로."""
    if s is None:
        return None
    f_p, gamma = (float(x) for x in s.split(","))
    return (f_p, gamma)


def build_sim(args):
    """명령행 인자로부터 mp.Simulation 객체와 계산에 필요한 기하학적 정보를 만든다.

    이 함수 하나가 spectrum 모드와 field 모드 양쪽에서 공통으로 쓰이는
    "시뮬레이션 준비" 단계를 담당한다 (실제 실행(run)과 결과 추출은
    호출하는 쪽인 run_spectrum / run_field가 담당).
    """
    # 1) 슬롯(공진기) 파라미터 객체 구성. --custom-drude-film이 있으면
    #    내장 Au/Pt 대신 그 값으로 단일극 Drude 매질을 쓰게 된다.
    res = ResonatorParams(
        l1=args.l1, w1=args.w1, t1=args.t1, film_metal=args.film_metal,
        film_custom_drude=_parse_drude_pair(args.custom_drude_film),
    )

    # 2) 나노로드 파라미터 객체 구성 (--rod none 이면 rod=None으로 두어
    #    로드 없는 "맨 슬롯" 구조를 시뮬레이션한다).
    rod = None
    if args.rod != "none":
        # position_ratio: "1:2"면 (1,2), "1:1"(중앙)이면 (1,1)
        ratio = (1.0, 2.0) if args.position == "1:2" else (1.0, 1.0)
        rod = RodParams(
            s=ROD_WIDTHS[args.rod],
            l2=args.l2,
            t2=args.t2,
            rod_metal=args.rod_metal,
            rod_custom_drude=_parse_drude_pair(args.custom_drude_rod),
            position_ratio=ratio,
        )

    # 3) 시뮬레이션 셀(전체 계산 영역) 크기 결정.
    #    각 방향으로: (구조물이 차지하는 크기) + (양쪽 진공 여백) + (양쪽 PML 두께)
    #    x: 슬롯 폭(w1)과 로드 길이(l2) 중 로드가 슬롯보다 크게 걸쳐 있으므로
    #       실질적으로 l2가 x 방향 구조 크기를 좌우한다 (w1은 l2에 비해
    #       무시할 만큼 작지만 식에는 그대로 더해 둔다).
    #    y: 슬롯 길이(l1)
    #    z: 금박 두께(t1)
    sx = args.w1 + args.l2 + 2 * args.pad_x + 2 * args.dpml
    sy = args.l1 + 2 * args.pad_y + 2 * args.dpml
    sz = args.t1 + 2 * args.pad_z + 2 * args.dpml
    cell = mp.Vector3(sx, sy, sz)

    # PML을 제외한 "내부" 영역 크기. 소스 평면과 플럭스 모니터 평면의
    # 가로/세로 크기를 이것으로 맞춰서, 셀 전체 단면을 덮는 평면파에
    # 가깝게 만든다.
    inner_x = sx - 2 * args.dpml
    inner_y = sy - 2 * args.dpml

    # 소스는 금박 위쪽 PML 안쪽(진공 여백 구간)에, 모니터(투과 측정면)는
    # 금박 아래쪽 PML 안쪽에 둔다. 0.3배만큼만 파고들게 한 것은 PML에
    # 너무 가깝지도, 금박에 너무 붙지도 않는 적당한 위치를 잡기 위함이다.
    z_src = sz / 2.0 - args.dpml - 0.3 * args.pad_z  # 금박 위쪽 소스 평면
    z_mon = -sz / 2.0 + args.dpml + 0.3 * args.pad_z  # 금박 아래쪽 투과-플럭스 평면

    # --empty 이면 구조물을 아예 넣지 않는다 (진공 전파만 계산 -> 이것이
    # 바로 투과율 정규화에 쓰이는 "입사 플럭스 P_inc(f)" 기준 실행이다).
    geometry = [] if args.empty else build_geometry(res, rod, cell_x=sx, cell_y=sy)

    # --- 입사 소스: p-편광(Ex) 평면파를 광대역 가우시안 펄스로 근사 ---
    # mp.GaussianSource(frequency, fwidth): 중심주파수 fcen, 대역폭 df인
    #   시간영역 가우시안 펄스. 여러 주파수 성분을 한 번에 담고 있으므로
    #   한 번의 시간영역 시뮬레이션으로 넓은 스펙트럼 응답을 동시에 얻을 수 있다.
    # is_integrated=True: 이 소스가 "면(area) 전체에 걸친 평면파 소스"처럼
    #   동작하도록 하는 정규화 옵션 (분산성 매질과 함께 넓은 면적 소스를
    #   쓸 때 올바른 플럭스 정규화를 위해 필요).
    # component=mp.Ex: 전기장의 x-성분(=편광 방향)을 강제하는 소스.
    # size=(inner_x, inner_y, 0): z-두께 0인 평면 형태의 소스 -> x-y 평면
    #   전체에 걸쳐 동일 위상으로 진동 -> 평면파에 가까운 근사가 됨.
    sources = [
        mp.Source(
            mp.GaussianSource(frequency=args.fcen, fwidth=args.df, is_integrated=True),
            component=mp.Ex,
            center=mp.Vector3(0, 0, z_src),
            size=mp.Vector3(inner_x, inner_y, 0),
        )
    ]

    # --- 대칭성을 이용한 계산량 절감 ---
    # Meep은 구조와 소스가 특정 거울면에 대해 대칭이면, 그 대칭을 이용해
    # 절반(또는 1/4)만 계산하고 나머지는 복사해서 채워준다 (셀 크기 자체는
    # 그대로 두고 내부적으로만 최적화).
    #   - X 방향 거울(x=0 평면): 구조(슬롯, 로드)와 소스가 모두 x=0에 대해
    #     대칭이므로 항상 성립한다. 거울면 "법선" 방향 성분인 Ex는 반사에
    #     대해 부호가 뒤집히므로 phase=-1.
    #   - Y 방향 거울(y=0 평면): 로드가 없거나(rod is None), 로드가 정중앙
    #     ("1:1")에 있을 때만 y=0에 대해 구조가 대칭이 된다 ("1:2" 위치는
    #     비대칭이므로 이 대칭을 쓸 수 없다). 거울면에 접하는(tangential)
    #     성분인 Ex는 반사에 대해 부호가 유지되므로 phase=+1.
    symmetries = []
    if not args.no_symmetry:
        symmetries.append(mp.Mirror(mp.X, phase=-1))
        if rod is None or args.position == "1:1":
            symmetries.append(mp.Mirror(mp.Y, phase=+1))

    # --- 시뮬레이션 객체 생성 ---
    # boundary_layers=[mp.PML(dpml)]: 셀 6면 모두를 두께 dpml짜리 완전정합층
    #   (Perfectly Matched Layer)으로 감싼다. 바깥으로 나가는 파동을 반사
    #   없이 흡수해서, "무한히 열린 공간"을 흉내낸다.
    # default_material=진공: 구조물이 없는 영역은 모두 공기(eps=1)로 채움.
    sim = mp.Simulation(
        cell_size=cell,
        resolution=args.resolution,
        boundary_layers=[mp.PML(args.dpml)],
        geometry=geometry,
        sources=sources,
        symmetries=symmetries,
        default_material=mp.Medium(epsilon=1.0),
    )

    # sim 자체와, 이후 run_spectrum/run_field에서 재사용할 기하 정보를
    # 함께 반환한다.
    return sim, dict(
        sx=sx, sy=sy, sz=sz, inner_x=inner_x, inner_y=inner_y,
        z_src=z_src, z_mon=z_mon, res=res, rod=rod,
    )


def run_spectrum(args):
    """광대역 투과 플럭스 스펙트럼을 계산하고 .npz 파일로 저장한다.

    여기서 저장하는 것은 "투과율"이 아니라 원시(raw) 플럭스 값이다.
    투과율 T(f) = P_trans(f)/P_inc(f) 계산(정규화)은 analyze.py에서,
    이 실행과 --empty 레퍼런스 실행 두 결과를 모두 읽어서 수행한다.
    """
    sim, geo = build_sim(args)

    # --- 플럭스 모니터 설치 ---
    # mp.FluxRegion: 플럭스(단위시간당 통과하는 전자기 에너지)를 적분해서
    #   기록할 평면(단면)을 정의한다. 여기서는 금박 아래쪽(z_mon)에서
    #   내부 영역 전체(inner_x x inner_y)를 덮는 평면을 잡는다 -> "슬롯을
    #   통과해 아래로 빠져나간 에너지"를 측정하는 것과 같다.
    # sim.add_flux(fcen, df, nfreq, region): 지정한 주파수 구간
    #   [fcen-df/2, fcen+df/2]를 nfreq개의 등간격 점으로 나누어, 각 주파수
    #   에서의 플럭스를 시간에 따라 누적(푸리에 변환)하도록 등록한다.
    #   실제 계산은 sim.run() 동안 자동으로 누적되고, 아래에서
    #   get_fluxes()로 최종값을 꺼낸다.
    trans_region = mp.FluxRegion(
        center=mp.Vector3(0, 0, geo["z_mon"]),
        size=mp.Vector3(geo["inner_x"], geo["inner_y"], 0),
    )
    trans_flux = sim.add_flux(args.fcen, args.df, args.nfreq, trans_region)

    # --- 시간영역 전파 실행 ---
    # mp.stop_when_fields_decayed(dt, component, pt, tol): dt 시간 간격마다
    #   지정한 점(pt)에서 component(여기선 Ex) 값을 확인하다가, 그 값이
    #   충분히 작아지면(정확히는, 최근 최댓값 대비 tol 이하로 감쇠하면)
    #   시뮬레이션을 자동으로 멈춘다. 즉 "펄스가 다 지나가고 남은 잔향까지
    #   충분히 잦아들 때까지" 계산을 계속하되, 불필요하게 오래 돌지는
    #   않도록 하는 표준적인 종료 조건이다.
    monitor_pt = mp.Vector3(0, 0, geo["z_mon"])
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ex, monitor_pt, args.decay_tol))

    # --- 결과 추출 ---
    # get_flux_freqs: 위에서 등록한 nfreq개 주파수 점들의 실제 주파수 값
    # get_fluxes: 각 주파수에서 누적된 플럭스(적분된 파워) 값
    freqs = np.array(mp.get_flux_freqs(trans_flux))
    flux = np.array(mp.get_fluxes(trans_flux))

    os.makedirs(args.outdir, exist_ok=True)
    tag = make_run_tag(args)
    outfile = os.path.join(args.outdir, f"flux_{tag}.npz")
    # mp.am_master(): MPI 병렬 실행 시 여러 프로세스(rank)가 동시에 파일을
    #   쓰면 충돌하므로, 마스터(rank 0) 프로세스만 파일로 저장하게 한다.
    if mp.am_master():
        np.savez(outfile, freqs=freqs, flux=flux, args=vars(args))
        print(f"[simulate.py] wrote {outfile}")


def run_field(args):
    """지정한 개별 주파수들에서 |Ex|^2 근접장(near-field) 공간 분포를 뽑아 저장한다.

    spectrum 모드로 먼저 공진 주파수(1차/3차 하모닉)의 위치를 대략 파악한
    뒤, 그 주파수들을 --freqs로 넘겨서 이 모드를 돌리는 두 단계 워크플로우를
    전제로 한다 (run_all.sh 참고).
    """
    if not args.freqs:
        raise SystemExit("--freqs is required in field mode, e.g. --freqs 0.05,0.15")
    freqs = [float(x) for x in args.freqs.split(",")]

    sim, geo = build_sim(args)

    # 필드를 기록할 평면: 금박 아랫면(z = -t1/2)에서 field_z_offset만큼 더
    # 아래로 내려간 곳. 논문 Fig. 2(c) 캡션의 "공진기 출구 바로 근처에서"
    # 라는 표현에 대응한다.
    field_plane_z = -geo["res"].t1 / 2.0 - args.field_z_offset
    plane = mp.Volume(
        center=mp.Vector3(0, 0, field_plane_z),
        size=mp.Vector3(geo["inner_x"], geo["inner_y"], 0),
    )
    # --- 주파수별 DFT 필드 모니터 등록 ---
    # sim.add_dft_fields(components, fcen, df, nfreq, where=...): 지정한
    #   volume(where) 안의 격자점마다, 지정한 필드 성분(여기선 Ex)의
    #   푸리에 성분을 nfreq개 주파수에서 누적한다. add_flux와 마찬가지로
    #   원리는 같지만, 이번엔 "면을 적분한 하나의 숫자"가 아니라
    #   "평면 위 모든 점에서의 필드 값"을 그대로 저장한다는 점이 다르다.
    #
    #   여기서는 "임의의 주파수 리스트를 한 번에 넘기는" 방식 대신,
    #   원하는 주파수 f 하나마다 (fcen=f, df=0, nfreq=1)로 별도 호출해서
    #   모니터를 하나씩 따로 만든다. add_flux/add_dft_fields가 예전부터
    #   안정적으로 지원해 온 (fcen, df, nfreq) 방식만 사용해서, API 버전에
    #   덜 민감하게 만들기 위함이다.
    dft_objs = [
        sim.add_dft_fields([mp.Ex], f, 0, 1, where=plane)
        for f in freqs
    ]

    monitor_pt = mp.Vector3(0, 0, plane.center.z)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ex, monitor_pt, args.decay_tol))

    os.makedirs(args.outdir, exist_ok=True)
    tag = make_run_tag(args)
    for f, dft_obj in zip(freqs, dft_objs):
        # get_dft_array(dft_obj, component, freq_index): 등록한 DFT 모니터
        #   에서 특정 주파수 인덱스(여기선 각 모니터가 주파수 1개짜리라
        #   항상 인덱스 0)의 복소수 필드 배열을 꺼낸다. shape은 대략
        #   (모니터 평면의 x방향 격자점 수, y방향 격자점 수)가 된다.
        ex = sim.get_dft_array(dft_obj, mp.Ex, 0)
        # 논문 Fig. 2(c)/3(c)가 그리는 것은 "x-방향 전기장의 제곱"이므로
        # 복소 진폭의 절댓값 제곱(세기, intensity)을 취한다.
        intensity = np.abs(ex) ** 2
        outfile = os.path.join(args.outdir, f"field_{tag}_f{f:.4f}.npy")
        if mp.am_master():
            np.save(outfile, intensity)
            print(f"[simulate.py] wrote {outfile}  shape={intensity.shape}")


def main():
    args = parse_args()
    if args.mode == "spectrum":
        run_spectrum(args)
    else:
        run_field(args)


if __name__ == "__main__":
    main()
