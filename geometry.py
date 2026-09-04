"""THz 나노공진기(슬롯 안테나) + 나노로드 지오메트리(구조) 생성기.

[좌표계 정의 -- 논문 Fig. 1(a)와 매핑]
    z 축 : 정상입사(normal incidence)하는 THz 펄스의 전파 방향
    x 축 : 전기장(E) 편광 방향 == 슬롯 구멍의 "짧은" 변(폭, w1) 방향.
           논문에서 "입사파는 직사각형의 긴 축에 수직으로 편광되어 있다"고
           했으므로, E는 슬롯의 폭 방향(x)을 향한다.
    y 축 : 슬롯 구멍의 "긴" 변(길이, l1) 방향. 나노로드가 공진기를 따라
           어느 위치("1:2" 지점인지 "1:1"(중앙) 지점인지)에 놓이는지는
           바로 이 y축 상의 위치로 정해진다.

[구조 개요]
금박(gold film)은 두께 t1(z 방향)을 가진 "사실상 무한히 넓은" 판으로
모델링하고, 그 판에 길이 l1(y 방향) x 폭 w1(x 방향)인 직사각형 구멍
(슬롯)을 하나 뚫는다. 이것은 주기적인 격자(grating)가 아니라 "구멍
하나짜리" 단일 나노공진기이며, 논문의 "얇은 금박에 뚫린 직사각형 구멍을
가정한다"는 서술과 정확히 대응된다.

나노로드는 금박 위에 얹혀 있으며, 긴 방향 l2는 x축을 따라(즉 폭 w1의
슬롯을 가로질러 다리처럼 걸쳐지도록, Fig. 1(b)의 SEM 사진과 같은 배치),
"폭" s는 y축을 따라 놓인다. 이 s가 바로 슬롯 표피 깊이(skin depth)보다
얇아야 pinch-harmonic 효과가 나타나는, Fig. 4에서 스윕하는 핵심 변수다.
로드의 중심은 슬롯 길이 l1을 (d1 : d2) 비율로 나누는 위치에 놓이며, d1은
y = -l1/2 쪽 가장자리에서부터 잰 거리다.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import meep as mp

from materials import get_metal


@dataclass
class ResonatorParams:
    """슬롯(나노공진기) 자체의 치수를 담는 파라미터 묶음.

    논문의 스케일-축소 FDTD 모델 기본값(l1=10 um, w1=20 nm, t1=20 nm,
    Au 금박)을 기본값으로 채워 두었다.
    """
    l1: float = 10.0        # 슬롯 길이 (um), y축 방향
    w1: float = 0.020       # 슬롯 폭 (um), x축 방향  (=20 nm)
    t1: float = 0.020       # 금박 두께 (um), z축 방향 (=20 nm)
    film_metal: str = "Au"
    # (f_p, gamma) 를 직접 지정하면 meep.materials.Au 대신 이 값으로
    # 단일극 Drude 매질을 만든다 (materials.get_metal 참고)
    film_custom_drude: Optional[Tuple[float, float]] = None


@dataclass
class RodParams:
    """나노로드의 치수/위치를 담는 파라미터 묶음."""
    s: float = 0.060        # 로드 "폭" (um), y축 방향 -- 논문에서 스윕하는 핵심 변수
                             # (60 nm="large" / 20 nm="small")
    l2: float = 0.30        # 로드 "길이" (um), x축 방향 (슬롯 폭 w1보다 커야
                             # 슬롯을 완전히 가로질러 막을 수 있다)
    t2: float = 0.020       # 로드 두께 (um), z축 방향
    rod_metal: str = "Pt"
    rod_custom_drude: Optional[Tuple[float, float]] = None  # 위와 동일한 오버라이드 용도
    position_ratio: Tuple[float, float] = (1.0, 2.0)  # 슬롯 길이 l1을 나누는 (d1 : d2) 비율


def rod_y_center(res: ResonatorParams, rod: RodParams) -> float:
    """(d1 : d2) 비율이 주어졌을 때, 로드 중심의 y 좌표를 계산한다.

    좌표 원점(y=0)은 공진기(슬롯)의 정중앙이다. d1은 y = -l1/2 가장자리
    (즉 슬롯이 시작하는 왼쪽 끝)로부터 잰 거리로 정의된다.

    예) position_ratio=(1,2) 이고 l1=10 이면:
        d1_frac = 1/(1+2) = 1/3
        d1 = 10 * 1/3 = 3.333 um   (왼쪽 끝에서부터의 거리)
        y0 = d1 - l1/2 = 3.333 - 5 = -1.667 um  (중심 기준 좌표)
    position_ratio=(1,1) (한가운데, "1:1 위치")이면 항상 y0 = 0 이 된다.
    """
    d1_frac = rod.position_ratio[0] / sum(rod.position_ratio)
    d1 = d1_frac * res.l1
    return d1 - res.l1 / 2.0


def build_geometry(
    res: ResonatorParams,
    rod: Optional[RodParams],
    cell_x: float,
    cell_y: float,
):
    """주어진 전체 셀(cell) 가로/세로 크기에 맞춰 Meep geometry 리스트를 만든다.

    금박과 (있다면) 로드는 단순한 mp.Block(직육면체) 객체로 만든다. 슬롯은
    "금박 블록을 만든 다음, 그 안쪽 일부를 진공(vacuum) 블록으로 다시
    덮어씌운다"는 방식으로 구현한다 -- Meep의 geometry 리스트는 뒤에 오는
    객체가 겹치는 영역에서 앞의 객체보다 우선한다는 규칙을 이용한 것이다.
    즉 리스트 순서가 [금박, 슬롯(진공), (로드)] 여야 슬롯이 실제로
    "구멍"으로 뚫린다.

    cell_x/cell_y는 시뮬레이션 셀 전체의 가로/세로 크기다. 금박은 이
    크기와 똑같이 만들어서 PML(흡수 경계층) 안쪽까지 꽉 채운다 -- 이는
    "무한히 넓은 판"을 표현할 때 Meep에서 흔히 쓰는 방법이다. 만약 금박을
    셀보다 살짝 작게 만들면 금박 가장자리와 PML 사이에 인위적인 진공
    틈이 생기고, 그 틈에서 원치 않는 회절/반사가 발생할 수 있다.
    """

    # get_metal()은 materials.py에서 정의한 함수로, 이름에 해당하는 내장
    # Lorentz-Drude 금속(mp.materials.Au/Pt) 또는 custom_drude로 지정한
    # 단일극 Drude 매질을 반환한다.
    gold = get_metal(res.film_metal, custom_drude=res.film_custom_drude)
    vacuum = mp.Medium(epsilon=1.0)  # 진공(공기)은 유전율 1

    # --- 1) 금박 (연속된 판) ---
    # size를 (cell_x, cell_y, t1)로 주면 x,y 방향으로는 셀 전체(PML까지
    # 포함)를 덮고, z 방향으로만 두께 t1만큼의 얇은 판이 된다.
    # center=(0,0,0): 판의 중심이 원점에 오도록, 즉 z=-t1/2 ~ +t1/2 범위.
    film = mp.Block(
        size=mp.Vector3(cell_x, cell_y, res.t1),
        center=mp.Vector3(0, 0, 0),
        material=gold,
    )

    # --- 2) 슬롯 (금박 위에 뚫는 진공 구멍) ---
    # x 방향 폭 w1, y 방향 길이 l1인 작은 직육면체를 "진공"으로 만들어서
    # film 위에 겹쳐 놓으면, geometry 리스트에서 film보다 뒤에 오기 때문에
    # 그 영역은 진공으로 대체된다 = 구멍이 뚫린 효과.
    # z 방향 크기를 t1보다 아주 살짝(1e-3 um) 크게 잡은 것은, 두 블록의
    # 표면(z=±t1/2)이 부동소수점 오차로 정확히 겹치지 않아 생기는
    # "면이 딱 맞닿았을 때의 불안정한 경계 판정" 문제를 피하기 위함이다.
    slot = mp.Block(
        size=mp.Vector3(res.w1, res.l1, res.t1 + 1e-3),
        center=mp.Vector3(0, 0, 0),
        material=vacuum,
    )

    geometry = [film, slot]

    # --- 3) 나노로드 (선택 사항) ---
    if rod is not None:
        rod_metal = get_metal(rod.rod_metal, custom_drude=rod.rod_custom_drude)
        y0 = rod_y_center(res, rod)
        # z0: 로드의 중심 z좌표. 금박 윗면은 z=+t1/2 이므로, 그 위에 로드
        # 두께 t2의 절반만큼 더 올라간 위치가 로드의 중심이 된다
        # (즉 로드가 금박 윗면에 바로 얹혀 있는 형태).
        z0 = res.t1 / 2.0 + rod.t2 / 2.0
        rod_block = mp.Block(
            size=mp.Vector3(rod.l2, rod.s, rod.t2),
            center=mp.Vector3(0, y0, z0),
            material=rod_metal,
        )
        geometry.append(rod_block)  # geometry 리스트의 맨 뒤 = 가장 우선순위 높음

    return geometry
