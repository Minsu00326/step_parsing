# STEP 파싱/시각화 파이프라인 정리 (Notion용)

## 1. 프로젝트 한 줄 요약
이 프로젝트는 `STEP(.stp/.step)` CAD 파일을 OpenCascade 바인딩으로 읽어서, Face 단위 메타데이터를 추출하고, 클릭 가능한 3D 뷰어로 `Face ↔ 형상` 대응을 확인하는 도구입니다.

## 2. 사용 라이브러리와 버전
### 2.1 requirements.txt 기준
- `cadquery-ocp>=7.8.1`
- `pythonocc-core>=7.7.2` (Darwin/Windows에서만 조건부)

### 2.2 현재 실행 환경(.venv) 실제 버전
- Python: `3.12.3`
- `cadquery-ocp`: `7.9.3.1`
- `pythonocc-core`: 미설치 (현재 환경은 OCP 경로 사용)
- 프론트엔드 3D 렌더러: `three.js 0.161.0` (CDN import)

## 3. STP 파일이 담는 정보
STEP 파일은 크게 `HEADER`와 `DATA`를 담습니다.

### 3.1 HEADER 예시 정보
- 파일 스키마 (`AUTOMOTIVE_DESIGN`, AP203/AP214/AP242 계열)
- 생성 소프트웨어/시간
- 단위 (`METRE`, `RADIAN` 등)

### 3.2 DATA 예시 정보
- 형상 엔티티: `ADVANCED_FACE`, `EDGE_CURVE`, `VERTEX` 등
- 제품/파트 구조: `PRODUCT`, `MANIFOLD_SOLID_BREP` 등
- 기하 표면 타입: `PLANE`, `CYLINDRICAL_SURFACE`, `CONICAL_SURFACE`, `TOROIDAL_SURFACE` 등

### 3.3 표현/사람이 남긴 정보(비형상)
- 레이어: `PRESENTATION_LAYER_ASSIGNMENT`
- 스타일: `STYLED_ITEM`, `PRESENTATION_STYLE_ASSIGNMENT`, `SURFACE_STYLE_USAGE`
- 색상: `COLOUR_RGB`, `FILL_AREA_STYLE_COLOUR`
- 사람이 준 이름 단서: `PRODUCT`, `MANIFOLD_SOLID_BREP`, `SHAPE_REPRESENTATION`의 name 문자열

## 4. 이 프로젝트의 파싱 방식
### 4.1 입력 로드
- 엔트리: `src/main.py`
- STEP 읽기: `src/step_loader.py`의 `STEPControl_Reader`
- 모드: `STEPControl_AsIs`로 루트 전송 후 `OneShape()` 획득
- 시각표현 읽기: `STEPCAFControl_Reader`로 Face별 원본 색상(`source_color_hex/rgb`, `source_color_source`)을 병행 추출

### 4.2 텍스트 기반 헤더/이름 통계 추출
- `FILE_SCHEMA`, `FILE_NAME`, `FILE_DESCRIPTION`, `SI_UNIT` 정규식 파싱
- 추가 통계:
- `ADVANCED_FACE` 이름 존재 개수
- `EDGE_CURVE` 이름 존재 개수
- `MANIFOLD_SOLID_BREP` 이름 샘플
- `PRODUCT` 이름 샘플
- 표현 정보 통계:
- `COLOUR_RGB` 총 개수/고유 팔레트
- `PRESENTATION_LAYER_ASSIGNMENT` 총 개수/이름 있는 레이어 수
- `STYLED_ITEM` 총 개수/이름 있는 항목 수
- 사람 가독 문자열:
- 솔리드/제품 이름 목록
- 용접 키워드(`WELD|SEAM|BEAD|FILLET|JOINT`) 히트 목록

### 4.3 Topology 순회 및 Face 메타데이터 추출
- 파일: `src/topology_extract.py`
- `TopExp_Explorer`로 Face를 순회하며 `Face1..FaceN` ID 부여
- Face별 추출:
- `surface_type` (Plane/Cylinder/Cone/Torus...)
- `area`, `center_of_mass`, `bbox_min/max`
- `edge_count`
- `edge_type_counts` (Line/Circle/BSplineCurve...)
- `dominant_edge_type`
- `mean_curvature`, `gaussian_curvature`, `min_curvature`, `max_curvature` (`BRepLProp_SLProps`)
- `contact_area_total`, `contact_length_total`, `contact_pairs_top` (옵션: `--compute-contact`)
- 원통/원뿔/구/토러스의 반지름/축 파라미터
- 시각/표현 필드:
- `source_color_hex`, `source_color_rgb`, `source_color_source`
- `source_layer_name`, `source_layer_description`, `source_layer_note`
- `source_part_name`, `source_part_names`, `source_part_note` (Face가 어느 Part/솔리드 소속인지)
- STEP 공식 엔티티명:
- `surface_step_entity` (예: `PLANE`, `CYLINDRICAL_SURFACE`)
- `dominant_edge_step_entity` (예: `LINE`, `CIRCLE`, `ELLIPSE`)
- STEP 원문 참조:
- `step_advanced_face_id`, `step_advanced_face_line`, `step_advanced_face_expr`
- `step_surface_ref_id`, `step_surface_entity_raw`, `step_surface_entity_line`, `step_surface_entity_expr`
- `metric_sources` (각 수치가 STEP 원문/OCCT 어느 계산 경로에서 왔는지)
- 초보자용 필드:
- `surface_type_ko`, `surface_desc_ko`
- `edge_mix_summary_ko`
- `easy_hint_ko`
- `display_label`

### 4.4 OBJ 메쉬 내보내기
- 파일: `src/tessellate_export.py`
- `BRepMesh_IncrementalMesh`로 삼각분할
- Face 단위로 OBJ 오브젝트 분리:
- `o Face1`
- `o Face2`
- ...
- 결과적으로 뷰어에서 객체 이름으로 Face 피킹 가능

### 4.5 리포트 생성
- 파일: `src/geometry_report.py`
- `out/faces_<stem>.csv`: 표 형태 분석용
- `out/report_<stem>.json`: 뷰어/자동화 처리용
- `faces_by_id` 맵을 함께 저장해서 조회를 빠르게 함

## 5. 산출물과 관계
### 5.1 `out/model_<stem>.obj`
- Face 단위 Mesh 그룹 (`o FaceN`)을 담음
- 뷰어에서 3D 클릭/선택 대상

### 5.2 `out/report_<stem>.json`
- Face별 상세 메타데이터
- 전체 통계(`counts`, `bbox`, `total_surface_area`, `total_volume`)
- STEP 헤더/엔티티 이름 통계
- 색/레이어/스타일 통계(`presentation_stats`)
- 사람이 남긴 이름 신호(`human_named_signals`, weld 키워드 히트 포함)
- 뷰어의 사이드바 설명/필터/라벨 데이터 소스

### 5.3 `out/faces_<stem>.csv`
- 엑셀/판다스/BI로 빠르게 분석하기 좋은 형태

## 6. 뷰어에서 결과가 보이는 방식
- 파일: `viewer/index.html`
- 로딩 데이터:
- `model.obj` (형상)
- `report.json` (설명)
- 동작:
- Mesh 이름 `FaceN`으로 클릭 피킹
- `faces_by_id[FaceN]`로 메타데이터 매칭
- STEP 원본 Face 색상 기반 렌더링, 검색/필터, 정렬, 이전/다음 이동, 선택 Face 확대
- Part 필터/표시: 선택 Face가 어느 Part 소속인지 표시
- 선택 토글/해제: 같은 Face 재클릭 또는 빈 공간 클릭 또는 `ESC`
- 투명도 슬라이더로 내부 확인(X-ray 강도) 조절
- 초보자 카드:
- 한글 타입명
- 쉬운 설명
- 면적/경계선 타입/주요 치수
- 원본 색상/색상 출처/레이어 표시
- Part 이름/STEP 공식 엔티티명(표면/경계) 표시

## 7. 현재 샘플 모델 기준 결과 스냅샷
- STEP schema: `AUTOMOTIVE_DESIGN`
- OCCT binding: `OCP(cadquery-ocp)`
- transfer mode: `STEPControl_AsIs`
- topology counts:
- solids: `10`
- shells: `10`
- faces: `229`
- edges: `948`
- vertices: `1896`
- total surface area: `50948.43232242`
- total volume: `202863.32064866`
- 이름 필드 통계:
- `ADVANCED_FACE named: 0/229`
- `EDGE_CURVE named: 0/466`
- 표현 정보 통계:
- `COLOUR_RGB total: 240`, `unique: 3`
- `LAYER_ASSIGNMENT total: 240`, `named: 0`
- `STYLED_ITEM total: 240`, `named: 0`
- 사람 이름 신호:
- `MANIFOLD_SOLID_BREP` 이름 10개 중 `WELD BEATS` 포함 4개
- 해석: 이 파일은 Face/Edge 원본 이름이 거의 없어, `FaceN + 기하속성` 기반 대응이 핵심

## 8. 실행 커맨드
```bash
.venv/bin/python -m src.main --input models/SierraPacific-W431.step --out out --log-level INFO
.venv/bin/python -m src.main --input models/welding_sample_with_label.step --out out --compute-contact --contact-tolerance 1e-4 --contact-max-pairs 20000 --contact-top-k 5
python viewer/serve.py --port 8000
```

- 뷰어(`http://127.0.0.1:8000/viewer/`) 상단의 `모델 선택` 드롭다운에서 `out`, `out_my_new` 등 `out*` 폴더 산출물을 페이지 새로고침 없이 전환할 수 있습니다.

## 9. 왜 `Cone`, `Torus` 같은 이름이 보이나?
- 임의 명칭이 아니라 표준 기하 분류와 대응되는 이름입니다.
- STEP 엔티티 대응:
- `CONICAL_SURFACE` ↔ Cone
- `TOROIDAL_SURFACE` ↔ Torus
- 코드에서 보이는 이름은 OCCT `GeomAbs_*` 분류명을 사람이 읽기 쉽게 매핑한 결과입니다.

## 10. 한계와 다음 개선안
- 한계:
- 일부 STEP은 Face/Edge 이름이 `NONE`으로 비어 있음
- 그래서 CAD 피처명(예: Extrude1, Fillet2)을 직접 복원하기 어려움
- 일부 STEP은 `PRESENTATION_LAYER_ASSIGNMENT`가 있어도 레이어 이름이 공백이거나 Face 직접 매핑이 제한적
- 개선안:
- `MANIFOLD_SOLID_BREP` 이름 기준으로 Face 그룹핑
- Assembly 계층(Product/Shape Representation) 매핑 확장
- CAD 원본 feature tree와 연결 가능한 별도 매핑 파일(JSON) 지원
- XCAF Label 기반 Face↔Layer 매핑 보강(가능한 파일 형식에서)
