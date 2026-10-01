# KAMP X-ray Research

X-ray 영상의 희소 이물 탐지를 위한 잔차 증강 코드. 2026 KAMP 제조데이터 분석 경진대회를 준비하며 진행 중인 연구를 바탕으로 정리했다.

학습 영상에서 국소 배경을 제거해 이물 코어를 추출하고, 장비별 잔차 bank를 구성한다. 합성 표적은 동일 장비의 제품 내부에 삽입하며, 경계와 기존 표적까지의 거리를 제한한다.

![잔차 증강 및 탐지 결과](preview.png)

## 실행

Python 3.10 이상. 기본 실행에는 NumPy와 Pillow만 필요하다.

```bash
git clone https://github.com/gigalove6783-pixel/kamp-xray-research.git
cd kamp-xray-research
python -m pip install -r requirements.txt
python run_pipeline.py --seed 2026 --output runs/example
python -m unittest -v
```

`runs/example`에 이미지·YOLO 라벨, `dataset.yaml`, `preview.png`, `metrics.json`이 생성된다. 기본 설정은 학습 12장, 검증 24장이다.

## 구성

- `augmentation.py`: 국소 배경 추정, 장비별 잔차 추출, 제품 영역 내 삽입
- `evaluation.py`: 국소 대비 기반 탐지, 일대일 매칭, point AP·정밀도·재현율 계산
- `run_pipeline.py`: 데이터 생성과 평가 실행
- `yolo_adapter.py`: 로컬 YOLO 체크포인트의 학습·추론 연결
- `test_pipeline.py`: 삽입 범위, 재현성, 장비 분리, 중복 검출 처리 검사

잔차는 반경 7–10픽셀의 주변 영역으로 배경을 추정한 뒤, 중심 반경 2픽셀의 어두운 신호만 남긴다. 삽입할 때는 9×9 패치 전체가 제품 마스크 안에 있어야 하며 표적 중심 간 거리를 20픽셀 이상 유지한다. 검증 영상은 학습 잔차 bank에 포함하지 않는다.

## 평가

예측을 점수순으로 정렬하고 같은 영상의 정답과 3픽셀 이내에서 일대일 매칭한다. 중복 예측은 미매칭 후보로 집계한다. `point_ap_3px`는 이 매칭 기준의 precision-recall 면적이며 COCO box mAP와 다르다.

[example_metrics.json](example_metrics.json)은 seed 2026 실행 결과다. 수록된 데이터는 절차적으로 생성한 합성 영상이며, 이 수치는 실제 KAMP 성능이나 증강에 따른 개선량을 나타내지 않는다. 기본 탐지기는 학습 가중치 없이 국소 대비를 사용한다.

## YOLO

Ultralytics를 별도로 설치하고 로컬 체크포인트를 지정한다.

```bash
python -m pip install ultralytics
python yolo_adapter.py train --weights /path/to/local.pt --data runs/example/dataset.yaml --device cpu --epochs 10
python yolo_adapter.py predict --weights /path/to/local.pt --source runs/example/images/val --device cpu
```

GPU는 `--device 0`으로 지정한다. CPU 파이프라인과 단위 검사를 실행 확인했으며, YOLO 어댑터의 학습·추론은 이 공개 코드에서 별도로 검증하지 않았다. Ultralytics 사용에는 해당 프로젝트의 라이선스가 적용된다.

## 데이터

대회 원본 영상, 원본에서 추출한 잔차, 학습 가중치는 저장소에 포함하지 않는다. 여기서는 합성 영상의 제품 마스크와 고정 표적 크기를 사용한다. 실제 연구의 영상 영역 추정·장비별 표적 크기 설정·GPU 학습·병렬 추론은 별도 파이프라인에서 다룬다.
