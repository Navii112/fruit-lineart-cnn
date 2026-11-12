"""Streamlit과 별개로 실행 가능한 과일 분리·두 CNN 비교 엔진."""
import io
import json
import threading
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import fruit_pipeline as pipeline

PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / 'fruit_outline_project'
CLASS_NAMES = ['apple', 'banana', 'lemon', 'strawberry', 'pineapple']
CLASS_KO = ['사과', '바나나', '레몬', '딸기', '파인애플']
RUN_REGISTRY = ROOT / 'results/controlled_outline_vs_lineart_v1/latest_experiment.json'
INFERENCE_LOCK = threading.RLock()


def experiment_path():
    if not RUN_REGISTRY.exists():
        raise FileNotFoundError('비교 학습 모델이 없습니다. ipynb의 3번과 4-A를 먼저 실행하세요.')
    registered = Path(json.loads(RUN_REGISTRY.read_text(encoding='utf-8'))['run_dir'])
    # 프로젝트 폴더를 다른 컴퓨터로 옮긴 경우에도 실행 폴더 이름으로 복구.
    relocated = RUN_REGISTRY.parent / registered.name
    return relocated if relocated.exists() else registered


def model_signature(run):
    seeds = json.loads((run / 'conditions.json').read_text(encoding='utf-8'))['training_seeds']
    seed = int(seeds[0])  # 테스트 점수로 seed를 고르지 않음
    files = [run / f'seed_{seed}/{variant}_{kind}'
             for variant in ['outline', 'lineart'] for kind in ['cnn.keras', 'config.json']]
    return seed, tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)


def predict_scores(model, tensors, rotation_mean=False):
    batch = np.asarray(tensors, dtype=np.float32)
    if batch.ndim != 4 or batch.shape[1:] != (96, 96, 1):
        raise ValueError('CNN 입력은 (N, 96, 96, 1)이어야 합니다.')
    if not np.isfinite(batch).all() or batch.min() < 0 or batch.max() > 1:
        raise ValueError('CNN 입력값 범위가 잘못되었습니다.')
    expanded = np.concatenate([np.rot90(batch, k, axes=(1, 2)) for k in range(4)]) if rotation_mean else batch
    probs = np.asarray(model.predict(np.ascontiguousarray(expanded), batch_size=32, verbose=0))
    if rotation_mean:
        probs = probs.reshape(4, len(batch), 5).mean(axis=0)
    if probs.shape != (len(batch), 5) or not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1, atol=1e-4):
        raise ValueError('CNN 출력이 5종 확률 형태와 다릅니다.')
    return probs


class FruitEngine:
    def __init__(self, run, seed):
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        self.models = {}
        self.run = Path(run)
        for variant in ['outline', 'lineart']:
            folder = self.run / f'seed_{seed}'
            cfg = json.loads((folder / f'{variant}_config.json').read_text(encoding='utf-8'))
            if cfg['classes'] != CLASS_NAMES or cfg['image_size'] != 96 or cfg['lineart_config'] != pipeline.LINEART_CONFIG:
                raise ValueError('학습 모델의 종류·크기·전처리 설정이 프로토타입과 다릅니다.')
            self.models[variant] = tf.keras.models.load_model(folder / f'{variant}_cnn.keras', compile=False)
        self.seed = seed

    def analyze(self, photo_bytes, variants, rotation_mean=False):
        if not variants or any(v not in self.models for v in variants):
            raise ValueError('분류할 CNN을 선택하세요.')
        with INFERENCE_LOCK:
            rgb, objects, audit = pipeline.detect_and_segment_fruits(io.BytesIO(photo_bytes))
            rows = []
            if objects:
                for variant in variants:
                    tensors = np.stack([item[variant].astype(np.float32)[..., None] / 255 for item in objects])
                    scores = predict_scores(self.models[variant], tensors, rotation_mean)
                    for idx, (item, prob) in enumerate(zip(objects, scores), 1):
                        best = int(prob.argmax())
                        ys, xs = np.where(item['mask'])
                        touches_border = bool(xs.min() == 0 or ys.min() == 0 or xs.max() == rgb.shape[1]-1 or ys.max() == rgb.shape[0]-1)
                        rows.append({'과일 번호': idx, '입력': '내부 특징 선화' if variant == 'lineart' else '외곽선',
                                     '예측': CLASS_KO[best], '모델 점수': float(prob[best]), 'variant': variant,
                                     'class': CLASS_NAMES[best], '사진 경계에 닿음': touches_border,
                                     **{f'점수_{name}': float(p) for name, p in zip(CLASS_NAMES, prob)}})
            return {'rgb': rgb, 'objects': objects, 'audit': audit, 'table': pd.DataFrame(rows),
                    'run': self.run.name, 'seed': self.seed, 'rotation_mean': rotation_mean}


def overview(result):
    image = Image.fromarray(result['rgb']).copy()
    draw = ImageDraw.Draw(image)
    font_path = Path('C:/Windows/Fonts/malgun.ttf')
    font = ImageFont.truetype(str(font_path), max(14, image.width // 45)) if font_path.exists() else ImageFont.load_default()
    for idx, obj in enumerate(result['objects'], 1):
        ys, xs = np.where(obj['mask'])
        box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        draw.rectangle(box, outline='#20b786', width=max(2, image.width // 300))
        rows = result['table'][result['table']['과일 번호'] == idx]
        main = rows[rows.variant == 'lineart'].iloc[0] if (rows.variant == 'lineart').any() else rows.iloc[0]
        name = main['예측'] if font_path.exists() else main['class']
        label = f"#{idx} {name} {main['모델 점수']:.0%}"
        x, y = box[0], box[1]
        bounds = draw.textbbox((x, y), label, font=font)
        draw.rectangle(bounds, fill='#123f32')
        draw.text((x, y), label, fill='white', font=font)
    return image


def png_bytes(array):
    image = array if isinstance(array, Image.Image) else Image.fromarray(np.asarray(array, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def result_zip(result):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('predictions.csv', result['table'].to_csv(index=False).encode('utf-8-sig'))
        archive.writestr('overview.png', png_bytes(overview(result)))
        archive.writestr('photo.png', png_bytes(result['rgb']))
        archive.writestr('settings.json', json.dumps({k: result[k] for k in ['run', 'seed', 'rotation_mean']}, indent=2))
        for idx, obj in enumerate(result['objects'], 1):
            cutout = result['rgb'].copy()
            cutout[~obj['mask']] = 255
            archive.writestr(f'fruit_{idx:02d}/cutout.png', png_bytes(cutout))
            archive.writestr(f'fruit_{idx:02d}/mask.png', png_bytes(obj['mask'].astype(np.uint8) * 255))
            archive.writestr(f'fruit_{idx:02d}/outline.png', png_bytes(255 - obj['outline']))
            archive.writestr(f'fruit_{idx:02d}/lineart.png', png_bytes(255 - obj['lineart']))
    return buffer.getvalue()
