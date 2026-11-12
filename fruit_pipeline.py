"""학습 노트북과 동일한 선화 변환 및 자동 분리 함수. UI·학습 코드는 실행하지 않음."""
import os
os.environ.setdefault('USE_TF', '0')
import inspect, json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps
ROOT = Path(__file__).resolve().parent / 'fruit_outline_project'
RESULT_DIR = ROOT / 'results' / 'lineart_v1_apple_banana_lemon_strawberry_pineapple'
SCENE_DIR = RESULT_DIR / 'automatic_scenes'
AUTO_CACHE = ROOT / 'model_cache'
IMAGE_SIZE = 96
LINEART_CONFIG = dict(margin=10, line_width=1, canny_low=35, canny_high=100,
    adaptive_block=11, adaptive_c=5, bilateral_d=5, bilateral_sigma=35, min_component_pixels=2)
DETECTOR_ID = 'IDEA-Research/grounding-dino-tiny'
SEGMENTER_ID = 'Zigeng/SlimSAM-uniform-77'
DETECTION_TEXT = 'an apple. a banana. a lemon. a strawberry. a pineapple.'
BOX_THRESHOLD, TEXT_THRESHOLD, BOX_NMS_IOU, MASK_NMS_IOU = 0.28, 0.25, 0.65, 0.80
MAX_OBJECTS, MAX_PHOTO_SIDE, MIN_MASK_PIXELS, MIN_SAM_SCORE = 20, 1024, 25, 0.50
_SCENE_MODELS = None


def read_rgb(source):
    if isinstance(source,np.ndarray):
        if source.ndim!=3 or source.shape[2]!=3: raise ValueError('RGB 3채널 배열이 필요합니다.')
        return source.astype(np.uint8)
    with Image.open(source) as image:
        image=ImageOps.exif_transpose(image).convert('RGBA')
        white=Image.new('RGBA',image.size,(255,255,255,255))
        return np.asarray(Image.alpha_composite(white,image).convert('RGB'))

def clean_object_mask(mask):
    binary=(np.asarray(mask)>0).astype(np.uint8)*255
    if binary.ndim!=2: raise ValueError('물체 마스크는 2차원이어야 합니다.')
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError('물체 영역이 없습니다.')
    contour=max(contours,key=cv2.contourArea)
    if cv2.contourArea(contour)<2: raise ValueError('물체 영역이 너무 작습니다.')
    clean=np.zeros_like(binary)
    cv2.drawContours(clean,[contour],-1,255,cv2.FILLED)
    return clean

def fruit_to_lineart(rgb,mask,image_size=96,margin=10,line_width=1,
                     canny_low=35,canny_high=100,adaptive_block=11,adaptive_c=5,
                     bilateral_d=5,bilateral_sigma=35,min_component_pixels=2):
    """학습과 실제 사진에 공통 적용: 외곽선 + 내부 명암 경계/무늬를 이진 선화로 변환."""
    rgb=read_rgb(rgb)
    clean=clean_object_mask(mask)
    if clean.shape!=rgb.shape[:2]: raise ValueError('사진과 마스크의 크기가 다릅니다.')
    if not 0<=margin<image_size/2: raise ValueError('선화 여백 설정이 잘못되었습니다.')
    if not 0<=canny_low<canny_high<=255: raise ValueError('Canny 기준값을 확인하세요.')
    if adaptive_block<3 or adaptive_block%2==0: raise ValueError('adaptive_block은 3 이상 홀수여야 합니다.')
    # 배경의 선이 CNN으로 전달되지 않도록 물체 외부를 먼저 흰색으로 치환
    isolated=rgb.copy(); isolated[clean==0]=255
    ys,xs=np.where(clean>0)
    x,y,w,h=int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)
    ratio=(image_size-2*margin)/max(w,h)
    nw,nh=max(1,round(w*ratio)),max(1,round(h*ratio))
    left,top=(image_size-nw)//2,(image_size-nh)//2
    normalized_rgb=np.full((image_size,image_size,3),255,np.uint8)
    normalized_mask=np.zeros((image_size,image_size),np.uint8)
    interp=cv2.INTER_AREA if ratio<1 else cv2.INTER_CUBIC
    normalized_rgb[top:top+nh,left:left+nw]=cv2.resize(isolated[y:y+h,x:x+w],(nw,nh),interpolation=interp)
    normalized_mask[top:top+nh,left:left+nw]=cv2.resize(clean[y:y+h,x:x+w],(nw,nh),interpolation=cv2.INTER_NEAREST)
    normalized_rgb[normalized_mask==0]=255
    gray=cv2.cvtColor(normalized_rgb,cv2.COLOR_RGB2GRAY)
    smooth=cv2.bilateralFilter(gray,int(bilateral_d),float(bilateral_sigma),float(bilateral_sigma))
    canny=cv2.Canny(smooth,int(canny_low),int(canny_high),L2gradient=True)
    dark_details=255-cv2.adaptiveThreshold(smooth,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY,int(adaptive_block),float(adaptive_c))
    # 흰 배경과 만나는 경계 띠를 내부 특징으로 오인하지 않도록 제외
    radius=max(2,int(adaptive_block)//2)
    interior=cv2.erode(normalized_mask,np.ones((2*radius+1,2*radius+1),np.uint8),iterations=1)
    detail_edges=cv2.Canny(dark_details,50,150)
    internal=cv2.bitwise_and(cv2.bitwise_or(canny,detail_edges),interior)
    # 작은 점 잡음은 제거하되, 씨/껍질 무늬 등 연결된 작은 특징은 유지
    if min_component_pixels>1:
        count,labels,stats,_=cv2.connectedComponentsWithStats(internal,connectivity=8)
        keep=np.zeros(count,np.uint8)
        if count>1: keep[1:]=(stats[1:,cv2.CC_STAT_AREA]>=min_component_pixels).astype(np.uint8)*255
        internal=keep[labels]
    outline=np.zeros_like(normalized_mask)
    contours,_=cv2.findContours(normalized_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outline,contours,-1,255,int(line_width))
    lineart=cv2.bitwise_or(outline,internal)
    tensor=lineart.astype('float32')[...,None]/255.0
    details={'rgb':rgb,'mask':clean,'normalized_rgb':normalized_rgb,'normalized_mask':normalized_mask,
             'outline':outline,'internal':internal,'lineart':lineart,'drawing':255-lineart,
             'bbox':(x,y,w,h),'touches_border':bool(x==0 or y==0 or x+w==clean.shape[1] or y+h==clean.shape[0])}
    return tensor,details

def load_scene_models():
    global _SCENE_MODELS
    if _SCENE_MODELS is not None: return _SCENE_MODELS
    # Transformers에서 Keras를 자동으로 로드하지 않도록 PyTorch만 사용
    os.environ['USE_TF'] = '0'
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    try:
        import torch
        import transformers
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
    except (ImportError, OSError) as exc:
        raise RuntimeError('자동 분리 라이브러리를 불러오지 못했습니다. 설치 셀 실행 후 커널을 재시작하세요.') from exc
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('자동 분리 모델 준비:', device, '| 저장된 로컬 캐시를 사용합니다.', flush=True)
    # 기존 노트북이 내려받은 캐시를 먼저 사용해 매 실행의 네트워크 확인을 생략.
    kw = {'cache_dir':str(AUTO_CACHE)}
    try:
        detector_processor = AutoProcessor.from_pretrained(DETECTOR_ID, **kw)
        detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            DETECTOR_ID, use_safetensors=True, **kw).to(device).eval()
        sam_processor = SamProcessor.from_pretrained(SEGMENTER_ID, **kw)
        sam = SamModel.from_pretrained(SEGMENTER_ID, use_safetensors=True, **kw).to(device).eval()
    except Exception as exc:
        raise RuntimeError('자동 분리 모델 캐시를 불러오지 못했습니다. 기존 ipynb의 6-A, 6-B로 모델을 먼저 내려받고 앱을 재시작하세요.') from exc
    _SCENE_MODELS = (torch, device, detector_processor, detector, sam_processor, sam)
    (SCENE_DIR/'model_versions.json').write_text(json.dumps({
        'detector':DETECTOR_ID, 'segmenter':SEGMENTER_ID,
        'detector_revision':getattr(detector.config,'_commit_hash',None),
        'segmenter_revision':getattr(sam.config,'_commit_hash',None),
        'transformers':transformers.__version__, 'torch':torch.__version__, 'device':device
    },indent=2),encoding='utf-8')
    return _SCENE_MODELS

def box_iou(a,b):
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    wh = np.maximum(0, np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    inter = float(wh[0]*wh[1])
    area_a = float(np.prod(np.maximum(0,a[2:]-a[:2])))
    area_b = float(np.prod(np.maximum(0,b[2:]-b[:2])))
    return inter/max(area_a+area_b-inter,1e-9)

def suppress_boxes(boxes,scores,threshold=0.65):
    kept=[]
    for idx in np.argsort(-np.asarray(scores)):
        if all(box_iou(boxes[idx],boxes[j]) <= threshold for j in kept): kept.append(int(idx))
    return kept

def mask_iou(a,b):
    a,b = np.asarray(a,dtype=bool),np.asarray(b,dtype=bool)
    return float(np.logical_and(a,b).sum()/max(np.logical_or(a,b).sum(),1))

def suppress_group_masks(objects):
    # 여러 개별 후보를 거의 통째로 포함하는 큰 합성 마스크를 제외하는 휴리스틱
    kept=[]; removed=[]
    for i,item in enumerate(objects):
        large=item['mask']; area=int(large.sum()); contained=[]
        for j,other in enumerate(objects):
            if i==j: continue
            small=other['mask']; small_area=int(small.sum())
            coverage=np.logical_and(large,small).sum()/max(small_area,1)
            if area>1.8*small_area and coverage>0.85:
                contained.append(j)
        has_distinct_pair=any(mask_iou(objects[a]['mask'],objects[b]['mask'])<0.25
                              for pos,a in enumerate(contained) for b in contained[pos+1:])
        if has_distinct_pair: removed.append(item)
        else: kept.append(item)
    return kept,removed

def detect_and_segment_fruits(source):
    torch,device,dp,detector,sp,sam = load_scene_models()
    rgb=read_rgb(source)
    h,w=rgb.shape[:2]
    scale=min(1.0,MAX_PHOTO_SIDE/max(h,w))
    if scale<1: rgb=cv2.resize(rgb,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
    h,w=rgb.shape[:2]
    image=Image.fromarray(rgb)
    print('1/3: 과일 후보 위치 자동 탐지 중...',flush=True)
    inputs=dp(images=image,text=DETECTION_TEXT,return_tensors='pt').to(device)
    with torch.inference_mode(): outputs=detector(**inputs)
    post=dp.post_process_grounded_object_detection
    threshold_name='box_threshold' if 'box_threshold' in inspect.signature(post).parameters else 'threshold'
    detected=post(outputs,inputs.input_ids,text_threshold=TEXT_THRESHOLD,
                  target_sizes=[(h,w)],**{threshold_name:BOX_THRESHOLD})[0]
    boxes=detected['boxes'].detach().cpu().numpy().reshape(-1,4)
    scores=detected['scores'].detach().cpu().numpy()
    labels=detected['text_labels'] if 'text_labels' in detected else detected.get('labels',['']*len(boxes))
    boxes[:,[0,2]]=boxes[:,[0,2]].clip(0,w)
    boxes[:,[1,3]]=boxes[:,[1,3]].clip(0,h)
    valid=np.flatnonzero((boxes[:,2]-boxes[:,0]>=3)&(boxes[:,3]-boxes[:,1]>=3))
    indices=[int(valid[i]) for i in suppress_boxes(boxes[valid],scores[valid],BOX_NMS_IOU)]
    if len(indices)>MAX_OBJECTS:
        print(f'후보 {len(indices)}개 중 점수가 높은 {MAX_OBJECTS}개만 처리합니다.')
        indices=indices[:MAX_OBJECTS]
    if not indices:
        return rgb,[],[{'status':'no_detection','reason':'설정한 5종의 과일 후보를 찾지 못했습니다.'}]
    print(f'2/3: 후보 {len(indices)}개 영역 분리 중...',flush=True)
    base=sp(images=image,return_tensors='pt').to(device)
    with torch.inference_mode(): embeddings=sam.get_image_embeddings(base['pixel_values'])
    accepted=[]; audit=[]
    for number,idx in enumerate(indices,1):
        box=boxes[idx].tolist()
        prompt=sp(images=image,input_boxes=[[box]],return_tensors='pt').to(device)
        with torch.inference_mode():
            sam_output=sam(image_embeddings=embeddings,input_boxes=prompt['input_boxes'],multimask_output=True)
        restored=sp.image_processor.post_process_masks(sam_output.pred_masks.cpu(),
                    prompt['original_sizes'].cpu(),prompt['reshaped_input_sizes'].cpu())[0][0]
        quality=sam_output.iou_scores.detach().cpu().numpy()[0,0]
        best=int(np.argmax(quality))
        mask=restored[best].numpy().astype(bool)
        # SAM이 박스 바깥의 다른 물체까지 포함하는 경우를 줄임
        x1,y1,x2,y2=box
        px=max(2,int((x2-x1)*0.08)); py=max(2,int((y2-y1)*0.08))
        allowed=np.zeros((h,w),bool)
        allowed[max(0,int(y1)-py):min(h,int(np.ceil(y2))+py),max(0,int(x1)-px):min(w,int(np.ceil(x2))+px)]=True
        mask &= allowed
        diagnostic={'candidate':number,'detector_hint':str(labels[idx]),'detector_score':float(scores[idx]),
                    'sam_score':float(quality[best]),'box':box}
        try:
            if quality[best] < MIN_SAM_SCORE: raise ValueError('분리 모델의 마스크 품질 점수가 낮음')
            tensor,details=fruit_to_lineart(rgb,mask,IMAGE_SIZE,**LINEART_CONFIG)
            binary=details['mask']>0
            if binary.sum()<MIN_MASK_PIXELS: raise ValueError('분리 영역이 너무 작음')
            if binary.mean()>0.90: raise ValueError('사진 대부분을 차지하는 잘못된 분리 영역')
            if any(mask_iou(binary,item['mask'])>MASK_NMS_IOU for item in accepted):
                raise ValueError('다른 후보와 중복된 마스크')
            accepted.append({'mask':binary,'tensor':tensor,'outline':details['outline'],'lineart':details['lineart'],
                             'normalized_mask':details['normalized_mask'],'diagnostic':diagnostic})
            audit.append({**diagnostic,'status':'accepted','reason':''})
        except ValueError as exc:
            audit.append({**diagnostic,'status':'rejected','reason':str(exc)})
        print(f'  영역 처리 {number}/{len(indices)}',flush=True)
    accepted,groups=suppress_group_masks(accepted)
    for item in groups:
        candidate=item['diagnostic']['candidate']
        for record in audit:
            if record.get('candidate')==candidate:
                record.update(status='rejected_group_mask',reason='여러 개별 후보를 포함한 큰 합성 영역')
    return rgb,accepted,audit
