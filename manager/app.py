import hmac, io, json, os, re, time, uuid, zipfile, threading, hashlib, shutil, tempfile, stat
from pathlib import Path
import redis
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

app=FastAPI(title='AI Engineering GPU Cluster',version='0.8.0')
MAX_CONCURRENT_JOBS=max(1,min(8,int(os.environ.get('GPU_MAX_CONCURRENT_JOBS','3'))))
VRAM_HEADROOM_MIB=int(os.environ.get('VRAM_HEADROOM_MIB','2048'))
VRAM_BASELINE_MIB=int(os.environ.get('VRAM_BASELINE_MIB','1100'))
VRAM_SAFETY_MULTIPLIER=float(os.environ.get('VRAM_SAFETY_MULTIPLIER','1.5'))
# Device-wide isolated observations, never derived from simultaneous jobs.
# Values are MiB *above* baseline; only confirmed RTX 5060 workloads initially.
OBSERVED_DELTA={
    'rtx 5060':{('yolo26n.pt',512,4):773,('yolo26n.pt',640,8):1706},
    'rtx a4000':{('yolo26n.pt',512,4):876,('yolo26n.pt',640,4):1145,
                 ('yolo26n.pt',640,8):1783,('yolo26s.pt',640,4):1990},
}

def gpu_family(device):
    name=str(device).lower()
    return next((family for family in OBSERVED_DELTA if family in name),None)

def node_limit(node,reported):
    # Any added node gets one conservative job by default; explicitly opt in per worker.
    return min(MAX_CONCURRENT_JOBS,max(1,min(8,int(reported))))


r=redis.Redis.from_url(os.environ['REDIS_URL'],decode_responses=True)
api_key=os.environ['API_KEY']; worker_key=os.environ['WORKER_KEY']
results=Path(os.environ.get('RESULTS_DIR','/results'));results.mkdir(parents=True,exist_ok=True)
datasets=Path(os.environ.get('DATASETS_DIR','/datasets'))
SAFE=re.compile(r'^[a-zA-Z0-9_-]{1,64}$')
def check(key, secret):
    if not key or not hmac.compare_digest(key,secret): raise HTTPException(401,'Invalid API key')
def user(x_api_key:str|None=Header(default=None)): check(x_api_key,api_key)
def agent(x_worker_key:str|None=Header(default=None)): check(x_worker_key,worker_key)
def record(jobid):
    raw=r.get('job:'+jobid)
    if not raw: raise HTTPException(404,'Unknown job')
    return json.loads(raw)
def save(j): r.set('job:'+j['id'],json.dumps(j))
class Job(BaseModel):
    student_id:str=Field(min_length=1,max_length=64)
    mode:str='train'
    dataset:str='dates-v16'
    model:str='yolo26n.pt'
    epochs:int=Field(default=3,ge=1,le=200)
    imgsz:int=Field(default=512,ge=160,le=640)
    batch:int=Field(default=4,ge=1,le=8)
    source_job_id:str|None=None  # Train from completed run best.pt

STALE_SECONDS=int(os.environ.get('STALE_JOB_SECONDS','120'))
RECOVERY_INTERVAL=int(os.environ.get('RECOVERY_INTERVAL_SECONDS','15'))

# Values collected on the RTX 5060 by the project owner (2026-10).
# PyTorch allocator figures, NOT guaranteed total hardware VRAM consumption.
SEED_PROFILES = [
    {'model':'yolo26n.pt','imgsz':512,'batch':4,'amp':True,'peak_allocated_gib':0.501,'peak_reserved_gib':0.529},
    {'model':'yolo26n.pt','imgsz':640,'batch':4,'amp':True,'peak_allocated_gib':0.698,'peak_reserved_gib':0.791},
    {'model':'yolo26n.pt','imgsz':640,'batch':8,'amp':True,'peak_allocated_gib':1.237,'peak_reserved_gib':1.422},
    {'model':'yolo26s.pt','imgsz':640,'batch':4,'amp':True,'peak_allocated_gib':1.457,'peak_reserved_gib':1.562},
]

def profile_key(model,imgsz,batch,amp=True):
    return f'profile:{model}:{imgsz}:{batch}:{int(amp)}'

def profile_for(model,imgsz,batch):
    raw=r.get(profile_key(model,imgsz,batch))
    return json.loads(raw) if raw else None

@app.on_event('startup')
def seed_profiles():
    for row in SEED_PROFILES:
        key=profile_key(row['model'],row['imgsz'],row['batch'])
        r.setnx(key,json.dumps({**row,'source':'user-benchmark','device':'RTX 5060','sample_count':1}))

def estimate_reservation(j,device):
    if j.get('mode')!='train':return None
    family=gpu_family(device)
    delta=OBSERVED_DELTA.get(family,{}).get((j.get('model'),j.get('imgsz'),j.get('batch')))
    return int((delta*VRAM_SAFETY_MULTIPLIER)+0.999) if delta is not None else None

@app.get('/scheduler/policy')
def policy(x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    return {'max_jobs_server_cap':MAX_CONCURRENT_JOBS,'headroom_mib':VRAM_HEADROOM_MIB,
            'baseline_reserve_mib':VRAM_BASELINE_MIB,'safety_multiplier':VRAM_SAFETY_MULTIPLIER,
            'profiles_by_gpu':{family:[{'model':m,'imgsz':size,'batch':batch,'delta_mib':v,
                'reservation_mib':int(v*VRAM_SAFETY_MULTIPLIER+0.999)} for (m,size,batch),v in profiles.items()]
                for family,profiles in OBSERVED_DELTA.items()},
            'unknown_profile':'exclusive_only','placement':'idle-first, then normalized reservation load',
            'hard_vram_limit':False}

@app.get('/profiles')
def profiles(x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    rows=[]
    for key in r.scan_iter('profile:*'):
        value=r.get(key)
        if value: rows.append(json.loads(value))
    return sorted(rows,key=lambda p:(p.get('model',''),p.get('imgsz',0),p.get('batch',0)))

@app.get('/jobs/{jobid}/metrics')
def job_metrics(jobid:str,x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    j=record(jobid)
    return j.get('metrics') or {'status':'not_recorded'}

@app.post('/internal/metrics/{jobid}')
def upload_metrics(jobid:str,data:dict,x_worker_key:str|None=Header(default=None)):
    agent(x_worker_key)
    j=record(jobid)
    if j['status']!='running' or j['node']!=data.get('node'):
        raise HTTPException(409,'Job ownership mismatch')
    allowed=('peak_gpu_used_mib','baseline_gpu_used_mib','peak_delta_gpu_used_mib',
             'peak_allocated_gib','peak_reserved_gib','sample_seconds','measurement_notes','concurrent_observation')
    j['metrics']={k:data[k] for k in allowed if k in data}
    j['updated_at']=time.time()
    save(j)
    # Store actual PyTorch peaks separately. Profile never implies hard memory isolation.
    if j['mode']=='train' and not data.get('concurrent_observation') and data.get('peak_reserved_gib') is not None:
        key=profile_key(j['model'],j['imgsz'],j['batch'])
        old=profile_for(j['model'],j['imgsz'],j['batch']) or {}
        observed=float(data['peak_reserved_gib'])
        profile={
            'model':j['model'],'imgsz':j['imgsz'],'batch':j['batch'],'amp':True,
            'device':data.get('device','RTX 5060'),
            'peak_allocated_gib':data.get('peak_allocated_gib'),
            'peak_reserved_gib':max(observed,float(old.get('peak_reserved_gib',0))),
            'sample_count':int(old.get('sample_count',0))+1,
            'source':'observed-running-max',
        }
        r.set(key,json.dumps(profile))
    return {'ok':True}

@app.get('/health')
def health(): return {'status':'ok','redis':r.ping()}
@app.get('/')
def root(): return FileResponse(Path(__file__).parent/'dashboard.html')
@app.post('/jobs')
def submit(j:Job,x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    if not SAFE.fullmatch(j.student_id) or not SAFE.fullmatch(j.dataset): raise HTTPException(422,'Invalid student/dataset name')
    if j.mode not in ('train','smoke'): raise HTTPException(422,'Invalid mode')
    if j.model not in ('yolo26n.pt','yolo26s.pt'): raise HTTPException(422,'Model not on MVP allowlist')
    if j.source_job_id:
        if not re.fullmatch(r'[a-f0-9]{32}',j.source_job_id):raise HTTPException(422,'Invalid source job')
        ancestor=record(j.source_job_id)
        if (ancestor.get('student_id')!=j.student_id or ancestor.get('mode')!='train' or
            ancestor.get('status')!='completed' or ancestor.get('model')!=j.model):
            raise HTTPException(422,'Source must be a completed training of the same model and student')
        locate_best_checkpoint(ancestor)
    if j.imgsz%32: raise HTTPException(422,'imgsz must be multiple of 32')
    # Conservative input limits, NOT hard 4 GiB VRAM enforcement.
    if j.mode=='train' and j.model=='yolo26s.pt' and j.batch>4: raise HTTPException(422,'YOLO26s batch limited to 4 until profiled')
    data=j.model_dump();data['memory_profile']=profile_for(j.model,j.imgsz,j.batch) if j.mode=='train' else None
    data['profile_note']='Measured isolated device delta available for B01/B03 on RTX 5060 only; other configurations run exclusively.'
    data.update(id=uuid.uuid4().hex,status='queued',node=None,created_at=time.time(),updated_at=time.time(),error=None,artifact=None)
    with r.lock('claim_lock',timeout=30,blocking_timeout=10):
        save(data);r.lpush('pending',data['id']);r.lpush('jobs_recent',data['id']);r.ltrim('jobs_recent',0,1999)
    return data
def _clear_children(root):
    """Clear only files under the configured application storage root."""
    root=root.resolve(strict=True)
    if root not in (Path('/results'),Path('/datasets')):
        # Resolve may return host-independent mount paths; do not follow
        # a symlink from an expected root to another location.
        raise HTTPException(409,'Unexpected storage root')
    count=0
    for entry in root.iterdir():
        if entry.is_symlink() or entry.is_file():entry.unlink()
        elif entry.is_dir():shutil.rmtree(entry)
        count+=1
    return count

def _assert_idle_reset():
    # Stop claiming jobs first; workers must have already finished.
    for key in r.scan_iter('running:*'):
        if r.scard(key):raise HTTPException(409,'Running jobs detected; stop workers and retry')
    for key in r.scan_iter('node:*'):
        raw=r.get(key)
        if not raw:continue
        try:node=json.loads(raw)
        except (ValueError,TypeError):continue
        if node.get('current_job') or node.get('current_jobs'):
            raise HTTPException(409,'A worker reports active training; stop workers and retry')

@app.post('/admin/reset')
def reset_competition_data(payload:dict,x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    if payload.get('confirmation')!='DELETE ALL COMPETITION DATA':
        raise HTTPException(422,'Confirmation phrase does not match')
    with r.lock('claim_lock',timeout=120,blocking_timeout=10):
        r.set('scheduler:paused','1')
        _assert_idle_reset()
        # Scope 1: job history, scheduling queue, metrics and training files.
        keys=['pending','jobs_recent']
        for pattern in ('job:*','running:*','cancel:*','profile:*','node:reconciled:*','node:quarantine:*'):
            keys.extend(r.scan_iter(pattern))
        keys=[key for key in set(keys) if key!='claim_lock']
        if keys:r.delete(*keys)
        result_count=_clear_children(results)
        # Scope 2: competition dataset uploads only.
        dataset_count=_clear_children(datasets)
        # Invalidate remote UI state, without modifying worker registrations or keys.
        generation=r.incr('competition:reset_generation')
    return {'ok':True,'results_removed':result_count,'datasets_removed':dataset_count,
            'generation':generation,'paused':True}

@app.get('/jobs')
def list_jobs(x_api_key:str|None=Header(default=None),limit:int=50):
    user(x_api_key)
    return [record(i) for i in r.lrange('jobs_recent',0,min(max(limit,1),200)-1)]
@app.get('/jobs/{jobid}')
def get_job(jobid:str,x_api_key:str|None=Header(default=None)):
    user(x_api_key);return record(jobid)
@app.get('/nodes')
def nodes(x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    workers=[]
    for key in r.scan_iter('node:*'):
        # Only node:<name> holds a heartbeat JSON object; internal keys
        # such as node:reconciled:<name> contain plain text/timestamps.
        if key.count(':') != 1:
            continue
        value=r.get(key)
        if not value:
            continue
        try:
            item=json.loads(value)
        except (json.JSONDecodeError,TypeError):
            continue
        if not isinstance(item,dict) or not item.get('node') or not isinstance(item.get('gpu'),dict):
            continue
        workers.append(item)
    return workers
@app.post('/internal/heartbeat')
def heartbeat(data:dict,x_worker_key:str|None=Header(default=None)):
    agent(x_worker_key)
    node=data.get('node','')
    if not isinstance(node,str) or not SAFE.fullmatch(node): raise HTTPException(422,'Invalid node')
    now=time.time()
    data['last_seen']=now
    with r.lock('claim_lock',timeout=30,blocking_timeout=10):
        r.setex('node:'+node,45,json.dumps(data))
        claimed=set(r.smembers('running:'+node))
        actual=set(data.get('current_jobs') or [])
        for jid in claimed & actual:
            try:
                j=record(jid)
                if j['status']=='running' and j.get('node')==node:
                    j['last_heartbeat']=now;save(j)
            except HTTPException:pass
        cancel=[jid for jid in actual if r.get('cancel:'+jid)=='1']
    return {'ok':True,'cancel_jobs':cancel,'scheduling_paused':r.get('scheduler:paused')=='1'}
def live_node_candidates(job):
    """Called under claim_lock; never use un-reconciled or quarantined nodes."""
    eligible=[]
    for key in r.scan_iter('node:*'):
        node=key.split(':',1)[1]
        if ':' in node or r.exists('node:quarantine:'+node) or not r.exists('node:reconciled:'+node):
            continue
        raw=r.get(key)
        if not raw: continue
        try:
            info=json.loads(raw); gpu_info=info['gpu'];total=int(gpu_info['total_mib']);free=int(gpu_info['free_mib'])
            if total<=0: continue
            limit=node_limit(node,info.get('max_jobs',1))
            running=[record(i) for i in r.smembers('running:'+node)]
            running=[j for j in running if j.get('status')=='running' and j.get('node')==node]
            if len(running)>=limit:continue
            estimate=estimate_reservation(job,gpu_info.get('name',''))
            reserved=sum(int(j.get('reserved_vram_mib') or 0) for j in running)
            if running and (estimate is None or any(j.get('reserved_vram_mib') is None for j in running)):continue
            if estimate is not None and estimate>4096:continue
            if running and VRAM_BASELINE_MIB+reserved+estimate+VRAM_HEADROOM_MIB>total:continue
            if free<VRAM_HEADROOM_MIB+(estimate if estimate is not None else 512):continue
            if not running and estimate is not None and VRAM_BASELINE_MIB+estimate+VRAM_HEADROOM_MIB>total:continue
            # Prefer idle nodes, then lower occupancy / reserved fraction.
            score=(1 if running else 0, reserved/total, len(running)/limit, node)
            eligible.append((score,node,estimate))
        except (ValueError,TypeError,KeyError,HTTPException):continue
    return sorted(eligible)

@app.post('/internal/claim')
def claim(data:dict,x_worker_key:str|None=Header(default=None)):
    agent(x_worker_key)
    node=data.get('node','')
    if not isinstance(node,str) or not SAFE.fullmatch(node):raise HTTPException(422,'Invalid node')
    with r.lock('claim_lock',timeout=30,blocking_timeout=10):
        if r.get('scheduler:paused')=='1':return {'job':None,'reason':'paused'}
        if r.exists('node:quarantine:'+node):return {'job':None,'reason':'node_quarantined'}
        if not r.exists('node:reconciled:'+node):return {'job':None,'reason':'worker_reconciliation_required'}
        if not r.exists('node:'+node):return {'job':None,'reason':'node_offline'}
        for jid in list(reversed(r.lrange('pending',0,-1))):
            try:j=record(jid)
            except HTTPException:r.lrem('pending',1,jid);continue
            if j['status']!='queued':r.lrem('pending',1,jid);continue
            options=live_node_candidates(j)
            if not options or options[0][1]!=node:continue
            _,_,estimate=options[0]
            if r.lrem('pending',1,jid)!=1:continue
            j.update(status='running',node=node,reserved_vram_mib=estimate,
                     admission_policy='profiled_shared' if estimate is not None else 'exclusive_unprofiled',
                     updated_at=time.time(),last_heartbeat=time.time(),attempt=int(j.get('attempt',0))+1)
            save(j);r.sadd('running:'+node,jid)
            return {'job':j}
    return {'job':None,'reason':'waiting_for_capacity_or_preferred_node'}

# Uploaded M3 subsets reuse the SAME DATASETS_DIR and worker download route.
# No Redis dataset registry is needed: job.dataset selects a directory under DATASETS_DIR.
@app.post('/datasets/upload')
async def upload_candidate_dataset(file: UploadFile=File(...), student_id:str=Form(...),
                                   sha256:str=Form(...), x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    if not SAFE.fullmatch(student_id) or not re.fullmatch(r'[a-f0-9]{64}',sha256):
        raise HTTPException(422,'Invalid student ID or SHA-256')
    datasets.mkdir(parents=True,exist_ok=True)
    max_archive=int(os.getenv('MAX_DATASET_UPLOAD_MB','1000'))*1024*1024
    max_unpacked=int(os.getenv('MAX_DATASET_UNPACKED_MB','3000'))*1024*1024
    with tempfile.TemporaryDirectory(prefix='.incoming-',dir=datasets) as tmp:
        stage=Path(tmp); packed=stage/'dataset.zip'; digest=hashlib.sha256(); size=0
        with packed.open('wb') as target:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>max_archive:raise HTTPException(413,'Dataset ZIP too large')
                digest.update(chunk);target.write(chunk)
        if digest.hexdigest()!=sha256:raise HTTPException(422,'Dataset checksum mismatch')
        extracted=stage/'extracted';extracted.mkdir()
        try:
            with zipfile.ZipFile(packed) as archive:
                items=archive.infolist()
                if len(items)>200000:raise HTTPException(413,'Too many dataset entries')
                total=0
                for item in items:
                    name=item.filename
                    path=Path(name)
                    kind=(item.external_attr>>16)&0o170000
                    if (not name or name.startswith(('/','\\')) or '\\' in name or
                        path.is_absolute() or '..' in path.parts or ':' in name or
                        kind not in (0,stat.S_IFREG,stat.S_IFDIR)):
                        raise HTTPException(422,'Unsafe ZIP path or file type')
                    total+=item.file_size
                    if total>max_unpacked:raise HTTPException(413,'Unpacked dataset too large')
                for item in items:
                    destination=extracted/Path(item.filename)
                    if item.is_dir():destination.mkdir(parents=True,exist_ok=True);continue
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    with archive.open(item) as source,destination.open('wb') as sink:
                        shutil.copyfileobj(source,sink,1024*1024)
        except (zipfile.BadZipFile,EOFError,RuntimeError) as exc:
            raise HTTPException(422,'Invalid dataset ZIP') from exc
        manifest=extracted/'manifest.json'
        if not manifest.is_file() or not (extracted/'data.yaml').is_file():
            raise HTTPException(422,'Expected data.yaml and manifest.json at ZIP root')
        try:
            metadata=json.loads(manifest.read_text(encoding='utf-8'))
        except (OSError,ValueError) as exc:
            raise HTTPException(422,'Invalid Dataset Studio manifest') from exc
        selected=metadata.get('selected_order')
        if metadata.get('status')!='complete' or not isinstance(selected,list) or len(selected)!=4 or len(set(map(str,selected)))!=4:
            raise HTTPException(422,'Expected a complete four-class subset')
        for split in ('train','valid','val','test'):
            if split=='valid' and (extracted/'val').is_dir():continue
            if split=='val' and (extracted/'valid').is_dir():continue
            if split in ('train','test') and not (extracted/split).is_dir():
                raise HTTPException(422,f'Missing {split} split')
        dataset_id=f'subset-{student_id.lower()}-{sha256[:16]}'
        final=datasets/dataset_id
        # Content-addressed immutable dataset; do not replace an in-use dataset.
        with r.lock('dataset_upload:'+dataset_id,timeout=300,blocking_timeout=30):
            if final.exists():
                if not final.is_dir():raise HTTPException(409,'Dataset name already exists')
                return {'dataset_id':dataset_id,'sha256':sha256,'reused':True}
            extracted.rename(final)
        return {'dataset_id':dataset_id,'sha256':sha256,'reused':False,'bytes':size}

def locate_best_checkpoint(job):
    if not job.get('artifact'):raise HTTPException(409,'Source run has no artifacts')
    archive=(results/job['artifact']).resolve()
    if not archive.is_relative_to(results.resolve()) or not archive.is_file():
        raise HTTPException(404,'Source artifacts unavailable')
    try:
        with zipfile.ZipFile(archive) as z:
            matches=[item for item in z.infolist() if item.filename.endswith('/weights/best.pt') and not item.is_dir()]
            if len(matches)!=1 or matches[0].file_size>500*1024*1024:
                raise HTTPException(422,'Source run must contain exactly one best.pt checkpoint under weights/')
            return archive,matches[0].filename
    except (zipfile.BadZipFile,OSError) as exc:
        raise HTTPException(422,'Source artifact ZIP invalid') from exc

@app.get('/internal/checkpoint/{jobid}')
def worker_checkpoint(jobid:str,x_worker_key:str|None=Header(default=None)):
    agent(x_worker_key)
    origin=record(jobid)
    if origin.get('status')!='completed':raise HTTPException(409,'Source training not completed')
    archive,entry=locate_best_checkpoint(origin)
    def chunks():
        with zipfile.ZipFile(archive) as z, z.open(entry) as stream:
            while data:=stream.read(1024*1024):yield data
    return StreamingResponse(chunks(),media_type='application/octet-stream')

@app.get('/internal/dataset/{name}')
def dataset(name:str,x_worker_key:str|None=Header(default=None)):
    agent(x_worker_key)
    if not SAFE.fullmatch(name): raise HTTPException(400,'Invalid dataset name')
    directory=datasets/name
    if not directory.is_dir(): raise HTTPException(404,'Dataset not found on management server')
    if not (directory/'data.yaml').is_file():raise HTTPException(422,'Dataset must contain data.yaml')
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for f in directory.rglob('*'):
            if f.is_file() and not f.is_symlink():z.write(f,f.relative_to(directory))
    buf.seek(0)
    return StreamingResponse(buf,media_type='application/zip')
@app.post('/internal/finish/{jobid}')
def finish(jobid:str,data:dict,x_worker_key:str|None=Header(default=None)):
    agent(x_worker_key)
    with r.lock('claim_lock',timeout=30,blocking_timeout=10):
        j=record(jobid)
        if j['status']!='running' or j['node']!=data.get('node'):
            raise HTTPException(409,'Job owner or state mismatch')
        status=data.get('status')
        if status not in ('completed','failed','cancelled'):raise HTTPException(422,'Invalid status')
        if r.get('cancel:'+jobid)=='1':status='cancelled'
        j.update(status=status,error=str(data.get('error',''))[:2000],updated_at=time.time())
        save(j);r.srem('running:'+j['node'],jobid);r.delete('cancel:'+jobid)
    return j

@app.post('/internal/artifact/{jobid}')
async def artifact(jobid:str,upload:UploadFile=File(...),x_worker_key:str|None=Header(default=None),x_node:str|None=Header(default=None)):
    agent(x_worker_key);j=record(jobid)
    if j['status']!='running' or j['node']!=x_node:raise HTTPException(409,'Job ownership mismatch')
    target=results/j['student_id']/jobid;target.mkdir(parents=True,exist_ok=True)
    destination=target/'output.zip';size=0
    try:
        with destination.open('wb') as f:
            while chunk:=await upload.read(1024*1024):
                size+=len(chunk)
                if size>int(os.environ.get('MAX_UPLOAD_MB','1000'))*1024*1024:raise HTTPException(413,'Artifact too large')
                f.write(chunk)
        if not zipfile.is_zipfile(destination):raise HTTPException(422,'Invalid ZIP archive')
        j['artifact']=str(destination.relative_to(results));save(j)
        return {'saved':True,'bytes':size}
    except Exception:
        destination.unlink(missing_ok=True);raise
# Live training log transport: workers append bounded UTF-8 chunks; clients
# request only bytes not yet seen. Logs are per-job and never shared by path.
@app.post('/internal/log/{jobid}')
def append_job_log(jobid:str, data:dict, x_worker_key:str|None=Header(default=None)):
    agent(x_worker_key)
    j=record(jobid)
    if j.get('node') != data.get('node') or j.get('status') not in ('running','completed','failed','cancelled'):
        raise HTTPException(409,'Job owner mismatch')
    chunk=data.get('text')
    if not isinstance(chunk,str) or len(chunk.encode('utf-8'))>65536:
        raise HTTPException(413,'Log chunk too large')
    offset=data.get('offset')
    if not isinstance(offset,int) or offset<0: raise HTTPException(422,'Invalid log offset')
    logpath=results/j['student_id']/jobid/'live.log'
    logpath.parent.mkdir(parents=True,exist_ok=True)
    with r.lock('log:'+jobid,timeout=20,blocking_timeout=10):
        current=logpath.stat().st_size if logpath.exists() else 0
        if offset!=current:
            return {'offset':current,'resync':True}
        payload=chunk.encode('utf-8')
        with logpath.open('ab') as stream:stream.write(payload)
        return {'offset':current+len(payload),'resync':False}

@app.get('/jobs/{jobid}/log')
def get_job_log(jobid:str, offset:int=0, limit:int=32768, x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    j=record(jobid)
    if offset<0 or limit<1 or limit>65536:raise HTTPException(422,'Invalid log window')
    path=results/j['student_id']/jobid/'live.log'
    if not path.is_file():return {'text':'','offset':0,'more':False}
    size=path.stat().st_size
    start=min(offset,size)
    with path.open('rb') as stream:
        stream.seek(start)
        data=stream.read(limit)
    # Use latin-1 transport to preserve byte offsets on UTF-8 boundaries;
    # UI decodes via replacement where needed.
    return {'text':data.decode('utf-8',errors='replace'),'offset':start+len(data),'more':start+len(data)<size}

@app.get('/jobs/{jobid}/artifact')
def get_artifact(jobid:str,x_api_key:str|None=Header(default=None)):
    user(x_api_key);j=record(jobid)
    if not j.get('artifact'):
        raise HTTPException(404,'No artifact has been uploaded for this job')
    # Reject an inconsistent stored path and report missing result files
    # as a useful 404 rather than letting FileResponse raise HTTP 500.
    relative=Path(j['artifact'])
    expected=Path(j['student_id'])/jobid/'output.zip'
    if relative!=expected:
        raise HTTPException(409,'Artifact metadata does not match the job')
    destination=results/relative
    if not destination.is_file():
        raise HTTPException(
            404,'Job artifact file is missing from manager storage. '
                'Check the results volume or whether competition data was reset.'
        )
    return FileResponse(destination,filename=jobid+'.zip',media_type='application/zip')

@app.get('/scheduler/reservations')
def reservations(x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    out={}
    for key in r.scan_iter('running:*'):
        node=key.split(':',1)[1]
        out[node]=[record(jid) for jid in r.smembers(key) if r.exists('job:'+jid)]
    return out

# Administrative queue controls. All mutating operations share claim_lock with claim/finish.
@app.get('/scheduler/status')
def scheduler_status(x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    return {'paused':r.get('scheduler:paused')=='1','pending':r.llen('pending'),
            'stale_threshold_seconds':STALE_SECONDS}

@app.post('/scheduler/pause')
def pause_scheduler(x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    with r.lock('claim_lock',timeout=30,blocking_timeout=10):r.set('scheduler:paused','1')
    return {'paused':True,'note':'Already running jobs continue'}

@app.post('/scheduler/resume')
def resume_scheduler(x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    with r.lock('claim_lock',timeout=30,blocking_timeout=10):r.delete('scheduler:paused')
    return {'paused':False}

@app.post('/jobs/{jobid}/cancel')
def cancel_job(jobid:str,x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    with r.lock('claim_lock',timeout=30,blocking_timeout=10):
        j=record(jobid)
        if j['status']=='queued':
            r.lrem('pending',0,jobid)
            j.update(status='cancelled',error='Cancelled by administrator',updated_at=time.time());save(j)
            return {'status':'cancelled','job_id':jobid}
        if j['status']=='running':
            r.set('cancel:'+jobid,'1')
            j['cancel_requested_at']=time.time();save(j)
            return {'status':'cancellation_requested','job_id':jobid,
                    'note':'Worker receives request on next heartbeat; job stays reserved until acknowledged'}
        return {'status':j['status'],'job_id':jobid,'note':'Terminal jobs unchanged'}

@app.get('/scheduler/stale')
def stale_jobs(x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    now=time.time();out=[]
    for key in r.scan_iter('running:*'):
        node=key.split(':',1)[1]
        for jid in r.smembers(key):
            try:j=record(jid)
            except HTTPException:continue
            if j.get('status')!='running':continue
            elapsed=now-float(j.get('last_heartbeat',j.get('updated_at',now)))
            if elapsed>=STALE_SECONDS:
                out.append({'id':jid,'node':node,'seconds_since_heartbeat':round(elapsed),
                            'node_online':r.exists('node:'+node)>0})
    return out

@app.post('/scheduler/recover')
def recover_stale(x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    return recover_stale_jobs()

def recover_stale_jobs():
    """Quarantine on heartbeat loss; never release capacity without host reconciliation."""
    now=time.time(); quarantined=[]
    with r.lock('claim_lock',timeout=30,blocking_timeout=10):
        for key in r.scan_iter('running:*'):
            node=key.split(':',1)[1]
            if r.exists('node:'+node): continue
            for jid in list(r.smembers(key)):
                try:j=record(jid)
                except HTTPException: continue
                if j.get('status')!='running' or j.get('node')!=node: continue
                if now-float(j.get('last_heartbeat',j.get('updated_at',now)))<STALE_SECONDS: continue
                r.set('node:quarantine:'+node,'heartbeat_expired')
                r.delete('node:reconciled:'+node)
                quarantined.append({'node':node,'job_id':jid})
    return {'quarantined':quarantined,'note':'No reservations released until host-side reconciliation'}

@app.post('/internal/reconcile')
def reconcile_node(data:dict,x_worker_key:str|None=Header(default=None)):
    """Worker confirms local Docker inventory is empty before releasing stale records."""
    agent(x_worker_key)
    node=data.get('node','')
    if not isinstance(node,str) or not SAFE.fullmatch(node): raise HTTPException(422,'Invalid node')
    if data.get('live_containers') != []: raise HTTPException(409,'Training containers still exist; stop or adopt them first')
    with r.lock('claim_lock',timeout=30,blocking_timeout=10):
        previous=[]
        for jid in list(r.smembers('running:'+node)):
            try:j=record(jid)
            except HTTPException:
                r.srem('running:'+node,jid); continue
            if j.get('status')=='running' and j.get('node')==node:
                j.update(status='failed',error='Node reconciled: prior execution ended without finish acknowledgement',updated_at=time.time())
                save(j);previous.append(jid)
            r.srem('running:'+node,jid);r.delete('cancel:'+jid)
        r.delete('node:quarantine:'+node)
        r.set('node:reconciled:'+node,str(time.time()))
    return {'ok':True,'abandoned_jobs_failed':previous}

@app.get('/scheduler/quarantined')
def quarantined_nodes(x_api_key:str|None=Header(default=None)):
    user(x_api_key)
    return [{'node':k.split(':')[-1],'reason':r.get(k)} for k in r.scan_iter('node:quarantine:*')]

def recovery_loop():
    while True:
        time.sleep(RECOVERY_INTERVAL)
        try:recover_stale_jobs()
        except Exception as e:print('Recovery watchdog:',repr(e),flush=True)

@app.on_event('startup')
def start_recovery_watchdog():
    threading.Thread(target=recovery_loop,daemon=True,name='recovery-watchdog').start()

@app.post('/scheduler/recover/{jobid}')
def recover_specific(jobid:str,x_api_key:str|None=Header(default=None)):
    """Explicit manual recovery when a restarted node is online but no longer owns the job."""
    user(x_api_key)
    with r.lock('claim_lock',timeout=30,blocking_timeout=10):
        if r.get('scheduler:paused')!='1':
            raise HTTPException(409,'Pause scheduling before manual recovery')
        j=record(jobid)
        if j.get('status')!='running':raise HTTPException(409,'Job is not running')
        node=j.get('node')
        hb=r.get('node:'+node)
        if hb:
            info=json.loads(hb)
            if jobid in (info.get('current_jobs') or []):
                raise HTTPException(409,'Worker still reports this job as active')
        age=time.time()-float(j.get('last_heartbeat',j.get('updated_at',time.time())))
        if age<STALE_SECONDS:
            raise HTTPException(409,f'Last job heartbeat {age:.0f}s ago; wait {STALE_SECONDS}s')
        j.update(status='failed',error='Manual recovery after heartbeat expired; verify GPU host container stopped before resubmitting',updated_at=time.time())
        save(j);r.srem('running:'+node,jobid);r.delete('cancel:'+jobid)
        r.set('node:quarantine:'+node,'manual_recovery_requires_reconciliation');r.delete('node:reconciled:'+node)
    return {'recovered_as_failed':jobid,'node':node,'warning':'Recovery does not terminate Docker containers'}
