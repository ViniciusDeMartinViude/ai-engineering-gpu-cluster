import io, json, os, re, shutil, socket, subprocess, tempfile, threading, time, traceback, zipfile
from pathlib import Path
import requests

BASE=os.environ.get('MANAGER_URL','http://127.0.0.1:8000').rstrip('/')
KEY=os.environ['WORKER_KEY'];NODE=os.environ.get('NODE_NAME',socket.gethostname().lower().replace('_','-'))
HEAD={'x-worker-key':KEY}
IMAGE=os.environ.get('TRAINER_IMAGE','ai-gpu-trainer:0.2')
DATA=Path(os.environ.get('DATA_CACHE',str(Path.home()/'ai-gpu-data/datasets'))).resolve()
OUT=Path(os.environ.get('OUTPUT_ROOT',str(Path.home()/'ai-gpu-data/results'))).resolve()
DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
active_jobs=set()
stop_requests={}  # job_id -> cancellation Event
active_specs={}
overlapped_jobs=set()  # Job IDs that have ever overlapped another active job.
active_lock=threading.RLock()
dataset_lock=threading.Lock()
DUAL_B01=os.environ.get("ENABLE_DUAL_B01","false").lower() in ("1","true","yes")
MAX_JOBS=max(1,min(8,int(os.environ.get('GPU_MAX_CONCURRENT_JOBS','1'))))
EXPERIMENT_LEVEL=max(0,min(2,int(os.environ.get('GPU_EXPERIMENT_LEVEL','0'))))
def compatible_pair(a,b):
    # The manager decides capacity under the Redis claim lock.
    # Fail closed for unprofiled work or stale/old manager replies.
    return (a.get('reserved_vram_mib') is not None and
            b.get('reserved_vram_mib') is not None and
            a.get('admission_policy')=='profiled_shared' and
            b.get('admission_policy')=='profiled_shared')

def gpu():
    try:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=8)
        parts=[v.strip() for v in out.splitlines()[0].split(',')]
        return {'name':parts[0],'total_mib':int(parts[1]),'free_mib':int(parts[2]),'utilization_pct':int(parts[3])}
    except Exception as e:return {'error':str(e)[:150]}

def post(endpoint,data,timeout=15):
    rr=requests.post(BASE+endpoint,json=data,headers=HEAD,timeout=timeout);rr.raise_for_status();return rr.json()

def heartbeat():
    while True:
        try:
            with active_lock: current=sorted(active_jobs)
            response=post('/internal/heartbeat',{'node':NODE,'current_job':current[0] if current else None,'current_jobs':current,'gpu':gpu(),'max_jobs':MAX_JOBS,'status':'busy' if current else 'idle'})
            with active_lock:
                for jid in response.get('cancel_jobs',[]):
                    event=stop_requests.get(jid)
                    if event:event.set()
        except Exception as e:print('Heartbeat:',e,flush=True)
        time.sleep(10)

def ensure_dataset(name):
    dest=DATA/name
    if (dest/'data.yaml').exists():return dest
    print('Downloading dataset:',name,flush=True)
    rr=requests.get(BASE+'/internal/dataset/'+name,headers=HEAD,timeout=600);rr.raise_for_status()
    dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(rr.content)) as z:
        for item in z.infolist():
            p=(dest/item.filename).resolve()
            if not p.is_relative_to(dest):raise ValueError('Unsafe ZIP member')
            if item.is_dir():p.mkdir(parents=True,exist_ok=True)
            else:
                p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(item) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst)
    if not (dest/'data.yaml').exists():raise RuntimeError('Dataset missing data.yaml')
    return dest

def gpu_used_mib():
    try:
        val=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used',
            '--format=csv,noheader,nounits'],text=True,timeout=8)
        return int(val.strip().splitlines()[0])
    except Exception:
        return None

def forward_live_log(job, logfile_path, finished):
    """Best-effort incremental log forwarding; never fail the training job."""
    read_pos=0
    remote_pos=0
    while True:
        try:
            if logfile_path.is_file():
                with logfile_path.open('rb') as source:
                    source.seek(read_pos)
                    payload=source.read(16000)
                if payload:
                    # Most trainer output is ASCII. Read complete UTF-8 codepoints
                    # so offsets refer to actual bytes stored on the manager.
                    try:
                        message=payload.decode('utf-8')
                    except UnicodeDecodeError as error:
                        if error.reason=='unexpected end of data' and error.start>0:
                            payload=payload[:error.start]
                            message=payload.decode('utf-8',errors='replace')
                        else:
                            message=payload.decode('utf-8',errors='replace')
                    response=post('/internal/log/'+job['id'],
                                  {'node':NODE,'offset':remote_pos,'text':message},timeout=20)
                    if response.get('resync'):
                        # Restart at beginning only if manager lost the log.
                        expected=int(response['offset'])
                        if expected==0:
                            read_pos=remote_pos=0
                        else:
                            remote_pos=expected
                        continue
                    read_pos+=len(payload)
                    remote_pos=int(response['offset'])
                    continue
        except Exception as exc:
            print('Log forward warning:',str(exc)[:160],flush=True)
        if finished.is_set():
            # Avoid an endless retry loop on a failed manager.
            break
        finished.wait(2)


def launch(job):
    folder=OUT/job['student_id']/job['id'];folder.mkdir(parents=True,exist_ok=True)
    # Give each job its own unique container name and shared memory.
    container_name='ai-train-'+job['id']
    cmd=['docker','run','--name',container_name,'--label','ai-engineering.managed=true','--label','ai-engineering.node='+NODE,'--label','ai-engineering.job='+job['id'],'--rm','--gpus','all','--shm-size','1g',
         '--user',f'{os.getuid()}:{os.getgid()}','-e','HOME=/tmp',
         '--cpus','4','--memory','10g','--pids-limit','512',
         '--security-opt','no-new-privileges','--cap-drop','ALL','--network','none',
         '-v',str(folder)+':/output','-e','PYTHONUNBUFFERED=1']
    if job['mode']=='train':
        dataset=ensure_dataset(job['dataset'])
        cmd+=['-v',str(dataset)+':/dataset:ro']
    if job.get('source_job_id'):
        checkpoint=folder/'source-best.pt'
        # The worker uses its existing private WORKER_KEY, never a candidate API key.
        rr=requests.get(BASE+'/internal/checkpoint/'+job['source_job_id'],headers=HEAD,stream=True,timeout=600)
        rr.raise_for_status()
        try:
            with checkpoint.open('wb') as handle:
                for chunk in rr.iter_content(1024*1024):
                    if chunk:handle.write(chunk)
        finally:rr.close()
        if checkpoint.stat().st_size==0:raise RuntimeError('Empty source checkpoint')
        cmd+=['-v',str(checkpoint)+':/checkpoint/best.pt:ro']
    cmd += [IMAGE,'--mode',job['mode'],'--model','/checkpoint/best.pt' if job.get('source_job_id') else job['model'],'--epochs',str(job['epochs']),
            '--imgsz',str(job['imgsz']),'--batch',str(job['batch'])]
    print('Executing:',job['id'],job['mode'],flush=True)
    baseline=gpu_used_mib();samples=[];stop=threading.Event()
    def sampler():
        while not stop.is_set():
            x=gpu_used_mib()
            if x is not None:samples.append(x)
            stop.wait(2)
    t=threading.Thread(target=sampler,daemon=True);t.start()
    rc=1
    cancelled=False
    log_path=folder/'container.log'
    log_done=threading.Event()
    log_forwarder=threading.Thread(
        target=forward_live_log,args=(job,log_path,log_done),
        daemon=True,name='log-'+job['id'][:8])
    log_forwarder.start()
    try:
        with log_path.open('w',encoding='utf-8') as logfile:
            try:
                proc=subprocess.Popen(cmd,stdout=logfile,stderr=subprocess.STDOUT)
                started=time.monotonic()
                while proc.poll() is None:
                    with active_lock: cancel_event=stop_requests.get(job['id'])
                    if cancel_event and cancel_event.is_set():
                        cancelled=True
                        subprocess.run(['docker','rm','-f',container_name],capture_output=True,timeout=30)
                        break
                    if time.monotonic()-started>4*60*60:
                        logfile.write('\nERROR: Training exceeded four-hour limit\n')
                        subprocess.run(['docker','rm','-f',container_name],capture_output=True,timeout=30)
                        break
                    time.sleep(1)
                try:rc=proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill();rc=124
                if cancelled:rc=130
            except Exception:
                subprocess.run(['docker','rm','-f',container_name],capture_output=True,timeout=30)
                raise
    finally:
        stop.set();t.join(timeout=12)
        log_done.set()
        log_forwarder.join(timeout=10)
    peak=max(samples) if samples else None
    metrics={'node':NODE,'device':gpu().get('name','unknown'),
             'baseline_gpu_used_mib':baseline,'peak_gpu_used_mib':peak,
             'peak_delta_gpu_used_mib':max(0,peak-baseline) if peak is not None and baseline is not None else None,
             'sample_seconds':2,
             'measurement_notes':'Whole GPU sampled; under concurrency includes other jobs. Per-job peak_delta is NOT independently attributable.'}
    log=(folder/'container.log').read_text(encoding='utf-8',errors='replace')
    for label in ('peak_allocated_gib','peak_reserved_gib'):
        matches=re.findall(r'(?m)^'+label+r'\s+([0-9]+(?:\.[0-9]+)?)',log)
        if matches:metrics[label]=float(matches[-1])
    with active_lock:
        concurrent = job['id'] in overlapped_jobs or len(active_jobs) > 1
    metrics['concurrent_observation']=concurrent
    if concurrent:
        metrics['peak_delta_gpu_used_mib']=None
        metrics['measurement_notes']+=' Not usable as per-job VRAM benchmark.'
    (folder/'memory_metrics.json').write_text(json.dumps(metrics,indent=2))
    try:post('/internal/metrics/'+job['id'],metrics)
    except Exception as e:print('Metrics upload warning:',e,flush=True)
    with zipfile.ZipFile(folder/'artifacts.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for f in folder.rglob('*'):
            if f.is_file() and f.name not in ('artifacts.zip','source-best.pt'):z.write(f,f.relative_to(folder))
    with (folder/'artifacts.zip').open('rb') as f:
        rr=requests.post(BASE+'/internal/artifact/'+job['id'],headers={**HEAD,'x-node':NODE},
          files={'upload':('output.zip',f,'application/zip')},timeout=1200)
        rr.raise_for_status()
    return rc

def execute(j):
    jid=j['id']
    try:
        rc=launch(j)
        post('/internal/finish/'+jid,{'node':NODE,
            'status':'cancelled' if rc==130 else ('completed' if rc==0 else 'failed'),
            'error':f'Exit code {rc}' if rc else ''})
        print('Job',jid,'exit',rc,flush=True)
    except Exception as e:
        print(traceback.format_exc(),flush=True)
        try:post('/internal/finish/'+jid,{'node':NODE,'status':'failed','error':str(e)})
        except Exception as ex:print('Unable to report failure:',ex,flush=True)
    finally:
        with active_lock:
            active_jobs.discard(jid)
            active_specs.pop(jid,None)
            overlapped_jobs.discard(jid)
            stop_requests.pop(jid,None)

def local_training_containers():
    """Include both labeled v0.7 and legacy ai-train-* containers from v0.6."""
    output=subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True,timeout=20)
    return sorted(x.strip() for x in output.splitlines() if x.strip().startswith('ai-train-'))

def reconcile_startup():
    # Never trust Redis to prove a GPU process has stopped. Inspect Docker locally.
    while True:
        try:
            containers=local_training_containers()
            if containers:
                print('SAFETY BLOCK: existing training containers:',containers,flush=True)
                print('Inspect them with docker ps -a; stop ONLY confirmed orphan jobs.',flush=True)
                time.sleep(10);continue
            result=post('/internal/reconcile',{'node':NODE,'live_containers':[]})
            print('Startup reconciliation:',result,flush=True)
            return
        except Exception as e:
            print('Reconciliation unavailable:',e,flush=True)
            time.sleep(10)

def main():
    print('Worker',NODE,'management',BASE,'GPU',gpu(), 'max_jobs',MAX_JOBS,'memory_aware',True,flush=True)
    # Run ONE instance of this agent on each GPU computer.
    reconcile_startup()
    threading.Thread(target=heartbeat,daemon=True).start()
    threads={}
    while True:
        try:
            threads={k:t for k,t in threads.items() if t.is_alive()}
            with active_lock: count=len(active_jobs)
            if count>=MAX_JOBS:
                time.sleep(2);continue
            j=post('/internal/claim',{'node':NODE,'max_jobs':MAX_JOBS})['job']
            if not j:
                time.sleep(3);continue
            with active_lock:
                compatible=not active_specs or all(compatible_pair(spec,j) for spec in active_specs.values())
            if not compatible:
                # The server must never assign an unprofiled concurrent job.
                print('SAFETY ERROR: incompatible concurrent claim',j['id'],flush=True)
                try:post('/internal/finish/'+j['id'],{'node':NODE,'status':'failed','error':'Unsafe concurrent claim'})
                except Exception:pass
                continue
            with active_lock:
                if active_jobs:
                    overlapped_jobs.update(active_jobs)
                    overlapped_jobs.add(j['id'])
                active_jobs.add(j['id'])
                stop_requests[j['id']]=threading.Event()
                active_specs[j['id']]=j
            t=threading.Thread(target=execute,args=(j,),daemon=False,name='job-'+j['id'][:8]);threads[j['id']]=t;t.start()
        except Exception as e:
            print('Manager connectivity:',e,flush=True);time.sleep(5)

if __name__=='__main__':main()
