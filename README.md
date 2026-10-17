# AI Engineering GPU Cluster — deployment kit

Install a **management computer** (FastAPI + Redis, GPU not required) and any number of **GPU worker computers** (Linux or Ubuntu under Windows WSL2 + Docker with NVIDIA GPU access). Prepared from the project's v0.9 chained-training patch, retaining v0.8 load balancing, dataset uploads, image sizes from 160 to 640, and worker-side checkpoint retrieval.

> **Operator-only deployment.** The shared `API_KEY` is an administrator credential, not a student login. Don't expose port 8000 publicly or share API_KEY with untrusted student machines. Run over a trusted LAN/VPN with host firewall restrictions. Worker authentication uses a separate `WORKER_KEY`.

## Requirements

- Management: Ubuntu 22.04/24.04 (native or WSL2), Docker Engine with Compose plugin **or** Docker Desktop WSL integration; Python 3 for install helper.
- GPU workers: Ubuntu 22.04/24.04, NVIDIA driver with GPU visible via `nvidia-smi`, Docker with NVIDIA GPU runtime/toolkit and permissions for `docker run --gpus all`; Python 3, venv, ~20GB+ free disk.
- Manager/worker machines can reach manager TCP port 8000. Only management machine needs inbound TCP/8000.

## Manager (computer without GPU is fine)

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/ai-engineering-gpu-cluster.git
cd ai-engineering-gpu-cluster
bash scripts/management-setup.sh
curl http://localhost:8000/health
```

`.env` is generated with random API_KEY and WORKER_KEY. Keep a backup; don't commit it. Manager's `state/datasets` is **writable** for the upload API and persistent across updates, along with `state/results` and the Docker Redis volume. Set `GPU_MAX_CONCURRENT_JOBS=3` in `.env` to allow the manager to admit up to 3 concurrent jobs per worker; concurrency is ultimately bounded by each worker's limit and available VRAM.

Copy `WORKER_KEY` (not API_KEY) from the manager `.env` to each worker's `worker.env`. Replace `MANAGER_URL` with the LAN address of the manager.

## GPU worker (each additional computer)

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/ai-engineering-gpu-cluster.git
cd ai-engineering-gpu-cluster
cp worker.env.example worker.env
nano worker.env   # set MANAGER_URL, WORKER_KEY, NODE_NAME, GPU_MAX_CONCURRENT_JOBS
bash scripts/worker-setup.sh
bash scripts/validate-worker.sh
bash scripts/start-worker.sh
```

Set `GPU_MAX_CONCURRENT_JOBS=3` for the RTX A4000 (16GB) and `=2` for RTX 5060 (8GB) as the planned pilot values; **sharing VRAM is not hard-isolated** and these limits still require workload testing. A new GPU type defaults to 1 concurrent job and unknown memory profiles are scheduled exclusively.

The worker process must remain running. See [Autostart](docs/AUTOSTART.md) for the systemd service template. Workers do **not** require the administrator API_KEY.

## Validate manager and training

```bash
bash scripts/validate-manager.sh
# Create/upload a four-class subset in Assessment Studio Remote Training and run a short 3-epoch job.
```

Manager `/health`, `/jobs`, `/nodes`, `/internal/dataset/{name}`, `/datasets/upload`, and `/internal/checkpoint/{jobid}` are included. Training input `source_job_id` enables chained training from a completed earlier job's best.pt. The manager keeps the same job/dataset endpoints used by Assessment Studio PR #15.

## Updating existing installations

Back up `.env`, `worker.env`, `state/`, and Redis data before changing versions. On management host: `git pull && docker compose -f docker-compose.management.yml up -d --build --no-deps manager`. On each worker: stop the running worker; `git pull && bash scripts/worker-setup.sh`, then start it again. Keep active jobs running until completion before restarting.

## Troubleshooting

- `Read-only file system /datasets`: ensure Compose maps `./state/datasets:/datasets:rw` and recreate manager.
- `404 Dataset not found`: verify uploaded subset exists in manager `state/datasets/<dataset_id>/data.yaml`.
- Job queued indefinitely: run manager validation; verify worker heartbeat, GPU runtime, worker key, VRAM reservations and worker reconciliation.
- Chained run fails: source job must be completed, same candidate and model architecture, and artifact must contain exactly one `weights/best.pt`.
- Docker permission errors: add account to docker group, log out/in, or use supported rootless configuration.

**Known limitation:** These packages have not been validated against your live computers. Test with a short disposable job before using for a competition. This project carries no privileged student authentication layer.
