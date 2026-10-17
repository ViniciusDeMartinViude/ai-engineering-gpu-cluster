# Start a GPU worker automatically (systemd)

Do this **on each Linux or WSL2 GPU worker**, after `worker.env` and venv are configured and training works manually. Use an absolute install path and the actual account name. Example:

```ini
# /etc/systemd/system/ai-gpu-worker.service
[Unit]
Description=AI Engineering GPU Worker
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
User=YOUR_LINUX_USER
WorkingDirectory=/home/YOUR_LINUX_USER/ai-engineering-gpu-cluster
ExecStart=/bin/bash /home/YOUR_LINUX_USER/ai-engineering-gpu-cluster/scripts/start-worker.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ai-gpu-worker.service
sudo journalctl -u ai-gpu-worker.service -f
```

For WSL2, systemd service startup depends on WSL distribution launching. Configure Windows startup of WSL separately if needed. Do not run a second manual worker agent under the same NODE_NAME concurrently.
