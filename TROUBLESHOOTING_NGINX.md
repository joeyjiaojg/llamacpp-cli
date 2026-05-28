# Troubleshooting: nginx on Port 8080

## Problem

When accessing `http://localhost:8080/backends`, you get:
```html
<html>
<head><title>404 Not Found</title></head>
<body>
<center><h1>404 Not Found</h1></center>
<hr><center>nginx/1.24.0 (Ubuntu)</center>
</body>
</html>
```

## Root Cause

nginx is already running on port 8080 on your host machine. When the proxy container uses `network_mode: host`, it tries to bind to port 8080 but nginx is already there.

## Solutions

### Option 1: Use Different Port (Easiest)

Run the proxy on a different port:

```bash
make start-proxy SUBNET=10.231.0.0/16 PROXY_PORT=8081
```

Then access at:
```bash
curl http://localhost:8081/backends
```

### Option 2: Stop nginx

If you don't need nginx:

```bash
sudo systemctl stop nginx
make start-proxy SUBNET=10.231.0.0/16
```

### Option 3: Change nginx Port

Edit nginx config to use a different port:

```bash
sudo nano /etc/nginx/sites-enabled/default
# Change: listen 8080; → listen 8090;
sudo nginx -t
sudo systemctl reload nginx
make start-proxy SUBNET=10.231.0.0/16
```

### Option 4: Use Bridge Network Instead of Host

Edit `docker-compose.proxy.yml`:

```yaml
services:
  lb-proxy:
    # Remove this line:
    # network_mode: host

    # Add port mapping instead:
    ports:
      - "8080:8080"

    # Add to same network as backends:
    networks:
      - llamacpp-net

networks:
  llamacpp-net:
    external: true
    name: llamacpp-cli_default
```

**Caveat**: Subnet discovery won't work with bridge network. You'll need to manually specify backends.

## How to Check What's on Port 8080

```bash
# Check with netstat
sudo netstat -tlnp | grep 8080

# Check with lsof
sudo lsof -nP -iTCP:8080 -sTCP:LISTEN

# Check with ss
ss -tlnp | grep 8080

# Check nginx config
sudo nginx -T | grep listen
```

## Recommended Solution for Production

Use Option 1 (different port) or Option 3 (change nginx port):

**For proxy:**
```bash
# Use port 8081 for proxy
make start-proxy SUBNET=10.231.0.0/16 PROXY_PORT=8081
```

**For clients:**
```bash
# Point to port 8081
export OPENAI_API_BASE=http://proxy-ip:8081/v1
```

Or configure nginx as a reverse proxy to the llamacpp lb-proxy:

```nginx
# /etc/nginx/sites-enabled/llamacpp
server {
    listen 8080;
    location / {
        proxy_pass http://127.0.0.1:8081;  # Forward to lb-proxy on 8081
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Then:
```bash
# lb-proxy on 8081
make start-proxy SUBNET=10.231.0.0/16 PROXY_PORT=8081

# nginx on 8080 forwards to 8081
# Clients still use port 8080
```

## Verification

After applying a solution:

```bash
# Check lb-proxy is running
docker ps | grep lb-proxy

# Check what's listening on ports
sudo netstat -tlnp | grep -E "8080|8081"

# Test lb-proxy directly
curl http://localhost:8081/backends  # or whatever port you used

# Expected output:
# {"backends": [...]}
```
