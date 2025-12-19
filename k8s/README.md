# TinyLLM Kubernetes Deployment with Service Mesh

This directory contains Kubernetes manifests for deploying TinyLLM with service mesh integration (Istio or Linkerd).

## Files

- `deployment.yaml` - Main deployment, service, and namespace with service mesh annotations
- `istio-virtualservice.yaml` - Istio-specific traffic management and security policies
- `linkerd-servicemesh.yaml` - Linkerd-specific routing and policy configuration
- `configmap.yaml` - Configuration for TinyLLM
- `monitoring.yaml` - ServiceMonitor for Prometheus metrics

## Prerequisites

### For Istio

```bash
# Install Istio
istioctl install --set profile=default -y

# Enable namespace for automatic sidecar injection
kubectl label namespace tinyllm istio-injection=enabled
```

### For Linkerd

```bash
# Install Linkerd
linkerd install | kubectl apply -f -
linkerd check

# Enable namespace for automatic proxy injection
kubectl annotate namespace tinyllm linkerd.io/inject=enabled
```

## Deployment

### 1. Create Namespace

```bash
kubectl apply -f deployment.yaml  # This creates the namespace first
```

### 2. Deploy Application

```bash
# Apply all manifests
kubectl apply -f deployment.yaml
kubectl apply -f configmap.yaml
kubectl apply -f monitoring.yaml
```

### 3. Deploy Service Mesh Configuration

#### For Istio:

```bash
kubectl apply -f istio-virtualservice.yaml
```

#### For Linkerd:

```bash
kubectl apply -f linkerd-servicemesh.yaml
```

## Features

### Service Mesh Capabilities

#### Istio Integration

- **Traffic Management**
  - VirtualService for routing and retries
  - DestinationRule for circuit breaking and load balancing
  - Gateway for external access with TLS

- **Security**
  - Strict mTLS between services
  - PeerAuthentication for authentication
  - AuthorizationPolicy for fine-grained access control

- **Observability**
  - Distributed tracing with Zipkin
  - Custom metrics and tags
  - Request telemetry

#### Linkerd Integration

- **Traffic Management**
  - HTTPRoute for path-based routing
  - TrafficSplit for canary deployments
  - ServiceProfile for per-route metrics and retries

- **Security**
  - Automatic mTLS with identity
  - Server and ServerAuthorization for access control
  - NetworkAuthentication for network-level security

- **Observability**
  - Golden metrics (success rate, latency, throughput)
  - Per-route metrics
  - Tap for real-time request inspection

### Health Checks

The deployment includes three types of health checks:

- **Liveness Probe** (`/health/live`) - Restarts pod if unhealthy
- **Readiness Probe** (`/health/ready`) - Removes from service if not ready
- **Startup Probe** (`/health/startup`) - Gives app time to start

### Resource Management

- CPU requests: 500m, limits: 2000m
- Memory requests: 512Mi, limits: 2Gi
- Sidecar proxy resources configured separately

## Monitoring

### With Istio

```bash
# Access Kiali dashboard
istioctl dashboard kiali

# View service graph
# Navigate to Graph -> Select tinyllm namespace

# View metrics
istioctl dashboard prometheus

# View traces
istioctl dashboard jaeger
```

### With Linkerd

```bash
# Access Linkerd dashboard
linkerd viz dashboard

# View service metrics
linkerd viz stat deploy/tinyllm -n tinyllm

# View live traffic
linkerd viz tap deploy/tinyllm -n tinyllm

# View service routes
linkerd viz routes deploy/tinyllm -n tinyllm
```

## Traffic Management

### Canary Deployment (Istio)

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: tinyllm-canary
spec:
  hosts:
  - tinyllm
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: tinyllm
        subset: canary
  - route:
    - destination:
        host: tinyllm
        subset: stable
      weight: 90
    - destination:
        host: tinyllm
        subset: canary
      weight: 10
```

### Canary Deployment (Linkerd)

Already configured in `linkerd-servicemesh.yaml` as TrafficSplit with 90/10 split.

## Security

### mTLS

Both Istio and Linkerd provide automatic mTLS:

- **Istio**: Configured via PeerAuthentication in STRICT mode
- **Linkerd**: Automatic mTLS for all meshed pods

### Authorization

- **Istio**: AuthorizationPolicy defines allowed operations
- **Linkerd**: ServerAuthorization controls access to services

## Troubleshooting

### Check Sidecar Injection

```bash
# For Istio
kubectl get pods -n tinyllm -o jsonpath='{.items[*].spec.containers[*].name}'
# Should show: tinyllm istio-proxy

# For Linkerd
kubectl get pods -n tinyllm -o jsonpath='{.items[*].spec.containers[*].name}'
# Should show: tinyllm linkerd-proxy
```

### View Sidecar Logs

```bash
# Istio
kubectl logs -n tinyllm <pod-name> -c istio-proxy

# Linkerd
kubectl logs -n tinyllm <pod-name> -c linkerd-proxy
```

### Check mTLS Status

```bash
# Istio
istioctl authn tls-check <pod-name>.tinyllm

# Linkerd
linkerd viz edges deployment/tinyllm -n tinyllm
```

### Debug Traffic

```bash
# Istio - enable access logs
kubectl edit configmap istio -n istio-system
# Set: accessLogFile: /dev/stdout

# Linkerd - tap live traffic
linkerd viz tap deploy/tinyllm -n tinyllm
```

## Performance Tuning

### Sidecar Resources

Adjust in `deployment.yaml`:

```yaml
# Istio
sidecar.istio.io/proxyCPU: "100m"
sidecar.istio.io/proxyMemory: "128Mi"

# Linkerd
config.linkerd.io/proxy-cpu-request: "100m"
config.linkerd.io/proxy-memory-request: "128Mi"
```

### Circuit Breaking

Configured in Istio DestinationRule:
- Max connections: 100
- Max pending requests: 50
- Outlier detection with 5 consecutive errors

## OpenTelemetry Integration

The deployment is configured for OpenTelemetry:

```yaml
env:
- name: OTLP_ENDPOINT
  value: "http://otel-collector:4317"
- name: ENABLE_TELEMETRY
  value: "true"
```

Both service meshes automatically add trace headers and propagate context.

## References

- [Istio Documentation](https://istio.io/latest/docs/)
- [Linkerd Documentation](https://linkerd.io/2/overview/)
- [Kubernetes Service Mesh Comparison](https://landscape.cncf.io/guide#orchestration-management--service-mesh)
