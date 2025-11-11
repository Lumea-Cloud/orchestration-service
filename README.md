# Orchestration Service

The Orchestration Service manages vLLM deployments on Kubernetes with GPU allocation and resource management for the AI Platform.

## Overview

This service is responsible for:
- Deploying and managing vLLM model instances on Kubernetes
- GPU resource allocation and tracking
- Pod health monitoring and auto-recovery
- Deployment scaling (manual and automatic)
- Node management (draining, cordoning)
- Resource utilization tracking

## Architecture

The service integrates with:
- **Kubernetes**: For container orchestration and GPU scheduling
- **PostgreSQL**: For storing deployment state and resource allocations
- **Auth/Tenant Service**: For tenant validation
- **API Gateway**: For external access routing

## Features

### Deployment Management
- Create vLLM deployments with specified resource requirements
- Delete deployments and clean up resources
- Restart deployments for recovery
- Get deployment status and pod information

### Resource Management
- Track GPU availability across the cluster
- Allocate GPUs to deployments
- Monitor resource utilization (GPU, CPU, Memory)
- Prevent resource overcommitment

### Scaling
- Manual scaling of deployments
- Automatic scaling based on GPU utilization
- Scale event tracking and history

### Node Operations
- Drain nodes for maintenance
- Cordon/uncordon nodes
- Monitor node health and availability

### Health Monitoring
- Continuous pod health checks
- Automatic recovery of failed pods
- Deployment status tracking
- Stuck deployment detection

## API Endpoints

### Internal API (`/internal`)

Used by other services in the platform:

- `POST /internal/deployments/create` - Create a new vLLM deployment
- `DELETE /internal/deployments/{deployment_id}` - Delete a deployment
- `GET /internal/deployments/{deployment_id}/status` - Get deployment status
- `GET /internal/resources/available` - Get available GPU resources
- `GET /internal/deployments` - List deployments with filtering
- `POST /internal/deployments/{deployment_id}/restart` - Restart a deployment

### Admin API (`/admin/v1`)

Administrative operations:

- `GET /admin/v1/infrastructure/nodes` - List GPU nodes
- `GET /admin/v1/infrastructure/gpu-utilization` - Get GPU utilization metrics
- `POST /admin/v1/models/{model_id}/scale` - Scale a model deployment
- `POST /admin/v1/infrastructure/nodes/{node_id}/drain` - Drain a node
- `POST /admin/v1/infrastructure/nodes/{node_id}/uncordon` - Uncordon a node
- `GET /admin/v1/scaling/events` - List scaling events
- `GET /admin/v1/infrastructure/health` - Get infrastructure health

## Database Schema

The service uses the `orchestration` schema with the following tables:

### gpu_nodes
Stores information about GPU nodes in the cluster:
- Node identification and network details
- GPU specifications and capacity
- Current resource allocation
- Node status and schedulability

### pod_allocations
Tracks pod-to-GPU mappings:
- Deployment and tenant association
- Pod location and status
- Resource allocation details
- Performance metrics

### deployments
Manages vLLM deployment state:
- Model and configuration details
- Replica count and scaling settings
- Resource requirements
- Endpoints and status

### scaling_events
Records scaling operations:
- Trigger type and reason
- Before/after replica counts
- Success/failure status

## Configuration

Key environment variables:

```bash
# Service
PORT=8003

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_platform
DB_USER=ai_platform
DB_PASSWORD=ai_platform_dev_2024
DB_SCHEMA=orchestration

# Kubernetes
K8S_CONTEXT=AI
K8S_NAMESPACE=ai-platform

# vLLM
VLLM_IMAGE=vllm/vllm-openai:latest
VLLM_RUNTIME_CLASS=nvidia

# GPU
GPU_TYPE=nvidia-l40s
GPU_MEMORY_GB=48
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Configure Kubernetes access:
```bash
# Ensure kubectl is configured with the AI context
kubectl config use-context AI
```

4. Initialize database schema:
```sql
CREATE SCHEMA IF NOT EXISTS orchestration;
```

## Running the Service

### Development
```bash
python main.py
```

### Docker
```bash
docker-compose up --build
```

### Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8003 --workers 4
```

## Kubernetes Requirements

### Cluster Setup
- Kubernetes cluster with GPU nodes
- NVIDIA GPU device plugin installed
- RuntimeClass "nvidia" configured
- Metrics server for resource monitoring

### Required Permissions
The service needs the following Kubernetes permissions:
- Create/update/delete Deployments and Services
- List/watch Nodes and Pods
- Create Pod evictions
- Update Node specifications

## Monitoring

The service provides:
- Health endpoint at `/health`
- Prometheus metrics (when enabled)
- Structured JSON logging
- Resource utilization tracking

## Background Services

### Health Monitor
- Runs every 30 seconds (configurable)
- Checks deployment and pod health
- Detects stuck deployments
- Triggers auto-recovery

### Resource Tracker
- Runs every 60 seconds (configurable)
- Updates node status from Kubernetes
- Tracks resource allocations
- Cleans up terminated pods

## Error Handling

The service implements:
- Graceful degradation on partial failures
- Automatic retry with exponential backoff
- Comprehensive error logging
- Status persistence in database

## Testing

Run tests with:
```bash
pytest tests/
```

## Security Considerations

- Kubernetes RBAC properly configured
- Database connections use SSL
- Service-to-service authentication required
- Resource quotas enforced

## Performance

Optimizations include:
- Connection pooling for database
- Kubernetes client caching
- Asynchronous operations
- Batch processing for updates

## Troubleshooting

### Common Issues

1. **Deployment stuck in CREATING**
   - Check pod events: `kubectl describe pod <pod-name>`
   - Verify GPU availability
   - Check image pull status

2. **GPU allocation failures**
   - Verify node GPU capacity
   - Check for node taints/tolerations
   - Review resource quotas

3. **Connection to Kubernetes failed**
   - Verify kubeconfig is mounted correctly
   - Check context name matches configuration
   - Ensure service account has proper permissions

## Support

For issues or questions, contact the platform team.

## License

Proprietary - All rights reserved