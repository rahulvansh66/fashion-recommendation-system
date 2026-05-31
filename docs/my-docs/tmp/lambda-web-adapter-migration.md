# Migration from Mangum to AWS Lambda Web Adapter

**Date:** May 26, 2026  
**Status:** Documentation Updated

## Summary

Updated project architecture documentation to use **AWS Lambda Web Adapter (LWA)** instead of Mangum for FastAPI-to-Lambda deployment. This aligns better with the project's core philosophy of "migration-friendly development" and "same code runs locally and on AWS."

## Why This Change?

### Mangum Approach (Previous)
- Requires adding `lambda_handler = Mangum(app)` to application code
- Python-specific (only ASGI/WSGI frameworks)
- Application must import and reference AWS Lambda adapter
- Good for quick prototypes

### AWS Lambda Web Adapter Approach (Current)
- **Zero code changes** — application code stays 100% pure FastAPI
- **Language agnostic** — works with any HTTP server (Node.js, Go, Rust, Java, Python)
- **No AWS dependencies** — application never knows it's running on Lambda
- **True portability** — same container runs on Lambda, ECS, Kubernetes, local Docker
- **Industry standard** — AWS's recommended approach for containerized web apps

## Key Differences

| Aspect | Mangum | AWS Lambda Web Adapter |
|--------|--------|------------------------|
| Code changes | Must add wrapper | Zero changes |
| Language support | Python only | Any language |
| Container required | Optional | Yes (Lambda container image) |
| AWS dependencies | Yes (`pip install mangum`) | No (runs as Lambda extension) |
| Portability | Good | Excellent |
| Production ready | Yes | Yes (AWS official) |

## Implementation Pattern

### Application Code (Identical for Local and AWS)

```python
# api/main.py
from fastapi import FastAPI
import uvicorn

app = FastAPI(title='Fashion Recommendation API')

@app.get('/recommendations/{user_id}')
async def get_recommendations(user_id: str, k: int = 10):
    # Business logic here
    return {'user_id': user_id, 'recommendations': [...]}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

### Dockerfile (Same for Local and AWS)

```dockerfile
FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ .

# Install AWS Lambda Web Adapter as a Lambda Extension
# Only active when running on Lambda; no effect locally
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# Environment variables
ENV PORT=8000

# Same command for local and Lambda
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### How It Works

1. **Local Development:**
   ```bash
   docker run -p 8000:8000 my-api-image
   # FastAPI runs normally, LWA extension is dormant
   ```

2. **AWS Lambda:**
   ```bash
   # Push same image to ECR, deploy to Lambda
   # LWA extension activates, intercepts Lambda events
   # Forwards events as HTTP requests to FastAPI on port 8000
   # Returns FastAPI responses back to Lambda runtime
   ```

## Benefits for This Project

1. **Learning Objective Alignment:** Demonstrates production-grade cloud portability patterns
2. **Cost Optimization:** Same container image = easier testing, no code divergence bugs
3. **Future Flexibility:** Can easily migrate to ECS, EKS, or any container platform
4. **Zero Lock-in:** Application code has no cloud provider dependencies
5. **Industry Best Practice:** AWS-recommended pattern for containerized web applications

## Files Updated

- `CLAUDE.md` — Updated migration patterns and comparison table
- `system-design/infrastructure-layer.md` — Updated architecture overview and Pattern 6
- `system-design/project-description.md` — Updated pipeline diagrams
- `system-design/project-structure.md` — Updated file descriptions and dependencies

## Dependencies Change

### Before (Mangum)
```txt
fastapi
mangum
uvicorn
redis
boto3
faiss-cpu
```

### After (Lambda Web Adapter)
```txt
fastapi
uvicorn
redis
boto3
faiss-cpu
# No mangum needed — LWA is a Lambda layer, not a Python package
```

## Migration Checklist (When Implementing)

- [ ] Update `requirements-serving.txt` to remove `mangum`, ensure `uvicorn` is present
- [ ] Update Dockerfile to include LWA layer
- [ ] Remove any `from mangum import Mangum` imports from code
- [ ] Remove any `lambda_handler = Mangum(app)` declarations
- [ ] Update Terraform/CloudFormation to deploy as Lambda container image (not zip)
- [ ] Test locally with Docker first
- [ ] Deploy to Lambda and validate same behavior

## References

- [AWS Lambda Web Adapter GitHub](https://github.com/awslabs/aws-lambda-web-adapter)
- [AWS Lambda Web Adapter Documentation](https://github.com/awslabs/aws-lambda-web-adapter/blob/main/README.md)
- [Mangum Documentation](https://mangum.io/) (for comparison)

## Notes

- Lambda Web Adapter requires using Lambda container images (not zip packages)
- This aligns perfectly with the project's "Docker Containers for Everything" pattern
- No additional cost — LWA is an open-source Lambda extension
- LWA adds ~1-2ms latency overhead (negligible for this use case)
