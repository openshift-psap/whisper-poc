cat << EOF | oc apply -f -
---
apiVersion: v1
kind: Pod
metadata:
  name: trt-standalone
  namespace: my-whisper-runtime
spec:
  containers:
  - name: trt-standalone
    image: quay.io/psap/whisper-poc:latest-trt
    imagePullPolicy: Always
    command:
      - bash
      - -c
    args:
      - |
        sleep infinity
    resources:
      limits:
        nvidia.com/gpu: 1  # Request 1 GPU
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule  # Ensure the pod is scheduled on GPU nodes
EOF
