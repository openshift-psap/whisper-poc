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
    volumeMounts:
    - name: shared-memory
      mountPath: /dev/shm
  volumes:
  - name: shared-memory
    emptyDir:
      medium: Memory
      sizeLimit: 32Gi
EOF

# Connect to the container with:
# oc exec -it trt-standalone -n my-whisper-runtime -- /bin/bash
