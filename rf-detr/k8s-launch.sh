#!/bin/bash
# Launches RF-DETR training on a Kubernetes cluster.
# Run from rf-detr/ directory. Safe to re-run — skips completed transfer batches.
set -uo pipefail

# REQUIRED: export K8S_NAMESPACE=<your-cluster-namespace> before running (see rf-detr/README.md's "Kubernetes Training" section).
NAMESPACE="${K8S_NAMESPACE:?Set K8S_NAMESPACE to your cluster namespace}"
# Resolved relative to this repo's top-level data/ directory -- symlink or
# copy your dataset there (see README.md's "Data & Model Availability" section):
#   ln -s /path/to/your/2um-coco-merged ../data/2um-coco-merged
DATASET_SRC="$(dirname "${BASH_SOURCE[0]}")/../data/2um-coco-merged"
RESOLVED_SRC="/tmp/rf-detr-dataset-resolved"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_SIZE=100

# Tracks how far we got — controls cleanup on failure
# init → pvcs_ready → data_ready → job_submitted → training_running
STAGE="init"

# ── Helpers ──────────────────────────────────────────────────────────────────
log()  { echo ""; echo "$*"; }
die()  { echo ""; echo "ERROR: $*"; exit 1; }
step() { echo ""; echo "$*"; echo "$(printf '─%.0s' {1..60})"; }

# ── Cleanup trap ─────────────────────────────────────────────────────────────
do_cleanup() {
    rm -rf "$RESOLVED_SRC" 2>/dev/null || true
    rm -f /tmp/rf-detr-meta.tar.gz /tmp/rf-detr-batch-*.tar.gz 2>/dev/null || true
    kubectl delete pod data-loader -n "$NAMESPACE" \
        --ignore-not-found --wait=false 2>/dev/null || true
    case "$STAGE" in
        job_submitted)
            log "  Removing incomplete training job..."
            kubectl delete job rf-detr-training -n "$NAMESPACE" \
                --ignore-not-found 2>/dev/null || true
            ;;
        training_running)
            log "  Training job is still running on the cluster."
            ;;
    esac
}

on_error_exit() {
    local code=$?
    [ $code -eq 0 ] && return
    log "=== Script failed (exit $code). Cleaning up... ==="
    do_cleanup
    log "=== Fix the issue above and re-run. ==="
}

on_interrupt() {
    log ""
    if [ "$STAGE" = "training_running" ]; then
        log "Detached. Training continues on cluster."
        log "  bash $SCRIPT_DIR/k8s-retrieve.sh"
        trap - EXIT; exit 0
    fi
    log "Interrupted. Cleaning up..."
    do_cleanup
    trap - EXIT; exit 130
}

trap on_error_exit EXIT
trap on_interrupt INT TERM

# ── Pre-flight checks ─────────────────────────────────────────────────────────
step "Pre-flight checks..."

kubectl get nodes &>/dev/null \
    || die "kubectl cannot reach the cluster. Check your kubeconfig."

[ -d "$DATASET_SRC" ] \
    || die "Dataset not found at: $DATASET_SRC\nSymlink or copy your dataset to ../data/2um-coco-merged (see README.md's \"Data & Model Availability\" section)."

which rsync &>/dev/null \
    || die "rsync not found. Install it: sudo apt-get install rsync"

log "  Cluster reachable, dataset found, rsync available."

# ── Step 1: PVCs ─────────────────────────────────────────────────────────────
step "[1/5] Creating PVCs..."
kubectl apply -f "$SCRIPT_DIR/k8s-pvcs.yaml" \
    || die "kubectl apply pvcs failed."

log "Waiting for PVCs to bind (timeout: 3 min)..."
for pvc in rf-detr-data rf-detr-outputs; do
    elapsed=0
    until kubectl get pvc "$pvc" -n "$NAMESPACE" \
            -o jsonpath='{.status.phase}' 2>/dev/null | grep -q "^Bound$"; do
        if [ $elapsed -ge 180 ]; then
            kubectl get storageclass 2>/dev/null || true
            kubectl describe pvc "$pvc" -n "$NAMESPACE" 2>/dev/null | tail -10 || true
            die "$pvc did not bind in 3 min. You may need to set storageClassName in k8s-pvcs.yaml."
        fi
        echo -n "  $pvc: waiting... (${elapsed}s)"$'\r'
        sleep 5; elapsed=$((elapsed + 5))
    done
    log "  $pvc is Bound"
done
STAGE="pvcs_ready"

# ── Step 2: Dataset transfer ──────────────────────────────────────────────────
step "[2/5] Dataset transfer..."

# Create data-loader pod with explicit memory to override namespace default (1Gi)
kubectl apply -n "$NAMESPACE" -f - <<EOF \
    || die "Failed to create data-loader pod."
apiVersion: v1
kind: Pod
metadata:
  name: data-loader
spec:
  restartPolicy: Never
  containers:
    - name: data-loader
      image: busybox
      command: ["sleep", "86400"]
      resources:
        requests:
          memory: "4Gi"
          cpu: "1"
        limits:
          memory: "4Gi"
          cpu: "1"
      volumeMounts:
        - name: data
          mountPath: /data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: rf-detr-data
EOF

log "Waiting for data-loader pod (image pull may take a few minutes)..."
kubectl wait pod/data-loader -n "$NAMESPACE" \
    --for=condition=Ready --timeout=600s \
    || {
        kubectl describe pod data-loader -n "$NAMESPACE" 2>/dev/null | tail -15 || true
        die "data-loader pod did not become Ready within 10 min."
    }
log "  data-loader pod is Ready."

# Resolve symlinks locally — kubectl cp copies symlinks as-is but targets
# don't exist in the container; rsync -L dereferences them to actual files
log "Resolving symlinks (rsync -aL)..."
rm -rf "$RESOLVED_SRC"
rsync -aL "$DATASET_SRC/" "$RESOLVED_SRC/" \
    || die "rsync failed."

EXPECTED=$(find "$RESOLVED_SRC/images" -name "*.png" 2>/dev/null | wc -l)
log "  $EXPECTED images to transfer."

# Transfer metadata (everything except images/) — small, done in one archive
METADATA_OK=$(kubectl exec data-loader -n "$NAMESPACE" -- \
    sh -c '[ -f /data/2um-coco-merged/annotations.json ] && echo yes || echo no' \
    2>/dev/null || echo "no")

if [ "$METADATA_OK" = "yes" ]; then
    log "Metadata already on PVC — skipping."
else
    log "Transferring metadata..."
    kubectl exec data-loader -n "$NAMESPACE" -- mkdir -p /data/2um-coco-merged /data/_chunks
    tar -czf /tmp/rf-detr-meta.tar.gz -C "$RESOLVED_SRC" --exclude="./images" . \
        || die "Failed to archive metadata."
    kubectl cp /tmp/rf-detr-meta.tar.gz "$NAMESPACE/data-loader:/data/_chunks/meta.tar.gz" \
        || die "Failed to transfer metadata archive."
    kubectl exec data-loader -n "$NAMESPACE" -- \
        sh -c 'tar -xzf /data/_chunks/meta.tar.gz -C /data/2um-coco-merged && rm /data/_chunks/meta.tar.gz' \
        || die "Failed to extract metadata on pod."
    rm /tmp/rf-detr-meta.tar.gz
    log "  Metadata transferred."
fi

# Transfer images in independent per-batch archives
# Resumable: skip batches whose images are already on the PVC
EXISTING=$(kubectl exec data-loader -n "$NAMESPACE" -- \
    sh -c 'find /data/2um-coco-merged/images -name "*.png" 2>/dev/null | wc -l' \
    2>/dev/null || echo "0")
EXISTING=$(echo "$EXISTING" | tr -d '[:space:]')

if [ "$EXISTING" -ge "$EXPECTED" ]; then
    log "All $EXPECTED images already on PVC — skipping image transfer."
else
    log "Images on PVC: $EXISTING / $EXPECTED. Resuming from batch $((EXISTING / BATCH_SIZE + 1))..."
    kubectl exec data-loader -n "$NAMESPACE" -- mkdir -p /data/2um-coco-merged/images /data/_chunks

    mapfile -t ALL_IMAGES < <(ls "$RESOLVED_SRC/images/")
    TOTAL_IMAGES=${#ALL_IMAGES[@]}
    TOTAL_BATCHES=$(( (TOTAL_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE ))
    START_BATCH=$(( EXISTING / BATCH_SIZE ))

    for (( b=START_BATCH; b<TOTAL_BATCHES; b++ )); do
        BATCH_NUM=$((b + 1))
        START=$(( b * BATCH_SIZE ))
        BATCH_FILES=("${ALL_IMAGES[@]:$START:$BATCH_SIZE}")
        ARCHIVE="/tmp/rf-detr-batch-${BATCH_NUM}.tar.gz"

        log "  Batch $BATCH_NUM/$TOTAL_BATCHES (${#BATCH_FILES[@]} images)..."
        tar -czf "$ARCHIVE" -C "$RESOLVED_SRC/images" "${BATCH_FILES[@]}" \
            || die "Failed to archive batch $BATCH_NUM."

        BATCH_OK=false
        for attempt in 1 2 3; do
            if kubectl cp "$ARCHIVE" "$NAMESPACE/data-loader:/data/_chunks/batch-${BATCH_NUM}.tar.gz"; then
                BATCH_OK=true; break
            fi
            log "  Attempt $attempt failed, retrying in 15s..."
            sleep 15
        done
        [ "$BATCH_OK" = "true" ] \
            || { rm -f "$ARCHIVE"; die "Batch $BATCH_NUM failed after 3 attempts. Re-run to resume."; }

        BEFORE=$(kubectl exec data-loader -n "$NAMESPACE" -- \
            sh -c 'find /data/2um-coco-merged/images -name "*.png" | wc -l' 2>/dev/null || echo "0")
        BEFORE=$(echo "$BEFORE" | tr -d '[:space:]')

        kubectl exec data-loader -n "$NAMESPACE" -- \
            sh -c "tar -xzf /data/_chunks/batch-${BATCH_NUM}.tar.gz \
                   -C /data/2um-coco-merged/images \
                   && rm /data/_chunks/batch-${BATCH_NUM}.tar.gz" \
            || die "Extraction of batch $BATCH_NUM failed on pod."

        AFTER=$(kubectl exec data-loader -n "$NAMESPACE" -- \
            sh -c 'find /data/2um-coco-merged/images -name "*.png" | wc -l' 2>/dev/null || echo "0")
        AFTER=$(echo "$AFTER" | tr -d '[:space:]')

        [ "$AFTER" -gt "$BEFORE" ] \
            || die "Batch $BATCH_NUM extracted but image count did not increase ($BEFORE → $AFTER)."

        rm -f "$ARCHIVE"
    done
fi

rm -rf "$RESOLVED_SRC"
kubectl exec data-loader -n "$NAMESPACE" -- rm -rf /data/_chunks 2>/dev/null || true
kubectl delete pod data-loader -n "$NAMESPACE" --ignore-not-found 2>/dev/null || true
STAGE="data_ready"
log "Dataset ready on PVC."

# ── Step 3: Submit training job ───────────────────────────────────────────────
step "[3/5] Submitting training job..."

if kubectl get job rf-detr-training -n "$NAMESPACE" &>/dev/null; then
    log "  Removing previous job..."
    kubectl delete job rf-detr-training -n "$NAMESPACE" --wait=true
    sleep 2
fi

kubectl apply -f "$SCRIPT_DIR/k8s-job.yaml" \
    || die "Failed to submit training job."
STAGE="job_submitted"
log "  Job submitted."

# ── Step 4: Wait for training pod ────────────────────────────────────────────
step "[4/5] Waiting for training pod (timeout: 10 min)..."

elapsed=0
while true; do
    PHASE=$(kubectl get pods -n "$NAMESPACE" \
        --selector=job-name=rf-detr-training \
        -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Unknown")
    WAIT_REASON=$(kubectl get pods -n "$NAMESPACE" \
        --selector=job-name=rf-detr-training \
        -o jsonpath='{.items[0].status.containerStatuses[0].state.waiting.reason}' \
        2>/dev/null || echo "")

    case "$WAIT_REASON" in
        ImagePullBackOff|ErrImagePull)
            kubectl describe pod -n "$NAMESPACE" \
                --selector=job-name=rf-detr-training 2>/dev/null | tail -20 || true
            die "Cannot pull image your-registry/rf-detr-trainer:latest. Check Docker Hub." ;;
    esac

    case "$PHASE" in
        Running) log "  Pod is Running."; break ;;
        Failed|Succeeded)
            kubectl logs -n "$NAMESPACE" \
                --selector=job-name=rf-detr-training --tail=50 2>/dev/null || true
            die "Pod entered $PHASE state before training started." ;;
    esac

    if [ $elapsed -ge 600 ]; then
        kubectl describe pod -n "$NAMESPACE" \
            --selector=job-name=rf-detr-training 2>/dev/null | tail -30 || true
        die "Pod did not reach Running within 10 min."
    fi

    MSG="  Status: $PHASE (${elapsed}s)"
    [ -n "$WAIT_REASON" ] && MSG="$MSG — $WAIT_REASON"
    echo -n "$MSG"$'\r'
    sleep 10; elapsed=$((elapsed + 10))
done

# ── Step 5: Confirm training started ─────────────────────────────────────────
step "[5/5] Streaming logs for 90s to confirm training started..."
timeout 90 kubectl logs -f job/rf-detr-training -n "$NAMESPACE" || true
echo "$(printf '─%.0s' {1..60})"

log "Waiting 15s to confirm pod is still healthy..."
sleep 15
FINAL_PHASE=$(kubectl get pods -n "$NAMESPACE" \
    --selector=job-name=rf-detr-training \
    -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Unknown")

if [ "$FINAL_PHASE" != "Running" ]; then
    kubectl logs -n "$NAMESPACE" \
        --selector=job-name=rf-detr-training --tail=50 2>/dev/null || true
    die "Pod is no longer Running (status: $FINAL_PHASE). Training crashed at startup."
fi

STAGE="training_running"

POD=$(kubectl get pods -n "$NAMESPACE" \
    --selector=job-name=rf-detr-training \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "<pod>")

# ── Done ─────────────────────────────────────────────────────────────────────
log "$(printf '═%.0s' {1..60})"
log "  Training is running."
log "$(printf '═%.0s' {1..60})"
log ""
log "Check training progress:"
log "  kubectl get job rf-detr-training -n $NAMESPACE"
log "  kubectl logs -f job/rf-detr-training -n $NAMESPACE"
log ""
log "Copy results / weights (run anytime — works mid-training or after):"
log "  bash $SCRIPT_DIR/k8s-retrieve.sh"
log ""
log "Snapshot MLflow metrics without full retrieval:"
log "  kubectl cp $NAMESPACE/$POD:/outputs/mlflow.db /tmp/mlflow-snapshot.db"
log "  cd $SCRIPT_DIR && uv run mlflow ui --backend-store-uri sqlite:////tmp/mlflow-snapshot.db"
log ""
log "After retrieval, clean up cluster resources:"
log "  kubectl delete -f $SCRIPT_DIR/k8s-job.yaml"
log "  kubectl delete pvc rf-detr-data rf-detr-outputs -n $NAMESPACE"
