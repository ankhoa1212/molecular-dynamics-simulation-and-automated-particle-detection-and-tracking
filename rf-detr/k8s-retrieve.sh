#!/bin/bash
# Retrieves RF-DETR training results from the cluster.
# Works both mid-training (MLflow snapshot) and post-training (full retrieval).
# Run from rf-detr/ directory.
set -uo pipefail

# REQUIRED: export K8S_NAMESPACE=<your-cluster-namespace> before running (see rf-detr/README.md's "Kubernetes Training" section).
NAMESPACE="${K8S_NAMESPACE:?Set K8S_NAMESPACE to your cluster namespace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READER_POD="rf-detr-reader"

die() { echo ""; echo "ERROR: $*"; exit 1; }

# ── Find the training pod ─────────────────────────────────────────────────────
POD=$(kubectl get pods -n "$NAMESPACE" \
    --selector=job-name=rf-detr-training \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

[ -n "$POD" ] \
    || die "No training pod found in namespace $NAMESPACE.\nHas the job been submitted? Run: kubectl get job rf-detr-training -n $NAMESPACE"

JOB_STATUS=$(kubectl get job rf-detr-training -n "$NAMESPACE" \
    -o jsonpath='{.status.conditions[?(@.type=="Complete")].type}' 2>/dev/null || echo "")
POD_PHASE=$(kubectl get pod "$POD" -n "$NAMESPACE" \
    -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")

echo ""
echo "Pod:    $POD"
echo "Phase:  $POD_PHASE"
echo ""

# ── kubectl cp helper: uses training pod when running, reader pod when complete ──
# kubectl cp requires kubectl exec, which fails on Succeeded/Failed pods.
# For completed jobs we spin up a temporary busybox pod that mounts the outputs
# PVC, copy from it, then delete it.

_start_reader_pod() {
    if kubectl get pod "$READER_POD" -n "$NAMESPACE" &>/dev/null; then
        echo "  (reader pod already exists, reusing)"
        return
    fi
    echo "  Spinning up temporary reader pod to access PVC..."
    kubectl run "$READER_POD" \
        --image=busybox \
        --restart=Never \
        -n "$NAMESPACE" \
        --overrides='{
          "spec": {
            "containers": [{
              "name": "reader",
              "image": "busybox",
              "command": ["sh", "-c", "sleep 3600"],
              "volumeMounts": [{"name": "outputs", "mountPath": "/outputs"}]
            }],
            "volumes": [{
              "name": "outputs",
              "persistentVolumeClaim": {"claimName": "rf-detr-outputs"}
            }]
          }
        }' &>/dev/null
    echo "  Waiting for reader pod to be ready..."
    kubectl wait --for=condition=Ready pod/"$READER_POD" -n "$NAMESPACE" --timeout=90s \
        || die "Reader pod did not become ready in time."
}

_stop_reader_pod() {
    if kubectl get pod "$READER_POD" -n "$NAMESPACE" &>/dev/null; then
        kubectl delete pod "$READER_POD" -n "$NAMESPACE" --ignore-not-found &>/dev/null
        echo "  Reader pod deleted."
    fi
}

_cp_src=""   # set to "pod" or "reader" after pod-phase check below

# ── Determine copy source ─────────────────────────────────────────────────────
if [ "$POD_PHASE" = "Running" ]; then
    _cp_src="$NAMESPACE/$POD"
else
    # Pod is Succeeded/Failed — need reader pod to access PVC
    _start_reader_pod
    _cp_src="$NAMESPACE/$READER_POD"
    trap _stop_reader_pod EXIT
fi

# ── Always: snapshot MLflow DB ────────────────────────────────────────────────
echo "Copying MLflow DB snapshot..."
kubectl cp "$_cp_src:/outputs/mlflow.db" "$SCRIPT_DIR/../data-setup/mlflow-a40.db" \
    || die "Failed to copy MLflow DB."

echo "  Saved to: $SCRIPT_DIR/../data-setup/mlflow-a40.db"
echo "  View metrics:"
echo "    cd $SCRIPT_DIR && uv run mlflow ui --backend-store-uri sqlite:///../data-setup/mlflow-a40.db"
echo ""

# ── If complete: full retrieval ───────────────────────────────────────────────
if [ "$JOB_STATUS" = "Complete" ]; then
    echo "Training is complete. Retrieving checkpoints..."

    kubectl cp "$_cp_src:/outputs/checkpoints" "$SCRIPT_DIR/checkpoints-a40" \
        || die "Failed to copy checkpoints."
    echo "  Checkpoints saved to: $SCRIPT_DIR/checkpoints-a40/"
    echo "  Best model:           $SCRIPT_DIR/checkpoints-a40/checkpoint_best_ema.pth"

    echo ""
    echo "Copying final MLflow DB..."
    kubectl cp "$_cp_src:/outputs/mlflow.db" "$SCRIPT_DIR/../data-setup/mlflow-a40.db" \
        || die "Failed to copy final MLflow DB."
    echo "  Saved to: $SCRIPT_DIR/../data-setup/mlflow-a40.db"
    echo "  (Kept separate from mlflow.db to avoid overwriting local runs.)"
    echo "  View A40 runs:"
    echo "    cd $SCRIPT_DIR && uv run mlflow ui --backend-store-uri sqlite:///../data-setup/mlflow-a40.db"
    echo "  View local runs:"
    echo "    cd $SCRIPT_DIR && uv run mlflow ui --backend-store-uri sqlite:///../data-setup/mlflow.db"

    echo ""
    echo "Use the model for particle tracking:"
    echo "  cd $SCRIPT_DIR/../particle-tracking"
    echo "  uv run python track.py --model rf-detr:../rf-detr/checkpoints-a40/checkpoint_best_ema.pth"

    echo ""
    echo "$(printf '─%.0s' {1..60})"
    echo "Clean up cluster resources (run after verifying results):"
    echo "  kubectl delete -f $SCRIPT_DIR/k8s-job.yaml"
    echo "  kubectl delete pvc rf-detr-data rf-detr-outputs -n $NAMESPACE"

else
    echo "Training is still running (phase: $POD_PHASE)."
    echo "Re-run this script after training completes for full retrieval."
    echo ""
    echo "Monitor progress:"
    echo "  kubectl get job rf-detr-training -n $NAMESPACE"
    echo "  kubectl logs -f job/rf-detr-training -n $NAMESPACE"
fi
