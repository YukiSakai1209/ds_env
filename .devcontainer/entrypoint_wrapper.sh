#!/bin/bash

echo "[Entrypoint Wrapper] Starting..."

# List of remote sources and local mount points
# Format: "user@host:/remote/path /local/mount/point"
MOUNTS=(
    "yuki1209@ncd-node01g:/home/vermeer/yuki1209 /home/vscode/vermeer"
    "yuki1209@ncd-node02g:/home/magritte/yuki1209 /home/vscode/magritte"
    "yuki1209@ncd-node03g:/home/chagall/yuki1209 /home/vscode/chagall"
    "yuki1209@ncd-node04g:/home/picasso/yuki1209 /home/vscode/picasso"
    "yuki1209@ncd-node05g:/home/ncd/yuki1209 /home/vscode/ncd"
    "yuki1209@ncd-node06g:/home/xnef-data1/yuki1209 /home/vscode/xnef-data1"
    "yuki1209@ncd-node07g:/home/xnef-data2/yuki1209 /home/vscode/xnef-data2"
    "yuki1209@ncd-node08g:/home/cns /home/vscode/cns"
)

# SSHFS base options
SSHFS_BASE_OPTS="-o IdentityFile=/home/vscode/.ssh/id_rsa -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o allow_other -o reconnect -o ServerAliveInterval=15 -o ConnectTimeout=30"

echo "[Entrypoint Wrapper] Mounting standard data servers..."
# Loop through mounts and attempt SSHFS connection for standard data servers
for mount_info in "${MOUNTS[@]}"; do
    REMOTE=$(echo "$mount_info" | awk '{print $1}')
    TARGET_MOUNT_POINT=$(echo "$mount_info" | awk '{print $2}') # Use the defined target mount point
    REMOTE_HOST=$(echo "$REMOTE" | cut -d':' -f1 | cut -d'@' -f2) # Extract hostname

    echo "[Entrypoint Wrapper] Processing standard mount for $TARGET_MOUNT_POINT..."

    echo "[Entrypoint Wrapper] Ensuring mount point $TARGET_MOUNT_POINT exists..."
    mkdir -p "$TARGET_MOUNT_POINT"
    chown vscode:vscode "$TARGET_MOUNT_POINT" # Ensure ownership

    # Start with base options
    CURRENT_SSHFS_OPTS="$SSHFS_BASE_OPTS"

    echo "[Entrypoint Wrapper] Attempting SSHFS mount from $REMOTE to $TARGET_MOUNT_POINT..."
    # Use fusermount to unmount first, in case it was already mounted uncleanly
    fusermount -u "$TARGET_MOUNT_POINT" > /dev/null 2>&1

    # Capture stderr from sshfs by redirecting it to stdout (2>&1)
    sshfs_output=$(sshfs "$REMOTE" "$TARGET_MOUNT_POINT" $CURRENT_SSHFS_OPTS 2>&1)
    sshfs_exit_code=$?

    if [ $sshfs_exit_code -eq 0 ]; then
        echo "[Entrypoint Wrapper] SSHFS mount successful for $REMOTE to $TARGET_MOUNT_POINT."
    else
        echo "[Entrypoint Wrapper] SSHFS mount FAILED for $REMOTE to $TARGET_MOUNT_POINT. Exit code: $sshfs_exit_code"
        echo "[Entrypoint Wrapper] sshfs output: $sshfs_output"
        # Consider adding a sleep or retry mechanism if needed
    fi
    # Add a small delay between mounts if issues persist
    # sleep 1
done

echo "[Entrypoint Wrapper] Finished mounting standard data servers."

# Now, mount the specific project workspace to /workspace if REMOTE_WORKSPACE_PATH is set
if [[ -n "$REMOTE_WORKSPACE_PATH" ]]; then
    echo "[Entrypoint Wrapper] REMOTE_WORKSPACE_PATH is set to: $REMOTE_WORKSPACE_PATH"
    WORKSPACE_MOUNT_POINT="/workspace"

    echo "[Entrypoint Wrapper] Ensuring workspace mount point $WORKSPACE_MOUNT_POINT exists..."
    mkdir -p "$WORKSPACE_MOUNT_POINT"
    chown vscode:vscode "$WORKSPACE_MOUNT_POINT"

    echo "[Entrypoint Wrapper] Attempting SSHFS mount for workspace from $REMOTE_WORKSPACE_PATH to $WORKSPACE_MOUNT_POINT..."
    fusermount -u "$WORKSPACE_MOUNT_POINT" > /dev/null 2>&1
    sshfs_output=$(sshfs "$REMOTE_WORKSPACE_PATH" "$WORKSPACE_MOUNT_POINT" $SSHFS_BASE_OPTS 2>&1)
    sshfs_exit_code=$?

    if [ $sshfs_exit_code -eq 0 ]; then
        echo "[Entrypoint Wrapper] SSHFS workspace mount successful."
    else
        echo "[Entrypoint Wrapper] SSHFS workspace mount FAILED. Exit code: $sshfs_exit_code"
        echo "[Entrypoint Wrapper] sshfs output: $sshfs_output"
    fi
else
    echo "[Entrypoint Wrapper] REMOTE_WORKSPACE_PATH environment variable not set. Skipping workspace mount."
fi

echo "[Entrypoint Wrapper] All mount attempts finished."

# Execute the command passed to the container (e.g., sleep infinity or the command from docker exec)
echo "[Entrypoint Wrapper] Executing command: $@"
exec "$@"
