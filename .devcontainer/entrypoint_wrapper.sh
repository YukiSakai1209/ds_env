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

# Loop through mounts and attempt SSHFS connection
for mount_info in "${MOUNTS[@]}"; do
    REMOTE=$(echo "$mount_info" | awk '{print $1}')
    MOUNT_POINT=$(echo "$mount_info" | awk '{print $2}')
    REMOTE_HOST=$(echo "$REMOTE" | cut -d':' -f1 | cut -d'@' -f2) # Extract hostname

    echo "[Entrypoint Wrapper] Processing mount for $MOUNT_POINT..."

    echo "[Entrypoint Wrapper] Ensuring mount point $MOUNT_POINT exists..."
    mkdir -p "$MOUNT_POINT"
    chown vscode:vscode "$MOUNT_POINT" # Ensure ownership

    # Start with base options
    CURRENT_SSHFS_OPTS="$SSHFS_BASE_OPTS"

    echo "[Entrypoint Wrapper] Attempting SSHFS mount from $REMOTE to $MOUNT_POINT..."
    # Use fusermount to unmount first, in case it was already mounted uncleanly
    fusermount -u "$MOUNT_POINT" > /dev/null 2>&1

    # Capture stderr from sshfs by redirecting it to stdout (2>&1)
    sshfs_output=$(sshfs "$REMOTE" "$MOUNT_POINT" $CURRENT_SSHFS_OPTS 2>&1)
    sshfs_exit_code=$?

    if [ $sshfs_exit_code -eq 0 ]; then
        echo "[Entrypoint Wrapper] SSHFS mount successful for $MOUNT_POINT."
    else
        echo "[Entrypoint Wrapper] SSHFS mount FAILED for $MOUNT_POINT. Exit code: $sshfs_exit_code"
        echo "[Entrypoint Wrapper] sshfs output: $sshfs_output"
        # Consider adding a sleep or retry mechanism if needed
    fi
    # Add a small delay between mounts if issues persist
    # sleep 1
done

echo "[Entrypoint Wrapper] All mount attempts finished."

# Execute the command passed to the container (e.g., sleep infinity or the command from docker exec)
echo "[Entrypoint Wrapper] Executing command: $@"
exec "$@"
