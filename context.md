# DevContainer 設定の問題と解決策のまとめ

## 概要

ユーザーは神経科学・精神疾患研究のための Docker 環境を構築しています。特に、複数の conda 環境（research, dynamicviz, r_env, tf_gpu, torch_gpu）を持つ設定を行っています。

## 問題の根本原因

1. **NVIDIA エントリポイントの扱い**:

   - Windsurf は自動的に`/opt/nvidia/nvidia_entrypoint.sh`をエントリポイントとして設定
   - このファイルが存在しないか、正しく設定されていないとコンテナが起動直後に終了

2. **Dockerfile と docker-compose.yml の不整合**:

   - Dockerfile では`USER vscode`が設定されているが、コマンド実行時に権限問題が発生
   - SSHFS マウントコマンドの構文とオプション指定方法に問題

3. **ユーザー作成方法の違い**:
   - 旧設定: `usermod -u 1001 vscode` (既存ユーザーの変更)
   - 新設定: `useradd --uid $USER_UID --gid $USER_GID -m $USERNAME` (新規ユーザー作成)
   - この違いにより、ユーザー環境やグループメンバーシップが異なる

## 解決策

### 解決のポイント

1. **シェル指定の統一**:

   - `/bin/sh -c`を使用してコマンドを実行（bash ではなく）
   - 複数行コマンドの正しい連結

2. **SSHFS オプションの最適化**:

   - `-o allow_other -o default_permissions`オプションを追加
   - これにより、Dockerfile で設定された`user_allow_other`と連携

3. **NVIDIA エントリポイントの維持**:

   - `/opt/nvidia/nvidia_entrypoint.sh`を正しく作成・設定
   - GPU サポートに必要なエントリポイントを維持

4. **devcontainer.json の設定最適化**:
   - `remoteUser`と`containerUser`の設定を維持
   - 複雑な`postCreateCommand`をシンプル化

### 具体的な解決策

最終的に機能した設定は以下の通りです：

1. **docker-compose.yml**:

   ```yaml
   entrypoint: /opt/nvidia/nvidia_entrypoint.sh
   command: |
     /bin/sh -c "
     chmod 600 /home/vscode/.ssh/id_rsa &&
     mkdir -p /home/vscode/vermeer &&
     # 他のディレクトリ作成
     sshfs yuki1209@ncd-node01g:/home/vermeer/yuki1209 /home/vscode/vermeer -o allow_other -o default_permissions -o reconnect -o ServerAliveInterval=15 -o ServerAliveCountMax=3 &&
     # 他のSSHFSマウント
     sleep infinity"
   ```

2. **Dockerfile**:

   ```Dockerfile
   # NVIDIAエントリポイントスクリプトを作成
   RUN mkdir -p /opt/nvidia \
       && echo '#!/bin/bash' > /opt/nvidia/nvidia_entrypoint.sh \
       && echo 'set -e' >> /opt/nvidia/nvidia_entrypoint.sh \
       && echo 'if [[ $# -eq 0 ]]; then' >> /opt/nvidia/nvidia_entrypoint.sh \
       && echo '    exec /bin/bash' >> /opt/nvidia/nvidia_entrypoint.sh \
       && echo 'else' >> /opt/nvidia/nvidia_entrypoint.sh \
       && echo '    exec "$@"' >> /opt/nvidia/nvidia_entrypoint.sh \
       && echo 'fi' >> /opt/nvidia/nvidia_entrypoint.sh \
       && chmod 755 /opt/nvidia/nvidia_entrypoint.sh
   ```

3. **devcontainer.json**:

   ```json
   "remoteUser": "vscode",
   "containerUser": "vscode",
   "updateRemoteUserUID": true
   ```

## 重要なポイント

- ホスト PC での GPU 認識は現状までで問題ないことを確認済み：
  - TensorFlow: `TensorFlow GPU available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`
  - PyTorch: `PyTorch GPU available: True, Device count: 1`
- PyTorch の CUDA が環境の CUDA のバージョンと一致していないが、PyTorch からの要請でもあり、現状 GPU 認識自体は問題なくできているので維持。
