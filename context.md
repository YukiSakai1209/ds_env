# DevContainer 設定の問題と解決策のまとめ

## 概要

ユーザーは神経科学・精神疾患研究のための Docker 環境を構築しています。特に、複数の conda 環境（research, dynamicviz, r_env, tf_gpu, torch_gpu）を持つ設定を行っています.

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

1. **docker-compose.yml**：

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

2. **Dockerfile**：

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

3. **devcontainer.json**：

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

## 2025-03-11 追加: ファイル権限の問題と対応

### 現状確認

1. **ローカルディレクトリの権限**：
   ```
   drwxr-xr-x 1 vscode vscode 4096 Mar 10 20:08 .
   ```
   - ディレクトリの権限: `drwxr-xr-x` (755) - 所有者のみ書き込み可能

2. **リモートサーバーのディレクトリ権限**：
   ```
   drwxr-sr-x 1 vscode vscode 136 Feb 13 2023 .
   ```
   - ディレクトリの権限: `drwxr-sr-x` (2755) - setgid bitが設定され、所有者のみ書き込み可能

3. **新規作成ファイルの権限**：
   ```
   -rw-r--r-- 1 vscode vscode 0 Mar 10 23:25 /home/vscode/vermeer/envs/ds_env/test_write_permission.txt
   ```
   - ファイルの権限: `rw-r--r--` (644) - 所有者のみ書き込み可能

### 試行した対応

1. **umaskの設定確認**：
   - `umask 0002`が設定されていることを確認
   - 設定後も新規ファイルは644権限で作成される

2. **ディレクトリ権限の変更**：
   - `find /home/vscode/vermeer -type d -exec chmod 775 {} \;`を試行
   - 一部のディレクトリは権限問題で変更できなかった

3. **ファイル権限の変更**：
   - 特定のディレクトリ内のファイル権限を変更
   - `/home/vscode/vermeer/RL_tasks/DFT_4p/simu_imbalance_effects`内のファイルは変更成功

### 問題点の分析

1. **SSHFSマウントの特性**：
   - SSHFSはリモートサーバーのファイルシステム権限を維持するため、ローカルのumask設定が反映されない場合がある
   - マウントオプションで権限を制御する必要がある

2. **コンテナ内のsudoコマンド**：
   - `sudo`コマンドがインストールされていないため、一部の権限変更が実行できない
   - `apt-get install -y sudo`が必要

### 今後の方針

1. **実用的なアプローチ**：
   - ディレクトリの権限を調整することが必要

2. **SSHFSマウントオプションの最適化**：
   - docker-compose.ymlのSSHFSマウントコマンドに`-o umask=0002`オプションを追加検討
   - ただし、前回の試行で問題が発生したため、慎重に実施

3. **コンテナ再構築時の対応**：
   - devcontainer.jsonのpostCreateCommandに以下を追加検討：
     ```
     apt-get update && apt-get install -y sudo && usermod -aG sudo vscode && echo 'vscode ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/vscode && echo 'umask 0002' >> ~/.bashrc && echo 'umask 0002' >> ~/.profile && chmod -R g+w /home/vscode/vermeer/envs/ds_env
     ```

4. **作業ワークフローの調整**：
   - 書き込みが必要なファイルは、権限問題のないローカルディレクトリで作業
   - 必要に応じてシンボリックリンクを活用

### 結論

現状では、コンテナは正常に起動し、基本的な機能は問題なく動作しています。ファイル権限の問題は、研究作業に実際に支障がある場合に限り、上記の方針に従って対応します。完璧な権限設定を目指すよりも、実際の研究作業の効率を優先する実用的なアプローチを採用します。

## 2025-03-11 追加: 権限設定に関する最終確認

### 現在の設定での問題点

#### 主要ディレクトリ（/home/vscode/vermeer/envs/ds_env）

**現状**：
- ディレクトリ権限: `drwxr-xr-x` (755) - 所有者のみ書き込み可能
- 新規ファイル権限: `rw-r--r--` (644) - 所有者のみ書き込み可能

**問題点**：
- 個人開発の場合、vscodeユーザーが所有者であれば実質的な問題はありません
- 同じユーザーでの作業であれば、ファイルの読み書きに支障はありません
- スクリプトの実行権限が必要な場合は、個別に`chmod +x`を実行する必要があります

#### リモートサーバー

**現状**：
- ディレクトリ権限: `drwxr-sr-x` (2755) - setgid bitが設定され、所有者のみ書き込み可能
- SSHFSマウントされたディレクトリは、リモートサーバーの権限設定が反映されます

**問題点**：
- リモートサーバー上で他のユーザーと共同作業する場合、グループ書き込み権限がないと他のユーザーがファイルを編集できない可能性があります
- 新規作成ファイルがグループ書き込み権限なしで作成されるため、共同作業者がファイルを編集できない場合があります
- setgid bitが設定されているため、新規作成されたファイルやディレクトリは親ディレクトリのグループを継承しますが、権限は継承しません

**結論**：
個人開発のみであれば、現状の設定でも大きな問題はありません。ただし、リモートサーバー上で他のユーザーと共同作業する場合は、グループ書き込み権限の問題が発生する可能性があります。

### umask 0002の効果とSSHFSマウントオプションの影響

#### umask 0002の効果

**umaskとは**：
- umaskは、新規作成されるファイルやディレクトリのデフォルト権限から「差し引く」ビットマスクです
- ファイルのデフォルト権限は666、ディレクトリのデフォルト権限は777です
- umask 0002の場合：
  - ファイル: 666 - 002 = 664 (`rw-rw-r--`) - グループ書き込み権限あり
  - ディレクトリ: 777 - 002 = 775 (`rwxrwxr-x`) - グループ書き込み権限あり

#### SSHFSマウントオプションに`umask=0002`を追加する効果

**期待される変化**：
- SSHFSでマウントされたディレクトリ内で新規作成されるファイルに、グループ書き込み権限(664)が付与されます
- SSHFSでマウントされたディレクトリ内で新規作成されるディレクトリに、グループ書き込み権限(775)が付与されます

**メリット**：
- リモートサーバー上での共同作業がスムーズになります
- 他のユーザーがファイルを編集する際に、権限の問題が発生しにくくなります

**注意点**：
- SSHFSの`umask`オプションは、マウント時に指定する必要があります
- 既存のファイルやディレクトリの権限は変更されません
- リモートサーバーの設定によっては、このオプションが無視される場合があります

### 権限変更の実施タイミングとメリット

#### 実施タイミング

権限変更コマンドは、以下のいずれかのタイミングで実施することを想定しています：

1. **手動での一時的な対応**: 必要に応じて手動で実行
2. **devcontainer.jsonのpostCreateCommand**: コンテナ作成時に自動実行
3. **起動スクリプト**: コンテナ起動時に毎回実行

個人開発の場合は、問題が発生した時に手動で実行する方法が最も簡単です。

#### 権限変更のメリット

```bash
chmod -R 775 /home/vscode/vermeer/envs/ds_env
chmod -R 664 /home/vscode/vermeer/envs/ds_env/*.txt
chmod -R 664 /home/vscode/vermeer/envs/ds_env/*.md
chmod -R 664 /home/vscode/vermeer/envs/ds_env/*.json
chmod -R 664 /home/vscode/vermeer/envs/ds_env/*.yml
```

**個人開発でのメリット**：
- 個人開発のみの場合、実質的なメリットは限定的です
- vscodeユーザーが所有者であれば、既に読み書き権限を持っているため

**共同開発環境でのメリット**：
- グループ内の他のユーザーもファイルを編集できるようになります
- 共同作業がスムーズになります
- 権限問題によるエラーが減少します

**注意点**：
- 実行ファイル（.sh, .py等）の実行権限が必要な場合は、別途`chmod +x`が必要です
- `-R`（再帰的）オプションを使用する場合、意図しないファイルの権限も変更される可能性があります

### 総合的な推奨

神経科学・精神疾患研究の効率的な進行を考慮すると：

1. **個人開発のみの場合**：
   - 現状の設定でも大きな問題はありません
   - 必要に応じて個別にファイルの権限を調整する方法で十分です

2. **共同開発を行う場合**：
   - SSHFSマウントオプションに`umask=0002`を追加することを検討してください
   - 重要なディレクトリの権限を775に、ファイルの権限を664に変更することを検討してください

3. **実用的なアプローチ**：
   - 問題が発生した時に対応する方針が最も効率的です
   - 完璧な権限設定を目指すよりも、研究作業の効率を優先してください
