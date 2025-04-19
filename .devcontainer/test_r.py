import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Rpy2のセットアップ
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Rの警告メッセージを抑制（任意）
ro.r('options(warn=-1)')

# Rパッケージのインポート
try:
    base = importr('base')
    stats = importr('stats')
    mixomics = importr('mixOmics')
    ggplot2 = importr('ggplot2') # mixOmicsのプロットに必要
    igraph = importr('igraph') # network plotに必要
except RRuntimeError as e:
    print(f"必要なRパッケージが見つからないか、ロードできませんでした。")
    print(f"R環境に 'mixOmics', 'ggplot2', 'igraph' をインストールしてください。")
    print(f"例: install.packages(c('mixOmics', 'ggplot2', 'igraph'))")
    raise e

# pandas DataFrameをRのDataFrameに変換する設定
pandas2ri.activate()

# === 1. シミュレーションデータの生成 ===
def generate_simulation_data(n_samples=100, n_features_a=5, n_features_b=50,
                             n_relevant_a=2, n_relevant_b_per_a=5,
                             effect_strength=1.5, random_seed=42):
    """
    A(離散)とB(連続)のシミュレーションデータを生成する関数。
    Aの特定の列がBの特定の列に影響を与えるように設計。
    """
    np.random.seed(random_seed)

    # A行列 (離散値: 0, 1, 2)
    A = pd.DataFrame(np.random.randint(0, 3, size=(n_samples, n_features_a)),
                     columns=[f'A_feat_{i+1}' for i in range(n_features_a)])
    A.index = [f'Subject_{i+1}' for i in range(n_samples)]

    # B行列 (連続値、基本は標準正規分布)
    B = pd.DataFrame(np.random.randn(n_samples, n_features_b),
                     columns=[f'B_feat_{i+1}' for i in range(n_features_b)])
    B.index = A.index

    # Aの最初のn_relevant_a列がBに影響を与える
    q_idx_start = 0
    for p_idx in range(n_relevant_a):
        relevant_a_col = A.iloc[:, p_idx]
        # このAの列が影響を与えるBの列インデックス
        q_indices = list(range(q_idx_start, q_idx_start + n_relevant_b_per_a))

        for q_idx in q_indices:
            if q_idx < n_features_b:
                # Aの値が大きいほどBの値が大きくなるような効果を追加
                effect = (relevant_a_col - relevant_a_col.mean()) * effect_strength
                B.iloc[:, q_idx] += effect

        q_idx_start += n_relevant_b_per_a
        if q_idx_start >= n_features_b:
            break # Bの列数を超えたら終了

    print(f"Generated A: {A.shape}")
    print(f"Generated B: {B.shape}")
    print(f"A columns {list(A.columns[:n_relevant_a])} influence B columns.")
    return A, B

# === 2. 前処理 ===
def preprocess_data(A, B):
    """
    Aをダミー変数化し、Bを標準化する関数。
    """
    # Aのダミー変数化
    # OneHotEncoderを使う方が列名管理などで柔軟性が高い場合がある
    encoder = OneHotEncoder(sparse_output=False, drop=None) # drop=None で全てのカテゴリを残す
    A_dummy = encoder.fit_transform(A)
    # 分かりやすい列名をつける
    dummy_feature_names = []
    for i, col in enumerate(A.columns):
        for category in encoder.categories_[i]:
            dummy_feature_names.append(f"{col}_{category}")
    A_dummy_df = pd.DataFrame(A_dummy, index=A.index, columns=dummy_feature_names)
    print(f"A after one-hot encoding: {A_dummy_df.shape}")

    # Bの標準化
    scaler = StandardScaler()
    B_scaled = scaler.fit_transform(B)
    B_scaled_df = pd.DataFrame(B_scaled, index=B.index, columns=B.columns)
    print(f"B after scaling: {B_scaled_df.shape}")

    return A_dummy_df, B_scaled_df, scaler # scalerも返す (新規データ用)

# === 3. sPLS-R パラメータチューニング (mixOmics::tune.spls) ===
def tune_spls_parameters(X_r, Y_r, n_components_max=5, n_repeats=5, folds=5,
                          keepX_grid=None, test_keepY_grid=None):
    """
    mixOmics::tune.spls を用いて最適な keepX と ncomp を探索する。
    """
    print(f"\nTuning sPLS parameters using mixOmics::tune.spls...")
    print(f"Max components to test: {n_components_max}")

    # keepXの探索グリッド (例)
    # Xの変数数に応じて調整が必要
    n_var_X = X_r.ncol
    if keepX_grid is None:
      # 例: 5個からXの変数数の半分まで、5刻み程度で探索
      step = max(5, n_var_X // 10)
      keepX_grid = ro.IntVector(list(range(5, n_var_X // 2 + step, step)))
      if len(keepX_grid) == 0 or max(keepX_grid) > n_var_X: # 変数が少ない場合
           keepX_grid = ro.IntVector([min(5, n_var_X)])
      print(f"Using keepX grid: {list(keepX_grid)}")

    # keepY は通常チューニングしない (全変数を使う) が、mixOmicsでは可能
    # test_keepY_grid が指定されれば試す（通常はNoneのまま）

    # チューニング実行
    tune_result = mixomics.tune_spls(
        X=X_r,
        Y=Y_r,
        ncomp=n_components_max,
        validation='Mfold', # M-fold CV
        folds=folds,
        nrepeat=n_repeats,
        test_keepX=keepX_grid,
        test_keepY=test_keepY_grid, # NoneならYはスパースにしない
        progressBar=True,
        # measure='MAE' # or 'MSE', 'RMSE', 'R2', 'Q2' - Q2 is often used
        measure='Q2' # Q2 (予測R2) を指標とする
    )

    # 最適なパラメータの取得
    # tune_resultオブジェクトの構造を確認してアクセスする必要がある
    # 多くのmixOmicsチューニング関数は $choice.ncomp や $choice.keepX を返す
    try:
        # Q2を最大化するコンポーネント数を選択
        q2_mean = ro.r['as.data.frame'](tune_result.rx2['measure.logits']['Q2.total'])
        q2_mean_np = pandas2ri.rpy2py(q2_mean)
        optimal_ncomp = q2_mean_np['means'].idxmax() + 1 # 1-based index -> count

        # optimal_ncomp における最適な keepX を取得
        optimal_keepX_vector = tune_result.rx2['choice.keepX'][optimal_ncomp-1] # R is 1-based

        # 結果をPythonのリストに変換
        optimal_keepX = list(optimal_keepX_vector)

        print(f"Optimal number of components based on Q2: {optimal_ncomp}")
        print(f"Optimal keepX for each component: {optimal_keepX}")
        return optimal_ncomp, optimal_keepX, tune_result
    except Exception as e:
        print(f"Error accessing tuning results: {e}")
        print("Defaulting to ncomp=1 and minimal keepX.")
        # エラー時のフォールバック
        optimal_ncomp = 1
        optimal_keepX = [min(list(keepX_grid))]
        return optimal_ncomp, optimal_keepX, tune_result


# === 4. 最終モデル学習と評価 (mixOmics::spls, mixOmics::perf) ===
def train_and_evaluate_spls(X_r, Y_r, ncomp, keepX, n_repeats=10, folds=5):
    """
    最適化されたパラメータで最終sPLSモデルを学習し、perfで性能評価。
    """
    print(f"\nTraining final sPLS model with ncomp={ncomp}, keepX={keepX}")

    # Rのベクトルに変換
    keepX_r = ro.IntVector(keepX)

    # sPLSモデル学習
    final_spls_model = mixomics.spls(
        X=X_r,
        Y=Y_r,
        ncomp=ncomp,
        keepX=keepX_r,
        mode='regression' # 回帰モードを指定
    )
    print("Final sPLS model trained.")

    # モデル性能評価 (交差検証 + パーミュテーション類似の考え方)
    # perf関数は内部でCVを行い、Q2, R2などの指標を計算する
    # validation='Mfold' でCV、progressBar=Trueで進捗表示
    # validation = 'loo' (Leave-One-Out CV) も可能だが時間がかかる
    print(f"\nEvaluating model performance using mixOmics::perf with {folds}-fold CV (repeats={n_repeats})...")
    # nrepeat を増やすと安定するが時間がかかる
    model_perf = mixomics.perf(
        final_spls_model,
        validation='Mfold',
        folds=folds,
        nrepeat=n_repeats,
        progressBar=True,
        # auc = False # 分類モードでない場合はFalse
    )
    print("Performance evaluation finished.")

    # 結果の表示 (Q2, R2など)
    # perfの結果オブジェクトから必要な情報を抽出する
    # 例: model_perf.rx2['Q2.total'] など
    print("\n--- Performance Metrics ---")
    try:
        q2_total = model_perf.rx2['Q2.total']
        r2_total = model_perf.rx2['R2'] # 通常 R2 は X と Y の両方に対して計算される
        print("Overall Q2 (cumulative predictive R-squared):")
        print(pandas2ri.rpy2py(ro.r['as.data.frame'](q2_total)))
        print("\nOverall R2 (cumulative explained variance):")
        print(pandas2ri.rpy2py(ro.r['as.data.frame'](r2_total)))

        # 各コンポーネントごとの指標も見れる
        print("\nQ2 per component:")
        print(pandas2ri.rpy2py(ro.r['as.data.frame'](model_perf.rx2['Q2'])))

    except Exception as e:
        print(f"Could not extract standard performance metrics: {e}")
        print("Perf object structure might differ. Please inspect 'model_perf'.")


    # --- 有意性について ---
    # mixOmics::perf は、内部的なCV/パーミュテーションに基づき、
    # 各コンポーネントがランダムな場合よりも良い性能を示すか評価します。
    # Q2 > 0 (特に > 0.05 や > 0.0975 などの閾値) であれば、
    # そのコンポーネントはノイズ以上の予測性能を持つと解釈されることが多いです。
    # ただし、これは厳密な統計的有意確率(p値)とは異なります。
    # 論文等で報告する際は、手法（例: Mfold CV with n repeats）と
    # 評価指標（例: Q2）を明記し、閾値（例: Q2>0.05）とその根拠
    # (例: 標準的な基準 or パーミュテーションに基づく経験的評価) を述べます。

    return final_spls_model, model_perf

# === 5. 可視化 (mixOmics plot functions) ===
def visualize_results(spls_model, perf_result, X_df, Y_df, A_orig, output_dir='spls_plots'):
    """
    mixOmicsのプロット関数を使って結果を可視化し、ファイルに保存する。
    """
    print(f"\nGenerating visualizations in directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Rで使用する準備
    X_r = pandas2ri.py2rpy(X_df)
    Y_r = pandas2ri.py2rpy(Y_df)
    ncomp = spls_model.rx2['ncomp'][0] # 学習済みモデルからコンポーネント数を取得

    # Aの元のカテゴリ情報 (色分け用) - データフレームをRに渡す
    A_orig_r = pandas2ri.py2rpy(A_orig)

    # --- 5.1 plotIndiv: サンプル（被験者）プロット ---
    # 各被験者を潜在空間にプロットし、Aのカテゴリで色分け
    # どのAのカテゴリが影響しているかを見るため、影響を与えたAの列を使う
    # ここでは例としてAの最初の列を使う
    group_var_r = A_orig_r.rx(True, 1) # 最初の列を選択
    legend_title = A_orig.columns[0]

    try:
        ro.r.png(os.path.join(output_dir, "plotIndiv_samples.png"), width=800, height=600)
        mixomics.plotIndiv(
            spls_model,
            comp=ro.IntVector([1, 2]), # プロットする潜在変数 (1番目と2番目)
            group=group_var_r,
            ind_names=False, # サンプル名を表示しない (多い場合)
            legend=True,
            legend_title=legend_title,
            title='sPLS-R Samples (Comp 1 vs 2)',
            ellipse=True # 信頼楕円を描画
        )
        ro.r['dev.off']()
        print("plotIndiv saved.")
    except Exception as e:
        print(f"Could not generate plotIndiv: {e}")
        if 'dev.off()' not in str(e): # dev.off()自体のエラーでなければ、デバイスを閉じる試み
             try: ro.r['dev.off']()
             except: pass

    # --- 5.2 plotVar: 変数プロット (Correlation Circle Plot) ---
    # 各変数（Aのダミー変数、Bの変数）と潜在変数の相関を円上にプロット
    try:
        ro.r.png(os.path.join(output_dir, "plotVar_correlation_circle.png"), width=800, height=800)
        mixomics.plotVar(
            spls_model,
            comp=ro.IntVector([1, 2]), # プロットする潜在変数
            var_names=True, # 変数名を表示 (数が多すぎると見にくい)
            cex=ro.IntVector([3, 3]), # 変数名のサイズ調整 (X, Y)
            cutoff=0.5, # 表示する相関の閾値 (例)
            title='sPLS-R Variables (Comp 1 vs 2)'
        )
        ro.r['dev.off']()
        print("plotVar saved.")
    except Exception as e:
        print(f"Could not generate plotVar: {e}")
        if 'dev.off()' not in str(e):
             try: ro.r['dev.off']()
             except: pass

    # --- 5.3 plotLoadings: ローディングプロット ---
    # 各潜在変数に対して、どの変数が重要か（ローディング値が高いか）をバープロットで示す
    for i in range(1, ncomp + 1):
        try:
            ro.r.png(os.path.join(output_dir, f"plotLoadings_comp{i}.png"), width=1000, height=600)
            mixomics.plotLoadings(
                spls_model,
                comp=i,
                method='mean', # or 'median'
                contrib='max', # or 'min', 'both'
                ndisplay=20, # 表示する変数数 (上位)
                title=f'sPLS-R Loadings (Comp {i})'
            )
            ro.r['dev.off']()
            print(f"plotLoadings for Comp {i} saved.")
        except Exception as e:
            print(f"Could not generate plotLoadings for Comp {i}: {e}")
            if 'dev.off()' not in str(e):
                 try: ro.r['dev.off']()
                 except: pass

    # --- 5.4 network: 変数関連ネットワーク ---
    # 変数間の関連性（特にスパースモデルで仮定されるもの）をネットワーク図で示す
    # 注意: このプロットは解釈が難しい場合がある
    try:
        # 相関の閾値を設定 (例: 0.6)
        cutoff_corr = 0.6
        ro.r.png(os.path.join(output_dir, f"network_corr_cutoff_{cutoff_corr}.png"), width=1000, height=1000)
        mixomics.network(
            spls_model,
            comp=ro.IntVector([1,2]), # 考慮するコンポーネント
            color_edge=ro.BoolVector([True]), # エッジの色付け
            cutoff=cutoff_corr, # 表示する関連性の閾値
            save='png', # R内部での保存 (ファイル名は自動生成される場合あり)
            name_save=os.path.join(output_dir, f"network_corr_cutoff_{cutoff_corr}") # 保存名を指定してみる
        )
        # network関数は直接ファイルに保存する機能があるかもしれない
        # dev.off() が不要な場合もある
        # エラーが出なければ dev.off() を試す
        try: ro.r['dev.off']()
        except: pass
        print(f"Network plot (cutoff {cutoff_corr}) potentially saved.")
    except Exception as e:
        print(f"Could not generate network plot: {e}")
        if 'dev.off()' not in str(e):
             try: ro.r['dev.off']()
             except: pass

    # --- 5.5 circosPlot: Circos プロット ---
    # Aの変数とBの変数の間の関連性を円状に表示
    # 表示する関連性の閾値を設定 (例: 0.6)
    cutoff_circos = 0.6
    try:
        # circosPlotは表示する変数が多いと非常に見にくくなる
        # keepXで選択されたX変数と、関連するY変数に絞るなどの工夫が必要な場合がある
        # ここでは、最初のコンポーネントにおける上位N個の変数で試す (例: N=15)
        comp_to_plot = 1
        n_vars_circos = 15
        # ローディングを取得して上位変数を選択 (plotLoadingsと同様のロジックが必要)
        # ここでは簡略化のため、モデルに含まれる全ての変数で試行
        ro.r.png(os.path.join(output_dir, f"circosPlot_comp{comp_to_plot}_cutoff_{cutoff_circos}.png"), width=800, height=800)
        mixomics.circosPlot(
            spls_model,
            comp=comp_to_plot,
            cutoff=cutoff_circos,
            # line = True, # 線でつなぐ
            color_X = ro.StrVector(['blue']*X_df.shape[1]), # X変数の色
            color_Y = ro.StrVector(['red']*Y_df.shape[1]),  # Y変数の色
            size_legend = 1.2
        )
        ro.r['dev.off']()
        print(f"Circos plot (Comp {comp_to_plot}, cutoff {cutoff_circos}) saved.")
    except Exception as e:
        print(f"Could not generate circosPlot: {e}")
        # 変数が多すぎるなどの理由でエラーになることがある
        if 'dev.off()' not in str(e):
             try: ro.r['dev.off']()
             except: pass

    # パフォーマンスプロット (Q2, R2など)
    try:
        # perfの結果をプロット
        ro.r.png(os.path.join(output_dir, "performance_plot.png"), width=800, height=600)
        ro.r.plot(perf_result) # Rの標準プロット関数を使う
        ro.r['dev.off']()
        print("Performance plot saved.")
    except Exception as e:
        print(f"Could not generate performance plot: {e}")
        if 'dev.off()' not in str(e):
             try: ro.r['dev.off']()
             except: pass


# === メイン処理 ===
if __name__ == "__main__":
    # --- 設定 ---
    N_SAMPLES = 150       # 被験者数
    N_FEATURES_A = 10     # Aの特徴量数
    N_FEATURES_B = 100    # Bの特徴量数
    N_RELEVANT_A = 3      # Bに影響を与えるAの特徴量数
    N_RELEVANT_B_PER_A = 8 # 1つの関連A特徴量が影響を与えるBの特徴量数
    EFFECT_STRENGTH = 2.0 # 関連性の強さ
    RANDOM_SEED = 123

    CV_FOLDS = 5          # CVの分割数
    CV_REPEATS = 10       # CVの繰り返し数 (チューニングと評価用)
    N_COMPONENTS_MAX = 5  # テストする最大コンポーネント数

    OUTPUT_DIR = "spls_analysis_results"

    # --- 1. データ生成 ---
    A, B = generate_simulation_data(
        n_samples=N_SAMPLES, n_features_a=N_FEATURES_A, n_features_b=N_FEATURES_B,
        n_relevant_a=N_RELEVANT_A, n_relevant_b_per_a=N_RELEVANT_B_PER_A,
        effect_strength=EFFECT_STRENGTH, random_seed=RANDOM_SEED
    )

    # --- 2. 前処理 ---
    A_dummy, B_scaled, b_scaler = preprocess_data(A, B)

    # --- rpy2 用に R オブジェクトへ変換 ---
    X_r = pandas2ri.py2rpy(A_dummy)
    Y_r = pandas2ri.py2rpy(B_scaled)
    A_orig_r = pandas2ri.py2rpy(A) # 可視化用に元のAも変換


    # --- 3. パラメータチューニング ---
    # keepXのグリッドを指定することも可能 (例: ro.IntVector([5, 10, 15]))
    # keepX_grid_custom = ro.IntVector([5, 10, 15, 20])
    optimal_ncomp, optimal_keepX, tune_results = tune_spls_parameters(
        X_r, Y_r,
        n_components_max=N_COMPONENTS_MAX,
        n_repeats=CV_REPEATS,
        folds=CV_FOLDS,
        # keepX_grid=keepX_grid_custom # 指定しない場合は自動生成
    )

    # --- 4. 最終モデル学習と評価 ---
    if optimal_ncomp > 0: # チューニングで有効なコンポーネントが見つかった場合
        final_model, performance = train_and_evaluate_spls(
            X_r, Y_r,
            ncomp=optimal_ncomp,
            keepX=optimal_keepX,
            n_repeats=CV_REPEATS,
            folds=CV_FOLDS
        )

        # --- 5. 可視化 ---
        visualize_results(final_model, performance, A_dummy, B_scaled, A, output_dir=OUTPUT_DIR)
        print(f"\nAnalysis finished. Results and plots are saved in '{OUTPUT_DIR}'")

    else:
        print("\nNo optimal components found during tuning (Q2 <= 0). Model training and visualization skipped.")

    # --- クリーンアップ (任意) ---
    # Rとの連携解除
    # pandas2ri.deactivate() # 必要に応じて
    