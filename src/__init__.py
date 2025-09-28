from collections import Counter 
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 共起ネットワーク分析関数
def collocation_network(df, threshold=0.10, min_cocount=1, method='jaccard'):
    # データ情報を取得
    items = df.columns.tolist()
    n_sample = df.shape[0]
    # 共起行列の計算
    X = df.values.astype(int)
    # 単独選択数
    item_counts = X.sum(axis=0)
    # 共起数（n11）
    co_count = X.T @ X
    np.fill_diagonal(co_count, 0) # 自己共起は0
    # 正規化指標
    n = float(n_sample)
    count_mat = co_count.astype(float)
    # ========================================
    # 共起重みの計算（jaccard指数）
    # ========================================
    # 各種分母に必要な配列
    a = item_counts.astype(float)
    b = item_counts.astype(float)
    A = a.reshape(-1, 1)
    B = b.reshape(1, -1)
    # jaccard指数の計算
    if method == 'jaccard':
        # J = n11 / (n10 + n01 + n11) = n11 / (a + b - n11)
        denom = (A + B - count_mat)
        with np.errstate(divide='ignore', invalid='ignore'):
            W = np.where(denom > 0, count_mat / denom, 0.0)
    # 自己共起は0
    np.fill_diagonal(W, 0.0)
    # ========================================
    # グラフ構築（閾値でエッジを抽出）
    # ========================================
    # ネットワークの初期化
    G = nx.Graph()
    # ノードの追加
    for i, item in enumerate(items):
        sel_count = int(item_counts[i])
        sel_rate = sel_count / n_sample
        G.add_node(item, count=sel_count, rate=sel_rate)
    print(f"[Info] Number of nodes: {G.number_of_nodes()}")
    # エッジの追加
    B = len(items)
    edges = []
    for i in range(B):
        for j in range(i + 1, B):
            w = float(W[i, j])
            c = int(co_count[i, j])
            if w >= threshold and c >= min_cocount:
                edges.append((items[i], items[j], {'weight': w, 'cocount': c}))
    G.add_edges_from(edges)
    print(f"[Info] Number of edges: {G.number_of_edges()}")
    # ========================================
    # コミュニティ検出（モジュールベース）
    # ========================================
    # コミュニティ検出
    communities = list(nx.community.greedy_modularity_communities(G, weight='weight'))
    # コミュニティサイズごとのコミュニティ数を表示
    print(f"[Info] Community size distribution:")
    print(f"       Detected {len(communities)} communities.")
    community_sizes = [len(c) for c in communities]
    size_counts = Counter(community_sizes)
    for size, count in sorted(size_counts.items()):
        print(f"       Size {size}: {count} communities")
    # コミュニティ情報をノード属性に追加
    node2comm = {}
    for c_id, comm_nodes in enumerate(communities):
        for v in comm_nodes:
            node2comm[v] = c_id
    nx.set_node_attributes(G, node2comm, 'community')
    # {コミュニティID: {ブランドのセット}} の辞書を作成
    communities_dict = {i: item_list for i, item_list in enumerate(communities)}
    return G, communities_dict

# コミュニティ追跡関数
def tracking_communities(period_analysis_results, threshold=0.2):
    tracked_communities = {}
    communities_counter = 0
    # 各期間を順に処理
    for i, period in enumerate(list(period_analysis_results.keys())):
        print(f"[Info] Processing Period: {period} ({i+1}/{len(list(period_analysis_results.keys()))}) ---")
        # その期間のコミュニティを取得
        current_comunities = period_analysis_results[period]['communities']
        # 最初の期間は全て新規コミュニティとして登録
        if i == 0:
            # 全て新規コミュニティとして登録
            for cid, items in current_comunities.items():
                tracked_communities[communities_counter] = {
                    'id': communities_counter,
                    'name': f"community_{communities_counter}", 
                    'history': {period: items}
                }
                communities_counter += 1
            print(f"[Info]   -> Found {len(current_comunities)} initial communities.")
        # 2期間目以降は前の期間と比較して追跡
        else:
            # 前の期間のコミュニティを取得
            prev_period = list(period_analysis_results.keys())[i - 1]
            prev_tracked_items = {wid: info['history'][prev_period] for wid, info in tracked_communities.items() if prev_period in info['history']}
            # 現在の期間でマッチしたコミュニティIDを記録
            matched_current_cids = set()
            # 前の期間の各コミュニティについて、最も類似度の高い現在のコミュニティを探す
            for wid, prev_items in prev_tracked_items.items():
                best_match_cid = -1 # 最も類似度の高いコミュニティID
                max_jaccard = 0 # 最も高いjaccard指数
                # 現在の各コミュニティと比較
                for cid, current_items in current_comunities.items():
                    # 前期間のコミュニティと現在のコミュニティでのjaccard指数を計算
                    interestion = len(prev_items.intersection(current_items)) # 積集合（AND）
                    union = len(prev_items.union(current_items)) # 和集合（OR）
                    # 和集合が無い場合はスキップ
                    if union == 0: continue
                    # jaccard指数の計算
                    jaccard = interestion / union
                    # 最も高いjaccard指数とそのコミュニティを記録
                    if jaccard > max_jaccard:
                        max_jaccard = jaccard
                        best_match_cid = cid
                # 閾値を超えた場合は同一コミュニティとみなす
                if max_jaccard > threshold and best_match_cid not in matched_current_cids:
                    tracked_communities[wid]['history'][period] = current_comunities[best_match_cid]
                    matched_current_cids.add(best_match_cid)
                    print(f"[Info]   - Match found: {tracked_communities[wid]['name']} -> Current community {best_match_cid} (Jaccard: {max_jaccard:.2f})")
            # マッチしなかったコミュニティは新規として登録
            unmatched_communities = {cid: items for cid, items in current_comunities.items() if cid not in matched_current_cids}
            for cid, items in unmatched_communities.items():
                tracked_communities[communities_counter] = {
                    'id': communities_counter,
                    'name': f"community_{communities_counter}", 
                    'history': {period: items}
                }
                communities_counter += 1
    return tracked_communities

class TemporalCommunityTracker:

    def __init__(self, period_df):
        self.period_list = list(period_df.keys())
        self.period_df = period_df
    
    def static_analysis(self, threshold=0.10, min_cocount=1, method='jaccard'):
        # 各データに対して共起ネットワーク分析
        self.period_analysis_results = {}
        for period, df in self.period_df.items():
            print(f"Analyzing period: {period}")
            G, communities_dict = collocation_network(df, threshold=threshold, min_cocount=min_cocount, method=method)
            self.period_analysis_results[period] = {'graph': G, 'communities': communities_dict}

    def dynamic_analysis(self, threshold=0.2):
        # 全データに対してコミュニティ追跡
        self.tracked_communities = tracking_communities(self.period_analysis_results, threshold=threshold)
        # コミュニティサイズの時系列推移テーブル
        volume_data = []
        for cid, info in self.tracked_communities.items():
            for period, brands in info['history'].items():
                # ボリューム：そのコミュニティに属するアイテムの総選択数
                volume = self.period_df[period][list(brands)].sum().sum()
                volume_data.append({'period': period, 'community_id': info['name'], 'volume': volume})
        df_volume = pd.DataFrame(volume_data).pivot(index='period', columns='community_id', values='volume').fillna(0)
        self.df_volume_result = df_volume.reindex(self.period_list, fill_value=0)
        # 各コミュニティの所属アイテムの時系列テーブル
        table_data = []
        # 追跡された各コミュニティ（世界観）をループ処理
        for community_id, info in self.tracked_communities.items():
            community_name = info['name']
            # そのコミュニティが存在した各期間の履歴をループ処理
            for period, items in info['history'].items():
                # ブランドリストをアルファベット順にソートして可読性を上げる
                sorted_items = sorted(list(items))
                # 1行分のデータを作成
                # ブランドリストは改行区切りの文字列に変換して見やすくする
                row = {
                    'community_id': community_id,
                    'community_name': community_name,
                    'period': period,
                    'imte_count': len(sorted_items),
                    'items': ", ".join(sorted_items) # 改行で連結
                }
                table_data.append(row)
        # リストからDataFrameを作成
        df_brand_table = pd.DataFrame(table_data)
        # PeriodとCommunity_IDでソート
        self.df_items_result = df_brand_table.sort_values(by=['period', 'community_id']).reset_index(drop=True)

    def plot_network(self, period, min_community_size=3,
                 figsize=(15, 10), plt_size=10, font_size=10, title=None,
                 seed=42):
        G = self.period_analysis_results[period]["graph"]
        communities_dict = self.period_analysis_results[period]["communities"]
        # ========================================
        # 指標計算
        # ========================================
        # 強度
        strength = {v: sum(d["weight"] for _, _, d in G.edges(v, data=True)) for v in G.nodes()}
        nx.set_node_attributes(G, strength, 'strength')
        # 次数
        degree = dict(G.degree())
        nx.set_node_attributes(G, degree, 'degree')
        # 媒介中心性
        betweenness = nx.betweenness_centrality(G, weight='weight')
        nx.set_node_attributes(G, betweenness, 'betweenness')
        # ========================================
        # 各ノードの座標計算
        # ========================================
        pos = nx.spring_layout(G, weight="weight", seed=seed)
        # ========================================
        # 描画するノード・エッジの決定
        # ========================================
        # ノードごとのコミュニティ
        node_comm = {v: G.nodes()[v]['community'] for v in G.nodes()}
        # 描画するノード
        plot_nodes = [v for v in G.nodes() if len(communities_dict[node_comm[v]]) >= min_community_size]
        # 描画するエッジ
        plot_edges = [(u, v) for u, v in G.edges() if u in plot_nodes and v in plot_nodes]
        # コミュニティ数
        n_communities = sum(1 for _, c in communities_dict.items() if len(c) >= min_community_size)
        # ========================================
        # 描画
        # ========================================
        plt.figure(figsize=figsize)
        # ノードの色をコミュニティごとに設定
        colors = cm.rainbow(np.linspace(0, 1, n_communities))
        node_colors = [colors[node_comm[v]] for v in plot_nodes]
        # ノードのサイズをサンプル数に基づいて設定
        node_sizes = [G.nodes[v]['rate'] * plt_size *100 for v in plot_nodes]

        # ノード描画
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=plot_nodes, 
                            node_color=node_colors,
                            node_size=node_sizes)
        # エッジ描画
        nx.draw_networkx_edges(G, pos,
                            edgelist=plot_edges,)
        # ラベル描画
        nx.draw_networkx_labels(G, pos,
                                labels={v: v for v in plot_nodes},
                                font_size=font_size)
        plt.title(title)
        plt.tight_layout()
        plt.show()
