import streamlit as st
import cv2
import io
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
from PIL import Image
import time
import japanize_matplotlib
import os
import sys

st.set_page_config(page_title="超音波プローブ点検アプリ", layout="wide")
st.title("🔊 超音波プローブ点検アプリ")

# プローブ名
st.markdown(
  """
  <style>
    /* subheaderの下余白を削除 */
    h3 {
      margin-bottom: 0px !important;
      padding-bottom: 0px !important;
    }
    /* selectboxの上余白を削除 */
    div[data-testid="stSelectbox"] {
      margin-top: -20px !important;
      padding-top: 0px !important;
    }
  </style>
  """,
  unsafe_allow_html=True
)
st.subheader('●プローブ名の選択')
st.markdown(
  """
  <style>
    /* selectbox 選択中の文字 */
    div[data-baseweb="select"] > div {
      font-size: 18px !important;
      font-weight: bold !important;
      color: #000000 !important;
    }
    /* ドロップダウンリスト内の文字 */
    div[data-baseweb="popover"] li {
      font-size: 18px !important;
      font-weight: bold !important;
      color: #000000 !important;
    }
    /* ドロップダウンリスト内のspan */
    div[data-baseweb="popover"] span {
      font-size: 18px !important;
      font-weight: bold !important;
      color: #000000 !important;
    }
  </style>
  """,
  unsafe_allow_html=True
)

probename = st.selectbox(
  label='',
  options=['', 'X20L', 'L18-4', 'L11-3', 'HL18-4', 'WL13-3', 'L14-4', 'MC10-3', 'EC9-3', 'C5-2', 'S4-2', 'S4-2A'])

uploaded_file = None
if probename in ['X20L', 'L18-4', 'L11-3', 'HL18-4', 'WL13-3', 'L14-4', 'MC10-3', 'EC9-3', 'C5-2', 'S4-2', 'S4-2A']:
  uploaded_file = st.sidebar.file_uploader("超音波画像をアップロード (PNG, JPG, BMP)", type=["png", "jpg", "jpeg", "bmp"])

# ── 注意文のプレースホルダー（画像アップロード後に消す） ──
caution_placeholder = st.empty()

# ── 本体設定項目の見出しプレースホルダー ──
setting_title_placeholder = st.empty()

# ── プリセット表のプレースホルダー（画像アップロード後に消す） ──
preset_placeholder = st.empty()

# 画像がアップロードされていない時だけ注意文とプリセット表を表示
if uploaded_file is None:
  caution_placeholder.markdown(
      """
      <div style="
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 16px 20px;
        font-size: 20px;
        color: #000000;
        font-weight: bold;
      ">
        ⚠️ 【空中放射画像取得における注意点】<br><br>
        ・超音波ゲルがプローブの音響レンズに付着していないことを確認してください。<br><br>
        ・超音波ゲルがプローブの音響レンズに固着している場合は、拭き取るようにしてください。<br><br>
        ・プローブホルダーなどの安定した場所において画像を取得するようにしてください。<br><br>
        ・空中放射画像取得時は、レンズ表面に触れないようにしてください。<br><br>
        ・プローブカバーや穿刺ブラケットが装着されていないことを確認してください。<br><br>
        ・超音波スキャンエリア内に、色付けやテキストの記載などをしないようにしてください。
      </div>
      """,
      unsafe_allow_html=True
    )
  # =========================================================
  # ▼▼▼ 本体設定条件（プリセット）表の表示 ▼▼▼
  # 画像未アップロード かつ プローブ選択済みの時のみ表示
  # =========================================================
  if probename in ['X20L', 'L18-4', 'L11-3', 'HL18-4', 'WL13-3', 'L14-4', 'MC10-3', 'EC9-3', 'C5-2', 'S4-2', 'S4-2A']:

    # 注意点と表の間に見出しを表示
    setting_title_placeholder.subheader("●超音波診断装置本体の設定項目")

    setting_labels = [
      "TGC",
      "Depth (cm)",
      "Focus (mm)",
      "P (Tx Level)",
      "THI",
      "Frequency",
      "BG",
      "DR",
      "B Filter",
      "コントラスト",
      "グレイマップ",
      "音線密度",
      "iXRET",
      "オブリーク強調",
      "フォーカス段数",
      "エッジ強調",
      "B SI-Filter",
      "Compound",
      "Sound Speed",
      "カラー",
      "時間平均",
    ]

    preset_data = {
      "設定項目": setting_labels,
      "C5-2": ["全て中央", 15, "最浅に設定", 100, "On", "Hres", 40, 70, 1, 50, 1, "H", "Off", "Off", 1, "Off", "Off", "Off", 0, "Off", "最大"],
      "S4-2": ["全て中央", 15, "最浅に設定", 100, "On", "Fres", 40, 60, 1, 50, 1, "H", "Off", "Off", 1, "Off", "Off", "Off", 0, "Off", "最大"],
      "L18-4": ["全て中央", 4, "最浅に設定", 100, "On", "Hres2", 40, 60, 1, 50, 1, "H", "Off", "Off", 1, "Off", "Off", "Off", 0, "Off", "最大"],
      "L14-4": ["全て中央", 4, "最浅に設定", 100, "On", "Hres", 30, 60, 1, 50, 1, "H", "Off", "Off", 1, "Off", "Off", "Off", 0, "Off", "最大"],
      "L11-3": ["全て中央", 4, "最浅に設定", 100, "On", "Hgen", 40, 60, 1, 50, 1, "H", "Off", "Off", 1, "Off", "Off", "Off", 0, "Off", "最大"],
      "HL18-4": ["全て中央", 4, "最浅に設定", 100, "Off", "Res", 40, 60, 1, 50, 1, "H", "Off", "Off", 1, "Off", "Off", "Off", 0, "Off", "最大"],
      "EC9-3": ["全て中央", 10, "最浅に設定", 100, "On", "Hres", 40, 60, 1, 50, 1, "H", "Off", "Off", 1, "Off", "Off", "Off", 0, "Off", "最大"],
      "MC10-3": ["全て中央", 4, "最浅に設定", 100, "On", "Hgen", 40, 60, 1, 50, 1, "H", "Off", "Off", 1, "Off", "Off", "Off", 0, "Off", "最大"],
      "WL13-3": ["全て中央", 4, "最浅に設定", 100, "On", "Hgen", 35, 60, 1, 50, 1, "H", "Off", "Off", 1, "Off", "Off", "Off", 0, "Off", "最大"],
      "S4-2A": ["全て中央", 10, "最浅に設定", 100, "On", "Fres", 30, 60, 1, 50, 1, "H", "Off", "Off", 1, "Off", "Off", "Off", 0, "Off", "最大"],
      "X20L": ["全て中央", 4, "最浅に設定", 100, "On", "Hres2", 40, 60, 1, 50, 1, "H", "Off", "Off", 1, "Off", "Off", "Off", 0, "Off", "最大"],
    }

    # ── 全列のDataFrameを作成後、選択プローブ列だけに絞り込む ──
    preset_df = pd.DataFrame(preset_data)
    preset_df = preset_df[["設定項目", probename]]

    # HTMLテーブルで線・文字を濃く表示
    table_html = preset_df.to_html(index=False)
    styled_html = f"""
<style>
  .preset-table {{
    border-collapse: collapse;
    width: 50%;
    font-size: 20px;
  }}
  .preset-table th {{
    background-color: #2c3e50;
    color: white;
    padding: 8px 12px;
    border: 2px solid #555555;
    text-align: center;
    font-weight: bold;
    font-size: 20px;
  }}
  .preset-table td {{
    padding: 6px 12px;
    border: 2px solid #555555;
    text-align: center;
    color: #000000;
    font-weight: bold;
    font-size: 20px;
  }}
  .preset-table tr:nth-child(even) {{
    background-color: #f2f2f2;
  }}
  .preset-table tr:hover {{
    background-color: #dce6f0;
  }}
</style>
{table_html.replace('<table border="1" class="dataframe">', '<table class="preset-table">')}
"""
    preset_placeholder.markdown(styled_html, unsafe_allow_html=True)

else:
  # 画像アップロード時に注意文・見出し・プリセット表を全て消す
  caution_placeholder.empty()
  setting_title_placeholder.empty()
  preset_placeholder.empty()
# ▲▲▲ 本体設定条件（プリセット）表の表示ここまで ▲▲▲

if probename in ['X20L', 'L18-4', 'L11-3', 'WL13-3', 'L14-4', 'EC9-3']:
  chnumber = 192
elif probename in ['HL18-4', 'MC10-3']:
  chnumber = 128
elif probename == 'C5-2':
  chnumber = 160
elif probename in ['S4-2', 'S4-2A']:
  chnumber = 64
else:
  chnumber = 0.1

if uploaded_file is not None:
  file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
  raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
  height, width = gray_img.shape

  bright_thresh = 5
  valid_ratio = 35

  if len(gray_img.shape) == 3:
    g2 = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
  else:
    g2 = gray_img.copy()

  h2, w2 = g2.shape
  col_sum = np.sum(g2 > bright_thresh, axis=0).astype(np.float32)
  row_sum = np.sum(g2 > bright_thresh, axis=1).astype(np.float32)

  col_thresh_val = col_sum.max() * (valid_ratio / 100)
  row_thresh_val = row_sum.max() * (valid_ratio / 100)

  col_valid = np.where(col_sum > col_thresh_val)[0]
  row_valid = np.where(row_sum > row_thresh_val)[0]

  vis2 = cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)

  x_min, x_max = int(col_valid[0]), int(col_valid[-1])
  y_min, y_max = int(row_valid[0]), int(row_valid[-1])
  rw, rh = x_max - x_min, y_max - y_min

  cv2.rectangle(vis2, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
  cv2.putText(vis2, f"ROI: {rw}x{rh}",
        (x_min, max(y_min - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
  roi5 = y_max - 685
  roi6 = x_min
  roi7 = x_max
  roi8 = y_min + 19

  roi_crop = g2[roi8:roi5, roi6:roi7]

  col21, col22 = st.columns(2)
  with col21:
    st.subheader("🖼️ 元画像")
    col21.image(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
  with col22:
    st.subheader("🖼️ ROI切り出し範囲")
    col22.image(roi_crop)

  roi_data = g2[roi8:roi5, roi6:roi7]
  mean_profile = np.mean(roi_data, axis=0)
  roix = np.arange(rw)

  mean_val2 = float(np.mean(mean_profile))
  mean_profile[0:4] = [x + 5 for x in mean_profile[0:4]]
  mean_profile[rw - 4:rw] = [x + 5 for x in mean_profile[rw - 4:rw]]

  std_val2 = float(np.std(mean_profile))
  max_val2 = int(np.max(mean_profile))
  min_val2 = int(np.min(mean_profile))
  maxmin_val2 = max_val2 - min_val2
  meanmin_val2 = round(abs(min_val2 - mean_val2), 2)

  mean_profile2 = np.mean(mean_profile)

  st.markdown("---")
  st.subheader("📈 輝度値のプロファイル")
  fig_line = px.line(
    x=roix, y=mean_profile,
    labels={'x': 'X座標 (px)', 'y': '輝度値 (0-255)'},
    title="ROI領域における輝度値"
  )
  fig_line.update_traces(line_color="#00A8FF")
  fig_line.update_layout(
  height=280,
  margin=dict(l=20, r=20, t=30, b=20),
  yaxis=dict(
   tickmode='linear',  # ★線形モード
   tick0=0,            # ★開始値
   dtick=5,            # ★5刻み
  )
 )
  st.plotly_chart(fig_line, use_container_width=True)

  profile_df = pd.DataFrame({
    "X座標_px": np.arange(rw),
    "Y座標_px": mean_profile,
  })
  csv_profile = profile_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
  st.download_button(
    label="📥 輝度プロファイルデータ (CSV) をダウンロード",
    data=csv_profile,
    file_name=f"line_profile_Y{rw}_{uploaded_file.name}.csv",
    mime="text/csv",
    use_container_width=True
  )

  threshold2 = 20
  profile_df['差分'] = (profile_df['Y座標_px'] - mean_profile2).abs()
  filtered_df = profile_df[profile_df['差分'] < threshold2]

  new_mean = filtered_df['Y座標_px'].mean()
  mean_profile3 = new_mean - mean_profile
  meanmin_val3 = new_mean - min_val2

  fig_line2 = px.line(
    x=roix, y=-(mean_profile3),
    labels={'x': 'X座標 (px)', 'y': '輝度値 (0-255)'},
    title="ROI領域における平均輝度値との差分"
  )
  fig_line2.add_hline(
    y=-20, line_dash="dash", line_color="red", line_width=2,
    annotation=dict(text="閾値: -20（劣化）", font=dict(color="red", size=16),
            align="right", xref="paper", x=1.0, yref="y", y=-20,
            yshift=-30, showarrow=False)
  )
  fig_line2.add_hline(
    y=-15, line_dash="dash", line_color="orange", line_width=2,
    annotation=dict(text="閾値: -15（要注意）", font=dict(color="orange", size=16),
            align="right", xref="paper", x=1.0, yref="y", y=-15,
            yshift=10, showarrow=False)
  )
  fig_line2.update_traces(line_color="#00A8FF")
  fig_line2.update_layout(
  height=280,
  margin=dict(l=20, r=20, t=30, b=20),
  yaxis=dict(
   tickmode='linear',  # ★線形モード
   tick0=0,            # ★開始値
   dtick=5,            # ★5刻み
  )
 )
  st.plotly_chart(fig_line2, use_container_width=True)

  profile_df2 = pd.DataFrame({
    "X座標_px": np.arange(rw),
    "Y座標_輝度値": -(mean_profile3),
  })
  csv_profile2 = profile_df2.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
  st.download_button(
    label="📥 平均輝度差分プロファイルデータ (CSV) をダウンロード",
    data=csv_profile2,
    file_name=f"diff_profile_{uploaded_file.name}.csv",
    mime="text/csv",
    use_container_width=True
  )

  profile_df3 = profile_df2[profile_df2['Y座標_輝度値'] < -20].copy()
  profile_df3['X座標_px'] = profile_df3['X座標_px'] * ((chnumber - 1) / (-(rw))) + ((-rw * chnumber) / (-(rw)))
  profile_df3['X座標_px'] = round(profile_df3['X座標_px'])
  list_from_column = profile_df3['X座標_px'].tolist()
  unique_data = list(dict.fromkeys(list_from_column))
  unique_data1 = pd.Series(unique_data, dtype='int64')
  unique_data2 = [f"{i}ch" for i in unique_data1]

  profile_df3b = profile_df2[(profile_df2['Y座標_輝度値'] < -15) & (profile_df2['Y座標_輝度値'] >= -20)].copy()
  profile_df3b['X座標_px'] = profile_df3b['X座標_px'] * ((chnumber - 1) / (-(rw))) + ((-rw * chnumber) / (-(rw)))
  profile_df3b['X座標_px'] = round(profile_df3b['X座標_px'])
  list_from_column_b = profile_df3b['X座標_px'].tolist()
  unique_data_b = list(dict.fromkeys(list_from_column_b))
  unique_data1_b = pd.Series(unique_data_b, dtype='int64')

  degraded_ch_set = set(unique_data1.tolist())
  unique_data1_b_filtered = unique_data1_b[~unique_data1_b.isin(degraded_ch_set)]
  unique_data2_b = [f"{i}ch" for i in unique_data1_b_filtered]

  st.markdown("---")
  st.header("📊プローブ劣化判定結果")
  if meanmin_val3 > 20:
    st.error('プローブ劣化判定：劣化chあり🚨')
    with st.expander("劣化している詳細chを確認する"):
      for i in unique_data2:
        st.write(i)
    with st.expander("要注意が必要な詳細chを確認する"):
      if unique_data2_b:
        for i in unique_data2_b:
          st.write(i)
  elif meanmin_val3 > 15:
    st.warning('プローブ劣化判定：要注意⚠️')
    with st.expander("要注意が必要な詳細chを確認する"):
      if unique_data2_b:
        for i in unique_data2_b:
          st.write(i)
      else:
        st.write("該当chなし")
  else:
    st.success('プローブ劣化判定:合格🎉')

  overlay_img = None
  image2 = None

  if meanmin_val3 > 20:
    image2 = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image2)
    roi2 = img_array[roi8:roi5, x_min:x_max].copy()
    gray_roi = cv2.cvtColor(roi2, cv2.COLOR_RGB2GRAY)

    profile_df6 = profile_df2[profile_df2['Y座標_輝度値'] < -20].copy()
    profile_df6 = profile_df6.drop(columns=['Y座標_輝度値'])
    list_from_column2 = profile_df6['X座標_px'].tolist()
    list_from_column3 = [f"{i}px" for i in list_from_column2]

    profile_df5 = pd.DataFrame(gray_roi).add_suffix('px')
    profile_df7 = profile_df5[list_from_column3].to_numpy()

    norm_gray = np.clip((new_mean - profile_df7) / new_mean * 255.0, 0, 255).astype(np.uint8)
    norm_gray_inv = 255 - norm_gray
    heatmap_bgr = cv2.applyColorMap(norm_gray_inv, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay_img = img_array.copy()
    roi_height = roi5 - roi8
    roi_width = x_max - x_min
    heatmap_roi_full = np.zeros((roi_height, roi_width, 3), dtype=np.uint8)

    for idx, col_px in enumerate(list_from_column2):
      if 0 <= col_px < roi_width:
        heatmap_roi_full[:, col_px, :] = heatmap_rgb[:, idx, :]

    alpha = 0.7
    beta = 1.0 - alpha
    roi_base = overlay_img[roi8:roi5, x_min:x_max].astype(np.float32)
    roi_heat = heatmap_roi_full.astype(np.float32)
    heat_mask = np.any(roi_heat > 0, axis=2, keepdims=True)
    blended_roi = np.where(
      heat_mask,
      (roi_base * beta + roi_heat * alpha).astype(np.uint8),
      roi_base.astype(np.uint8)
    )
    overlay_img[roi8:roi5, x_min:x_max] = blended_roi

    col51, col52 = st.columns(2)
    with col51:
      st.subheader("元画像")
      st.image(image2, use_container_width=True)
    with col52:
      st.subheader("重ね合わせ画像")
      st.image(overlay_img, use_container_width=True)

  # =========================================================
  # ▼▼▼ PDFレポート生成関数 ▼▼▼
  # =========================================================
  def generate_pdf_report():
    buf = io.BytesIO()
    now_str = datetime.datetime.now().replace(microsecond=0)

    with PdfPages(buf) as pdf:

      # =============================================
      # ページ1
      # =============================================
      fig1 = plt.figure(figsize=(16, 20))
      gs1 = gridspec.GridSpec(
        4, 2,
        figure=fig1,
        height_ratios=[0.08, 0.30, 0.55, 0.07],
        hspace=0.35, wspace=0.25
      )

      ax_title = fig1.add_subplot(gs1[0, :])
      ax_title.axis("off")
      ax_title.text(
        0.5, 0.5,
        "🔊 超音波プローブ劣化判定アプリ",
        transform=ax_title.transAxes,
        fontsize=20, fontweight="bold", ha="center", va="center",
        color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", edgecolor="none")
      )

      ax_info = fig1.add_subplot(gs1[1, 0])
      ax_info.axis("off")
      ax_info.set_facecolor("#f8f9fa")
      info_lines = [
        ("プローブ名", f"{probename}"),
        ("チャンネル数", f"{int(chnumber)} ch"),
        ("解析ファイル", f"{uploaded_file.name}"),
        ("実行日時",  f"{now_str}"),
        ("平均輝度値", f"{mean_profile2:.2f}"),
        ("最大輝度値", f"{max_val2}"),
        ("最小輝度値", f"{min_val2}"),
        ("平均-最小差分", f"{meanmin_val3:.2f}"),
      ]
      ax_info.set_title("基本情報", fontsize=13, fontweight="bold",
               pad=8, loc="left", color="#333333")
      y_pos = 0.92
      for label, value in info_lines:
        ax_info.text(0.03, y_pos, f"{label}",
               transform=ax_info.transAxes,
               fontsize=10, color="#555555", va="top")
        ax_info.text(0.45, y_pos, f": {value}",
               transform=ax_info.transAxes,
               fontsize=10, color="#111111", va="top", fontweight="bold")
        y_pos -= 0.115
      ax_info.patch.set_facecolor("#f8f9fa")
      for spine in ax_info.spines.values():
        spine.set_visible(False)

      ax_judge = fig1.add_subplot(gs1[1, 1])
      ax_judge.axis("off")
      ax_judge.set_title("劣化判定結果", fontsize=13, fontweight="bold",
                pad=8, loc="left", color="#333333")

      if meanmin_val3 > 20:
        badge_color = "#FF4444"
        badge_bg  = "#FFEEEE"
        badge_text = "劣化chあり 🚨"
        section_data = [
          ("【劣化ch】", unique_data2,                 "#FF4444"),
          ("【要注意ch】", unique_data2_b if unique_data2_b else ["なし"], "#FF8800"),
        ]
      elif meanmin_val3 > 15:
        badge_color = "#FF8800"
        badge_bg  = "#FFF3E0"
        badge_text = "要注意 ⚠️"
        section_data = [
          ("【要注意ch】", unique_data2_b if unique_data2_b else ["なし"], "#FF8800"),
        ]
      else:
        badge_color = "#00AA44"
        badge_bg  = "#E8F5E9"
        badge_text = "合格 🎉"
        section_data = []

      ax_judge.text(
        0.5, 0.88,
        f"プローブ劣化判定：{badge_text}",
        transform=ax_judge.transAxes,
        fontsize=15, fontweight="bold",
        color=badge_color, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4",
             facecolor=badge_bg,
             edgecolor=badge_color, linewidth=2.5)
      )

      y_start = 0.72
      for section_title, ch_list, sec_color in section_data:
        ax_judge.text(0.03, y_start, section_title,
               transform=ax_judge.transAxes,
               fontsize=10, fontweight="bold",
               color=sec_color, va="top")
        y_start -= 0.07

        col_count = 0
        x_left, x_right = 0.03, 0.53
        for ch in ch_list:
          x_pos = x_left if col_count % 2 == 0 else x_right
          ax_judge.text(x_pos, y_start, ch,
                 transform=ax_judge.transAxes,
                 fontsize=9, color="#333333", va="top")
          if col_count % 2 == 1:
            y_start -= 0.065
          col_count += 1
          if y_start < 0.01:
            break
        if col_count % 2 == 1:
          y_start -= 0.065
        y_start -= 0.04

      for spine in ax_judge.spines.values():
        spine.set_visible(False)

      ax_orig = fig1.add_subplot(gs1[2, 0])
      ax_orig.imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
      ax_orig.set_title("🖼️ 元画像", fontsize=12, fontweight="bold",
               pad=6, loc="left", color="#333333")
      ax_orig.axis("off")

      ax_roi = fig1.add_subplot(gs1[2, 1])
      ax_roi.imshow(roi_crop, cmap="gray")
      ax_roi.set_title("🖼️ ROI切り出し範囲", fontsize=12, fontweight="bold",
               pad=6, loc="left", color="#333333")
      ax_roi.axis("off")

      ax_foot1 = fig1.add_subplot(gs1[3, :])
      ax_foot1.axis("off")
      ax_foot1.text(
        0.5, 0.5,
        f"実行日時: {now_str}　|　解析ファイル: {uploaded_file.name}　|　プローブ: {probename}",
        transform=ax_foot1.transAxes,
        fontsize=9, ha="center", va="center", color="#888888"
      )

      pdf.savefig(fig1, bbox_inches="tight", dpi=150)
      plt.close(fig1)

      # =============================================
      # ページ2
      # =============================================
      fig2 = plt.figure(figsize=(16, 20))
      gs2 = gridspec.GridSpec(
        3, 1,
        figure=fig2,
        height_ratios=[0.06, 0.44, 0.44],
        hspace=0.30
      )

      ax_t2 = fig2.add_subplot(gs2[0])
      ax_t2.axis("off")
      ax_t2.text(
        0.5, 0.5,
        "📈 輝度値のプロファイル",
        transform=ax_t2.transAxes,
        fontsize=17, fontweight="bold", ha="center", va="center",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="none")
      )

      ax_lp = fig2.add_subplot(gs2[1])
      ax_lp.plot(roix, mean_profile, color="#00A8FF", linewidth=1.4, label="輝度値")
      ax_lp.set_xlim(0, rw)
      ax_lp.set_ylim(0, 255)
      ax_lp.set_xlabel("X座標 (px)", fontsize=12)
      ax_lp.set_ylabel("輝度値 (0-255)", fontsize=12)
      ax_lp.set_title("ROI領域における輝度値", fontsize=14,
              fontweight="bold", pad=10, loc="left", color="#333333")
      ax_lp.legend(fontsize=11)
      ax_lp.grid(True, alpha=0.3, linestyle="--")
      ax_lp.set_facecolor("#fafafa")
      for spine in ax_lp.spines.values():
        spine.set_color("#cccccc")

      diff_y = -(mean_profile3)
      ax_dp = fig2.add_subplot(gs2[2])
      ax_dp.plot(roix, diff_y, color="#00A8FF", linewidth=1.4, label="差分輝度値")
      ax_dp.axhline(y=-20, color="red", linestyle="--",
             linewidth=2.0, label="閾値: -20（劣化）")
      ax_dp.axhline(y=-15, color="orange", linestyle="--",
             linewidth=2.0, label="閾値: -15（要注意）")
      ax_dp.set_xlim(0, rw)
      ax_dp.set_ylim(-70, 30)
      ax_dp.set_xlabel("X座標 (px)", fontsize=12)
      ax_dp.set_ylabel("輝度値 (0-255)", fontsize=12)
      ax_dp.set_title("ROI領域における平均輝度値との差分", fontsize=14,
              fontweight="bold", pad=10, loc="left", color="#333333")
      ax_dp.legend(fontsize=11, loc="upper right")
      ax_dp.grid(True, alpha=0.3, linestyle="--")
      ax_dp.set_facecolor("#fafafa")
      ax_dp.text(rw * 0.99, -20 - 4, "閾値: -20（劣化）",
            color="red", fontsize=10, ha="right")
      ax_dp.text(rw * 0.99, -15 + 2, "閾値: -15（要注意）",
            color="orange", fontsize=10, ha="right")
      for spine in ax_dp.spines.values():
        spine.set_color("#cccccc")

      pdf.savefig(fig2, bbox_inches="tight", dpi=150)
      plt.close(fig2)

      # =============================================
      # ページ3（劣化ありの場合のみ）
      # =============================================
      if overlay_img is not None and image2 is not None:
        fig3 = plt.figure(figsize=(16, 20))
        gs3 = gridspec.GridSpec(
          3, 2,
          figure=fig3,
          height_ratios=[0.06, 0.44, 0.44],
          hspace=0.25, wspace=0.15
        )

        ax_t3 = fig3.add_subplot(gs3[0, :])
        ax_t3.axis("off")
        ax_t3.text(
          0.5, 0.5,
          "🌡️ ヒートマップ重ね合わせ分析",
          transform=ax_t3.transAxes,
          fontsize=17, fontweight="bold", ha="center", va="center",
          color="white",
          bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="none")
        )

        ax_orig2 = fig3.add_subplot(gs3[1, :])
        ax_orig2.imshow(np.array(image2))
        ax_orig2.set_title("元画像", fontsize=13, fontweight="bold",
                  pad=8, loc="left", color="#333333")
        ax_orig2.axis("off")

        ax_over = fig3.add_subplot(gs3[2, :])
        ax_over.imshow(overlay_img)
        ax_over.set_title("重ね合わせ画像 (劣化ch可視化)",
                 fontsize=13, fontweight="bold",
                 pad=8, loc="left", color="#333333")
        ax_over.axis("off")

        pdf.savefig(fig3, bbox_inches="tight", dpi=150)
        plt.close(fig3)

      d = pdf.infodict()
      d["Title"] = "超音波プローブ劣化判定レポート"
      d["Author"] = "超音波プローブ劣化判定アプリ"
      d["Subject"] = f"プローブ: {probename} / ファイル: {uploaded_file.name}"

    buf.seek(0)
    return buf.getvalue()

  st.markdown("---")
  st.header("📷カメラ撮影")
  st.header("●プローブのS/N")
  label1 = st.checkbox("プローブのS/N(ラベル)をカメラで撮影しますか？")
  if label1:
    st.camera_input("プローブのS/N(ラベル)をカメラで撮影してください", width=600)

  st.header("●プローブ外観")
  col11, col12, col13 = st.columns(3)
  with col11:
    probe1 = st.checkbox("プローブ外観をカメラで撮影しますか？(1枚目)")
    if probe1:
      st.camera_input("プローブ外観をカメラで撮影してください(1枚目)", width=600)
  with col12:
    probe2 = st.checkbox("プローブ外観をカメラで撮影しますか？(2枚目)")
    if probe1 and probe2:
      st.camera_input("プローブ外観をカメラで撮影してください(2枚目)", width=600)
  with col13:
    probe3 = st.checkbox("プローブ外観をカメラで撮影しますか？(3枚目)")
    if probe1 and probe2 and probe3:
      st.camera_input("プローブ外観をカメラで撮影してください(3枚目)", width=600)

  now = datetime.datetime.now().replace(microsecond=0)
  st.header(f"●実行日時:{now}")
  st.header("●自由記述欄")
  st.text_area("コメントを入力してください")

  # =========================================================
  # ▼▼▼ 画面印刷（PDF保存）ボタン ▼▼▼
  # =========================================================
  import streamlit.components.v1 as components

  if "print_count" not in st.session_state:
    st.session_state.print_count = 0

  if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

  if "print_just_clicked" not in st.session_state:
    st.session_state.print_just_clicked = False

  if uploaded_file.name != st.session_state.last_uploaded_file:
    st.session_state.print_count = 0
    st.session_state.print_just_clicked = False
    st.session_state.last_uploaded_file = uploaded_file.name

  st.markdown("---")
  st.subheader("📄 分析レポートのダウンロード")

  info_placeholder = st.empty()

  if st.session_state.print_count == 0:
    info_placeholder.info(
      "💡 ボタンを押すとブラウザの印刷ダイアログが開きます。\n"
      "送信先を **「PDFに保存」** に変更してダウンロードしてください。"
    )

  st.markdown(
    """
    <style>
    @media print {
     [data-testid="stSidebar"]  { display: none !important; }
     [data-testid="stToolbar"]  { display: none !important; }
     [data-testid="stHeader"]  { display: none !important; }
     [data-testid="stDecoration"] { display: none !important; }
     iframe           { display: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True
  )

  col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
  with col_btn2:
    if st.button(
      "🖨️ この画面をPDFとして保存する",
      use_container_width=True,
      type="primary"
    ):
      st.session_state.print_count += 1
      st.session_state.print_just_clicked = True
      info_placeholder.empty()

  if st.session_state.print_just_clicked:
    info_placeholder.empty()
    components.html(
      f"""
      <script>
        // count={st.session_state.print_count}
        window.parent.print();
      </script>
      """,
      height=0
    )
    st.session_state.print_just_clicked = False
  # ▲▲▲ 印刷ボタンここまで ▲▲▲
