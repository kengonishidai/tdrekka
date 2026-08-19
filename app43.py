import streamlit as st
import cv2
import io
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
import time

st.set_page_config(page_title="超音波画像輝度分析アプリ", layout="wide")
st.title("🔊 超音波プローブ劣化判定アプリ")
st.markdown("(プローブの空中放射画像のグレースケール輝度値を解析し、劣化具合を判定します。)")

#プローブ名
probename = st.selectbox(
    '●プローブ名を選択してください(画像をアップロードする前に選択してください)',
    ['X20L', 'L18-4', 'L11-3', 'HL18-4', 'WL13-3', 'L14-4', 'MC10-3', 'EC9-3', 'C5-2', 'S4-2', 'S4-2A'])

if probename == 'X20L' or probename == 'L18-4' or probename == 'L11-3' or probename == 'WL13-3' or probename == 'L14-4' or probename == 'EC9-3':
   chnumber = 192
elif probename == 'HL18-4' or probename == 'MC10-3':
   chnumber = 128
elif probename == 'C5-2':
   chnumber = 160
else:
   chnumber = 64

if probename:
   uploaded_file = st.sidebar.file_uploader("超音波画像をアップロード (PNG, JPG, BMP)", type=["png", "jpg", "jpeg", "bmp"])

# サイドバー: ファイルアップロード
if uploaded_file is not None:
    # 画像読み込み & グレースケール変換
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    height, width = gray_img.shape

    bright_thresh = 60
    valid_ratio = 50

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

    x_min, x_max = int(col_valid[0]),  int(col_valid[-1])
    y_min, y_max = int(row_valid[0]),  int(row_valid[-1])
    rw, rh = x_max - x_min, y_max - y_min

    cv2.rectangle(vis2, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
    cv2.putText(vis2, f"ROI: {rw}x{rh}",
                        (x_min, max(y_min-10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    roi5 = y_max-685
    roi6 = x_min
    roi7 = x_max

    # ROIのみ切り出し
    roi_crop = g2[y_min:roi5, roi6:roi7]

    col21, col22 = st.columns(2)

    with col21:
       st.subheader("🖼️ 元画像")
       col21.image(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB),
                       )
    with col22:
       st.subheader("🖼️ ROI切り出し範囲")
       col22.image(roi_crop)

    # 1. メイン画面：画像表示とROIオーバーレイ
    
    # ROIデータの抽出
    roi_data = g2[y_min:roi5, roi6:roi7]
    

    mean_profile = np.mean(roi_data, axis=0)
    roix = np.arange(rw)

    mean_val2 = float(np.mean(mean_profile))
    mean_profile[0:4] = [x + 5 for x in mean_profile[0:4]]
    mean_profile[rw - 4 :rw] = [x + 5 for x in mean_profile[rw - 4 :rw]]

    std_val2 = float(np.std(mean_profile))
    max_val2 = int(np.max(mean_profile))
    min_val2 = int(np.min(mean_profile))
    maxmin_val2 = max_val2 - min_val2
    meanmin_val2 = round(abs(min_val2 - mean_val2),2)

    mean_profile2 = np.mean(mean_profile)
  
    st.subheader("📈 輝度値のプロファイル")
    # 輝度プロファイルデータ(ROI領域)
    fig_line = px.line(
        x = roix,
        y = mean_profile,
        labels={'x': 'X座標 (px)', 'y': '輝度値 (0-255)'},
        title="ROI領域における輝度値"
    )
    fig_line.update_traces(line_color="#00A8FF")
    fig_line.update_layout(height=280, yaxis_range=[0, 255], margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_line, use_container_width=True)
  
    # プロファイルデータのCSV出力(ROI領域)
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

    new_mean = filtered_df['Y座標_px'] .mean()
    mean_profile3 = new_mean - mean_profile
    meanmin_val3 = new_mean - min_val2

    # 平均輝度の差分プロファイルデータ(ROI領域)
    fig_line2 = px.line(
        x=roix,
        y=-(mean_profile3),
        labels={'x': 'X座標 (px)', 'y': '輝度値 (0-255)'},
        title="ROI領域における平均輝度値との差分"
    )

    fig_line2.add_hline(
      y=-20,                          # meanmin_val3 > 20 の閾値ライン
      line_dash="dash",               # 破線
      line_color="red",               # 赤色
      line_width=2,
      annotation_text="閾値: -20",   # ラベル
      annotation_position="top right",
      annotation_font_color="red"
  )
    fig_line2.update_traces(line_color="#00A8FF")
    fig_line2.update_layout(height=280, yaxis_range=[-70, 30], margin=dict(l=20, r=20, t=30, b=20))
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

    # 不合格ch判定アルゴリズム(平均からの差分輝度値)
    profile_df3 = profile_df2[profile_df2['Y座標_輝度値'] < -20]
    profile_df3['X座標_px'] = profile_df3['X座標_px'] * ((chnumber - 1 )/(-(rw))) + ((-rw*chnumber)/(- (rw)))
    profile_df3['X座標_px'] = round(profile_df3['X座標_px'])
    list_from_column =  profile_df3['X座標_px'].tolist()
    unique_data = list(dict.fromkeys(list_from_column))
    unique_data1 = pd.Series(unique_data, dtype='int64')

    unique_data2 = []
    for i in unique_data1:
      unique_data2.append(f"{i}ch")

   # 中間ch判定アルゴリズム(平均からの差分輝度値)
    profile_df3b = profile_df2[(profile_df2['Y座標_輝度値'] < -15) & (profile_df2['Y座標_輝度値'] >= -20)]
    profile_df3b = profile_df3b.copy()
    profile_df3b['X座標_px'] = profile_df3b['X座標_px'] * ((chnumber - 1) / (-(rw))) + ((-rw * chnumber) / (-(rw)))
    profile_df3b['X座標_px'] = round(profile_df3b['X座標_px'])
    list_from_column_b = profile_df3b['X座標_px'].tolist()
    unique_data_b = list(dict.fromkeys(list_from_column_b))
    unique_data1_b = pd.Series(unique_data_b, dtype='int64')

    unique_data2_b = []
    for i in unique_data1_b:
      unique_data2_b.append(f"{i}ch")

    # 劣化ch判定アルゴリズム
    st.header("📊プローブ劣化判定結果")   
    if meanmin_val3 >20 :
        st.error('プローブ劣化判定：不合格🚨')
        with st.expander("劣化した詳細chを確認する"):
               for i in unique_data2:
                  st.write(i)
        with st.expander("経過観察が必要な詳細chを確認する"):
            if unique_data2_b:
              for i in unique_data2_b:
                st.write(i)
    elif meanmin_val3 > 15 :
        st.warning('プローブ劣化判定：要経過観察⚠️')
        with st.expander("経過観察が必要な詳細chを確認する"):
            if unique_data2_b:
              for i in unique_data2_b:
                st.write(i)
            else:
              st.write("該当chなし")
    else:
        st.success('プローブ劣化判定:合格🎉')

    if meanmin_val3 > 20 :
      # ---------------------------------------------------------
      # ヒートマップの作成
      # --------------------------------------------------------
      # 1. 指定ROIの切り出し
      image2 = Image.open(uploaded_file).convert("RGB")
      img_array = np.array(image2)
      roi2 = img_array[y_min:roi5, x_min:x_max].copy()

      # 2. グレースケール（輝度）変換 (Rec. 601標準)
      gray_roi = cv2.cvtColor(roi2, cv2.COLOR_RGB2GRAY)

      profile_df6 = profile_df2[profile_df2['Y座標_輝度値'] < -20]
      profile_df6 = profile_df6.drop(columns=['Y座標_輝度値'])
      list_from_column2 =  profile_df6['X座標_px'].tolist()
      list_from_column3 =[] 
      for i in list_from_column2:
        list_from_column3.append(f"{i}px")

      profile_df5 = pd.DataFrame(gray_roi)
      profile_df5 = profile_df5.add_suffix('px')

      profile_df7 = profile_df5[list_from_column3]
      profile_df7 = profile_df7.to_numpy() 
    
      # 2. 暗部（gray <= max_threshold）のマスクを作成
      mask = profile_df7 <= new_mean

      # 3. 暗さに応じた数値の正規化（0〜255に引き伸ばす）
      # 暗いピクセル(0)ほど高数値(255)になるように反転スケーリング
      norm_gray = np.zeros_like(profile_df7, dtype=np.float32)
     
      norm_gray = np.clip((new_mean - profile_df7) / new_mean * 255.0, 0, 255)
      norm_gray = norm_gray.astype(np.uint8)

      # 4. カラーマップの適用 (OpenCVはBGRで出力されるためRGBに変換)
      norm_gray_inv = 255 - norm_gray  # ← 反転追加
      heatmap_bgr = cv2.applyColorMap(norm_gray_inv, cv2.COLORMAP_JET)
      heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # =========================================================
    # 5. ヒートマップと元画像の重ね合わせ
    # =========================================================
    # 元画像をコピーしてオーバーレイ用ベース画像を作成
      overlay_img = img_array.copy()

    # ヒートマップを元画像と同じサイズのキャンバスに配置
      heatmap_full = np.zeros_like(img_array, dtype=np.uint8)

    # ROI内の劣化列座標リストを元に、元画像座標へマッピング
      roi_height = roi5 - y_min
      roi_width = x_max - x_min

    # ヒートマップをROIのフルサイズに拡張（劣化列のみ → ROI全体へ）
    # まずROI全体サイズのゼロ画像を作成
      heatmap_roi_full = np.zeros((roi_height, roi_width, 3), dtype=np.uint8)

    # 劣化列のみヒートマップ色を埋め込む
      for idx, col_px in enumerate(list_from_column2):
       if 0 <= col_px < roi_width:
         heatmap_roi_full[:, col_px, :] = heatmap_rgb[:, idx, :]

    # ROI領域に配置
      heatmap_full[y_min:roi5, x_min:x_max] = heatmap_roi_full

    # アルファブレンディングで重ね合わせ
      alpha = 0.7  # ヒートマップの透明度（0.0〜1.0）
      beta = 1.0 - alpha

    # 劣化列が存在するROI範囲のみブレンド
      roi_base = overlay_img[y_min:roi5, x_min:x_max].astype(np.float32)
      roi_heat = heatmap_roi_full.astype(np.float32)

    # ヒートマップが存在するピクセル（黒でない部分）のみブレンド
      heat_mask = np.any(roi_heat > 0, axis=2, keepdims=True)  # shape: (H, W, 1)
      blended_roi = np.where(
      heat_mask,
      (roi_base * beta + roi_heat * alpha).astype(np.uint8),
      roi_base.astype(np.uint8)
    )

      overlay_img[y_min:roi5, x_min:x_max] = blended_roi

    # =========================================================
    # 6. 表示
    # =========================================================
      col51, col52 = st.columns(2)
      with col51:
        st.subheader("元画像")
        st.image(image2, use_container_width=True)
      with col52:
        st.subheader("重ね合わせ画像")
        st.image(overlay_img, use_container_width=True)

    st.header("📷カメラ撮影")
    #プローブのS/N
    st.header("●プローブのS/N")
    label1 = st.checkbox("プローブのS/N(ラベル)をカメラで撮影しますか？")
    if label1:
        st.camera_input("プローブのS/N(ラベル)をカメラで撮影してください",width = 600)

    #プローブの外観
    st.header("●プローブ外観")
    col11,col12,col13 = st.columns(3)

    with col11:
         probe1 = st.checkbox("プローブ外観をカメラで撮影しますか？(1枚目)")
         if probe1:
           st.camera_input("プローブ外観をカメラで撮影してください(1枚目)",width = 600)

    with col12:
         probe2 = st.checkbox("プローブ外観をカメラで撮影しますか？(2枚目)")
         if probe1 and probe2:
            st.camera_input("プローブ外観をカメラで撮影してください(2枚目)",width = 600)

    with col13:
         probe3 = st.checkbox("プローブ外観をカメラで撮影しますか？(3枚目)")
         if probe1 and probe2 and probe3:
            st.camera_input("プローブ外観をカメラで撮影してください(3枚目)",width = 600) 

    now = datetime.datetime.now().replace(microsecond=0)
    st.header(f"●実行日時:{now}")

    st.header("●自由記述欄")
    st.text_area("コメントを入力してください")

else:
    st.info("👈 サイドバーから劣化判定したいプローブの空中放射画像（PNG/JPG等）をアップロードしてください。")

