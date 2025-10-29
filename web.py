# web.py
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib, pickle
import shap
import matplotlib
import matplotlib.pyplot as plt

# 兼容 numpy 旧别名
if not hasattr(np, 'bool'):
    np.bool = bool

# ============== 字体/中文显示 ==================
def setup_chinese_font():
    """设置中文字体（优先系统字体，其次 ./fonts 目录）"""
    try:
        import matplotlib.font_manager as fm
        chinese_fonts = [
            'WenQuanYi Zen Hei','WenQuanYi Micro Hei','SimHei','Microsoft YaHei',
            'PingFang SC','Hiragino Sans GB','Noto Sans CJK SC','Source Han Sans SC'
        ]
        available = [f.name for f in fm.fontManager.ttflist]
        for f in chinese_fonts:
            if f in available:
                matplotlib.rcParams['font.sans-serif'] = [f, 'DejaVu Sans', 'Arial']
                matplotlib.rcParams['font.family'] = 'sans-serif'
                return f

        # 尝试加载 ./fonts 下自带字体
        fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        candidates = [
            'NotoSansSC-Regular.otf','NotoSansCJKsc-Regular.otf',
            'SourceHanSansSC-Regular.otf','SimHei.ttf','MicrosoftYaHei.ttf'
        ]
        if os.path.isdir(fonts_dir):
            import matplotlib.font_manager as fm
            for fname in candidates:
                fpath = os.path.join(fonts_dir, fname)
                if os.path.exists(fpath):
                    fm.fontManager.addfont(fpath)
                    fam = fm.FontProperties(fname=fpath).get_name()
                    matplotlib.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans', 'Arial']
                    matplotlib.rcParams['font.family'] = 'sans-serif'
                    return fam
    except Exception:
        pass

    # 兜底
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    return None

chinese_font = setup_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = matplotlib.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============== 页面配置 ==================
st.set_page_config(
    page_title="基于XGBoost算法预测肺移植术后AKI风险的网页计算器",
    page_icon="🏥",
    layout="wide"
)

# ============== 特征与中文名 ==================
feature_names_display = [
    'icu_admission',      # 再入ICU（0/1）
    'Magnesium_group',    # 镁分组（类别编码 1/2/3）
    'Hypertension',       # 高血压（0/1）
    'APTT_.s',            # APTT
    'Hb',                 # 血红蛋白
    'ALP',                # 碱性磷酸酶
    'K',                  # 钾
    'BUN',                # 血尿素氮
    'Magnesium_CV'        # 镁变异系数
]
feature_names_cn = ['再入ICU','镁分组','高血压','APTT','血红蛋白','碱性磷酸酶','钾','血尿素氮','镁变异系数']
feature_dict = dict(zip(feature_names_display, feature_names_cn))
feature_dict.update({'APTT_.s': 'APTT', 'APTT_.s.': 'APTT'})  # 别名映射

variable_descriptions = {
    'icu_admission': '是否再入ICU（0=否，1=是）',
    'Magnesium_group': '镁分组（建议按 1/2/3 编码：1=1.7-2, 2=≤1.7, 3=>2）',
    'Hypertension': '是否有高血压（0=无，1=有）',
    'APTT_.s': 'APTT（秒）',
    'APTT_.s.': 'APTT（秒）',
    'Hb': '血红蛋白（g/L 或 g/dL，按模型口径）',
    'ALP': '碱性磷酸酶（U/L）',
    'K': '血钾（mmol/L）',
    'BUN': '血尿素氮（mmol/L 或 mg/dL，按模型口径）',
    'Magnesium_CV': '镁变异系数（% 或比例，按模型口径）'
}

# ============== 工具函数 ==================
def _clean_number(x):
    """把 '[3.3101046E-1]'、'3,210'、' 12. ' 等转成 float；失败返回 NaN"""
    if isinstance(x, str):
        s = x.strip().strip('[](){}').replace(',', '')
        try:
            return float(s)
        except Exception:
            return np.nan
    return x

@st.cache_resource
def load_model(model_path: str = './xgb_model.pkl'):
    """加载 xgboost 模型，兼容旧版训练产物：补 use_label_encoder / gpu_id / n_gpus / predictor 等缺失属性"""
    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        # 🔧 兼容补丁：老版本 XGBoost 训练的模型里常见的已废弃/迁移属性
        try:
            if hasattr(model, "__class__") and model.__class__.__name__.startswith("XGB"):
                # 这些属性的存在只为避免 get_params() getattr 报错；值不影响 1.7.6 推理
                defaults = {
                    "use_label_encoder": False,   # 1.x 时代参数，2.x 已废弃
                    "gpu_id": 0,                  # 老版本 GPU 选择；1.7.6 不再需要
                    "n_gpus": 1,                  # 有些旧代码保存过这个
                    "predictor": None,            # 旧参数：cpu_predictor/gpu_predictor
                    "tree_method": getattr(model, "tree_method", None),
                }
                for k, v in defaults.items():
                    if not hasattr(model, k):
                        setattr(model, k, v)
        except Exception:
            pass

        # 尝试获取特征名（优先 sklearn 风格，再退 Booster）
        model_feature_names = None
        try:
            if hasattr(model, 'feature_names_in_'):
                model_feature_names = list(model.feature_names_in_)
        except Exception:
            pass
        if model_feature_names is None:
            try:
                booster = getattr(model, 'get_booster', lambda: None)()
                if booster is not None and hasattr(booster, 'feature_names'):
                    model_feature_names = list(booster.feature_names)
            except Exception:
                model_feature_names = None

        return model, model_feature_names
    except Exception as e:
        raise RuntimeError(f"无法加载模型: {e}")

def predict_proba_safe(model, X_df):
    """优先用 sklearn predict_proba；失败则补属性重试；仍失败则回退到 booster 直接预测概率"""
    # 第一次尝试
    try:
        return model.predict_proba(X_df)
    except AttributeError:
        # 再补一次容错属性（如果模型是从别处传来的）
        for k, v in {"use_label_encoder": False, "gpu_id": 0, "n_gpus": 1, "predictor": None}.items():
            if not hasattr(model, k):
                setattr(model, k, v)
        return model.predict_proba(X_df)
    except Exception:
        # 回退：直接用 booster 预测（要求模型 objective 为二/多分类概率）
        import xgboost as xgb
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is None:
            raise
        dm = xgb.DMatrix(X_df.values, feature_names=list(X_df.columns))
        pred = booster.predict(dm, output_margin=False)
        # pred 形状：二分类通常 (n,), 多分类 (n, K)
        if isinstance(pred, np.ndarray):
            if pred.ndim == 1:  # 二分类概率（正类）
                proba_pos = pred.astype(float)
                return np.vstack([1 - proba_pos, proba_pos]).T
            elif pred.ndim == 2:
                return pred.astype(float)
        raise RuntimeError("Booster 预测回退失败：未知输出形状")

# ============== 主逻辑 ==================
def main():
    # 侧边栏
    st.sidebar.title("基于XGBoost算法预测肺移植术后AKI风险的网页计算器")
    st.sidebar.image(
        "https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg",
        width=200
    )
    st.sidebar.markdown("""
    ### 关于本系统
    通过XGBoost对临床指标进行分析，预测肺移植术后AKI风险。

    **输出：**
    - AKI发生与未发生的概率
    - 风险分层（低/中/高）
    - SHAP模型解释

    """)
    with st.sidebar.expander("变量说明"):
        for f in feature_names_display:
            st.markdown(f"**{feature_dict.get(f, f)}**：{variable_descriptions.get(f, '')}")

    st.title("AKI风险预测（XGBoost）")
    st.markdown("### 请在下方录入全部特征后进行预测")

    # 加载模型
    try:
        model, model_feature_names = load_model('./xgb_model.pkl')
        st.sidebar.success("模型加载成功")
    except Exception as e:
        st.sidebar.error(f"模型加载失败：{e}")
        return

    # 输入区域
    st.header("患者指标录入")
    c1, c2, c3 = st.columns(3)
    with c1:
        icu_admission = st.selectbox("再入ICU", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否", index=0)
        hypertension = st.selectbox("高血压", options=[0, 1], format_func=lambda x: "有" if x == 1 else "无", index=0)
        k = st.number_input("钾（mmol/L）", value=4.0, step=0.1, min_value=1.5, max_value=8.0)
    with c2:
        mg_label_map = {1: '1.7-2', 2: '≤1.7', 3: '>2'}
        magnesium_group_label = st.selectbox("镁分组", options=[1, 2, 3], format_func=lambda x: mg_label_map.get(x, str(x)), index=0)
        aptt = st.number_input("APTT（秒）", value=30.0, step=0.1, min_value=10.0, max_value=100.0)
        bun = st.number_input("血尿素氮（按口径）", value=5.0, step=0.1)
    with c3:
        hb = st.number_input("血红蛋白（按口径）", value=120.0, step=1.0)
        alp = st.number_input("碱性磷酸酶（U/L）", value=80.0, step=1.0)
        mg_cv = st.number_input("镁变异系数（按口径）", value=5.0, step=0.1)

    if st.button("开始预测", type="primary"):
        # 组装输入
        user_inputs = {
            'icu_admission': icu_admission,
            'Magnesium_group': magnesium_group_label,
            'Hypertension': hypertension,
            'APTT_.s': aptt,
            'Hb': hb,
            'ALP': alp,
            'K': k,
            'BUN': bun,
            'Magnesium_CV': mg_cv,
        }

        # 特征名对齐（别名→页面键）
        alias_to_user_key = {
            'APTT_.s.': 'APTT_.s','APTT_.s': 'APTT_.s','aptt_.s': 'APTT_.s','APTT_s':'APTT_.s','APTT.s':'APTT_.s',
            'magnesium_group': 'Magnesium_group','ICU_admission':'icu_admission','icu_Admission':'icu_admission'
        }

        # 构造输入 DataFrame
        if model_feature_names:
            resolved_values, missing_features = [], []
            for c in model_feature_names:
                ui_key = alias_to_user_key.get(c, c)
                val = user_inputs.get(ui_key, None)
                if val is None:
                    missing_features.append(c)
                resolved_values.append(val)
            if missing_features:
                st.error(f"以下模型特征未在页面录入或名称不匹配：{missing_features}")
                with st.expander("调试信息：模型与输入特征名对比"):
                    st.write("模型特征名：", model_feature_names)
                    st.write("页面输入键：", list(user_inputs.keys()))
                return
            input_df = pd.DataFrame([resolved_values], columns=model_feature_names)
        else:
            input_df = pd.DataFrame([[user_inputs[c] for c in feature_names_display]], columns=feature_names_display)

        # 清洗 & 转数值
        input_df = input_df.applymap(_clean_number)
        for c in input_df.columns:
            input_df[c] = pd.to_numeric(input_df[c], errors='coerce')
        if input_df.isnull().any().any():
            st.error("存在缺失/不可解析的输入值，请检查填写格式（数值不要带括号或方括号）。")
            with st.expander("调试：当前输入DataFrame"):
                st.write(input_df)
            return

        # ======== 预测 ========
        try:
            proba = predict_proba_safe(model, input_df)[0]
            if len(proba) == 2:
                no_aki_prob = float(proba[0]); aki_prob = float(proba[1])
            else:
                raise ValueError("返回的概率维度异常")

            # 展示结果
            st.header("AKI风险预测结果")
            a, b = st.columns(2)
            with a:
                st.subheader("未发生AKI概率")
                st.progress(no_aki_prob)
                st.write(f"{no_aki_prob:.2%}")
            with b:
                st.subheader("AKI发生概率")
                st.progress(aki_prob)
                st.write(f"{aki_prob:.2%}")

            risk_level = "低风险" if aki_prob < 0.3 else ("中等风险" if aki_prob < 0.7 else "高风险")
            risk_color = "green" if aki_prob < 0.3 else ("orange" if aki_prob < 0.7 else "red")
            st.markdown(f"### AKI风险评估: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)

            # ======= SHAP 解释 =======
            st.write("---"); st.subheader("模型解释（SHAP）")
            try:
                # 优先通用入口
                try:
                    explainer = shap.Explainer(model)
                    sv = explainer(input_df)  # Explanation
                    shap_value = np.array(sv.values[0])
                    expected_value = sv.base_values[0] if np.ndim(sv.base_values) else sv.base_values
                except Exception:
                    # 回退 TreeExplainer
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_df)
                    if isinstance(shap_values, list):
                        shap_value = np.array(shap_values[1][0])
                        ev = explainer.expected_value
                        expected_value = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
                    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                        shap_value = shap_values[0, :, 1]
                        ev = explainer.expected_value
                        expected_value = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
                    else:
                        shap_value = np.array(shap_values[0])
                        expected_value = explainer.expected_value

                current_features = list(input_df.columns)

                # --- 瀑布图 ---
                st.subheader("SHAP瀑布图")
                import matplotlib.font_manager as fm
                try:
                    c_fonts = [
                        'WenQuanYi Zen Hei','WenQuanYi Micro Hei','Noto Sans CJK SC',
                        'Source Han Sans SC','SimHei','Microsoft YaHei','PingFang SC','Hiragino Sans GB'
                    ]
                    avail = [f.name for f in fm.fontManager.ttflist]
                    for f in c_fonts:
                        if f in avail:
                            plt.rcParams['font.sans-serif'] = [f, 'DejaVu Sans']; break
                except Exception:
                    plt.rcParams['font.sans-serif'] = ['DejaVu Sans','Arial']
                plt.rcParams['axes.unicode_minus'] = False

                fig_waterfall = plt.figure(figsize=(12, 8))
                display_data = input_df.iloc[0].copy()
                # 映射离散变量为中文
                try:
                    if 'icu_admission' in display_data.index:
                        display_data['icu_admission'] = {0:'否',1:'是'}.get(int(display_data['icu_admission']), display_data['icu_admission'])
                    if 'Hypertension' in display_data.index:
                        display_data['Hypertension'] = {0:'无',1:'有'}.get(int(display_data['Hypertension']), display_data['Hypertension'])
                    if 'Magnesium_group' in display_data.index:
                        display_data['Magnesium_group'] = {1:'1.7-2',2:'≤1.7',3:'>2'}.get(int(display_data['Magnesium_group']), display_data['Magnesium_group'])
                except Exception:
                    pass

                try:
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_value,
                            base_values=expected_value,
                            data=display_data.values,
                            feature_names=[feature_dict.get(f, f) for f in current_features]
                        ),
                        max_display=len(current_features),
                        show=False
                    )
                except Exception:
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_value,
                            base_values=expected_value,
                            data=display_data.values,
                            feature_names=current_features
                        ),
                        max_display=len(current_features),
                        show=False
                    )

                # 修正 Unicode 负号，强制字体
                for ax in fig_waterfall.get_axes():
                    for text in ax.texts:
                        s = text.get_text()
                        if '−' in s: text.set_text(s.replace('−','-'))
                        if chinese_font: text.set_fontfamily(chinese_font)
                    for label in ax.get_yticklabels() + ax.get_xticklabels():
                        t = label.get_text()
                        if '−' in t: label.set_text(t.replace('−','-'))
                        if chinese_font: label.set_fontfamily(chinese_font)
                    if chinese_font:
                        ax.set_xlabel(ax.get_xlabel(), fontfamily=chinese_font)
                        ax.set_ylabel(ax.get_ylabel(), fontfamily=chinese_font)
                        ax.set_title(ax.get_title(), fontfamily=chinese_font)

                plt.tight_layout()
                st.pyplot(fig_waterfall); plt.close(fig_waterfall)

                # --- 力图 ---
                st.subheader("SHAP力图")
                try:
                    import streamlit.components.v1 as components
                    force_plot = shap.force_plot(
                        expected_value,
                        shap_value,
                        display_data,
                        feature_names=[feature_dict.get(f, f) for f in current_features]
                    )
                    shap_html = f"""
                    <head>{shap.getjs()}</head>
                    <body><div class="force-plot-container">{force_plot.html()}</div></body>
                    """
                    components.html(shap_html, height=400, scrolling=False)
                except Exception as e:
                    st.warning(f"力图生成失败：{e}")

            except Exception as e:
                st.error(f"无法生成SHAP解释：{e}")
                import traceback; st.error(traceback.format_exc())

        except Exception as e:
            st.error(f"预测或结果展示失败：{e}")
            import traceback; st.error(traceback.format_exc())

    st.write("---")
    st.caption("© AKI Risk Calculator (XGBoost)")

if __name__ == "__main__":
    main()
