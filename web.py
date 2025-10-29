import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import matplotlib
import shap
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

if not hasattr(np, 'bool'):
    np.bool = bool

def setup_chinese_font():
    """设置中文字体（云端优先加载本地fonts目录内的CJK字体）"""
    try:
        import os
        import matplotlib.font_manager as fm

        # 优先尝试系统已安装字体
        chinese_fonts = [
            'WenQuanYi Zen Hei',
            'WenQuanYi Micro Hei',
            'SimHei',
            'Microsoft YaHei',
            'PingFang SC',
            'Hiragino Sans GB',
            'Noto Sans CJK SC',
            'Source Han Sans SC'
        ]

        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in chinese_fonts:
            if font in available_fonts:
                matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial']
                matplotlib.rcParams['font.family'] = 'sans-serif'
                print(f"使用中文字体: {font}")
                return font

        # 若系统无中文字体，尝试从./fonts 目录加载随应用打包的字体
        candidates = [
            'NotoSansSC-Regular.otf',
            'NotoSansCJKsc-Regular.otf',
            'SourceHanSansSC-Regular.otf',
            'SimHei.ttf',
            'MicrosoftYaHei.ttf'
        ]
        fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        if os.path.isdir(fonts_dir):
            for fname in candidates:
                fpath = os.path.join(fonts_dir, fname)
                if os.path.exists(fpath):
                    try:
                        fm.fontManager.addfont(fpath)
                        fp = fm.FontProperties(fname=fpath)
                        fam = fp.get_name()
                        matplotlib.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans', 'Arial']
                        matplotlib.rcParams['font.family'] = 'sans-serif'
                        print(f"使用本地打包字体: {fam} ({fname})")
                        return fam
                    except Exception as ie:
                        print(f"加载本地字体失败 {fname}: {ie}")

        # 兜底：使用英文字体（中文将显示为方框）
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        print("未找到中文字体，使用默认英文字体")
        return None

    except Exception as e:
        print(f"字体设置失败: {e}")
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        return None

chinese_font = setup_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="基于XGBoost算法预测肺移植术后AKI风险的网页计算器",
    page_icon="🏥",
    layout="wide"
)


if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans', 'Arial']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False 


global feature_names_display, feature_dict, variable_descriptions


feature_names_display = [
    'icu_admission',      # 再入ICU（0/1）
    'Magnesium_group',    # 镁分组（类别编码）
    'Hypertension',       # 高血压（0/1）
    'APTT_.s',            # APTT
    'Hb',                 # 血红蛋白
    'ALP',                # 碱性磷酸酶
    'K',                  # 钾
    'BUN',                # 血尿素氮
    'Magnesium_CV'        # 镁变异系数
]

feature_names_cn = [
    '再入ICU', '镁分组', '高血压', 'APTT', '血红蛋白', '碱性磷酸酶', '钾', '血尿素氮', '镁变异系数'
]

feature_dict = dict(zip(feature_names_display, feature_names_cn))
# 统一显示名称：无论是 'APTT_.s' 还是 'APTT_.s.'，界面均显示为 'APTT'
feature_dict.update({'APTT_.s': 'APTT', 'APTT_.s.': 'APTT'})

# 变量说明字典（9个新特征）
variable_descriptions = {
    'icu_admission': '是否再入ICU（0=否，1=是）',
    'Magnesium_group': '镁分组（建议按0/1/2编码）',
    'Hypertension': '是否有高血压（0=无，1=有）',
    'APTT_.s': 'APTT（秒）',
    'APTT_.s.': 'APTT（秒）',
    'Hb': '血红蛋白（g/L 或 g/dL，按模型数据口径）',
    'ALP': '碱性磷酸酶（U/L）',
    'K': '血钾（mmol/L）',
    'BUN': '血尿素氮（mmol/L 或 mg/dL，按模型数据口径）',
    'Magnesium_CV': '镁变异系数（% 或比例，按模型数据口径）'
}

@st.cache_resource
def load_model(model_path: str = './xgb_model.pkl'):
    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        model_feature_names = None
        if hasattr(model, 'feature_names_in_'):
            model_feature_names = list(model.feature_names_in_)
        else:
            try:
                booster = getattr(model, 'get_booster', lambda: None)()
                if booster is not None:
                    model_feature_names = booster.feature_names
            except Exception:
                model_feature_names = None

        return model, model_feature_names
    except Exception as e:
        raise RuntimeError(f"无法加载模型: {e}")


def main():
    global feature_names_display, feature_dict, variable_descriptions

    # 侧边栏标题
    st.sidebar.title("基于XGBoost算法预测肺移植术后AKI风险的网页计算器")
    st.sidebar.image("https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg", width=200)

    # 添加系统说明到侧边栏
    st.sidebar.markdown("""
    # 系统说明

    ## 关于本系统
    这是一个基于XGBoost算法的肺移植术后AKI风险预测系统，通过分析患者的临床指标来预测AKI发生的可能性。

    ## 预测结果
    系统输出：
    - AKI发生概率
    - 未发生AKI概率
    - 风险分层（低/中/高）

    ## 使用方法
    1. 在主界面填写患者的各项指标（一个页面集中展示）
    2. 点击“开始预测”按钮
    3. 查看预测结果与模型解释（SHAP）

    ## 提示
    - 请按模型数据口径填写数值（单位以模型训练集为准）
    - 若某些类别变量的编码不确定，请与训练口径保持一致
    """)

    # 添加变量说明到侧边栏
    with st.sidebar.expander("变量说明"):
        for feature in feature_names_display:
            st.markdown(f"**{feature_dict[feature]}**: {variable_descriptions[feature]}")

    # 主页面标题
    st.title("基于XGBoost算法预测肺移植术后AKI风险的网页计算器")
    st.markdown("### 请在下方录入全部特征后进行预测")

    # 加载模型
    try:
        model, model_feature_names = load_model('./xgb_model.pkl')
        st.sidebar.success("模型加载成功！")
    except Exception as e:
        st.sidebar.error(f"模型加载失败: {e}")
        return


    # 单页面输入区域
    st.header("患者指标录入")
    col1, col2, col3 = st.columns(3)

    # 输入控件
    with col1:
        icu_admission = st.selectbox("再入ICU", options=[0, 1], format_func=lambda x: "是" if x == 1 else "否", index=0)
        hypertension = st.selectbox("高血压", options=[0, 1], format_func=lambda x: "有" if x == 1 else "无", index=0)
        k = st.number_input("钾（mmol/L）", value=4.0, step=0.1, min_value=1.5, max_value=8.0)

    with col2:
        # 镁分组：按用户要求显示区间标签，但内部编码使用 1/2/3
        mg_label_map = {1: '1.7-2', 2: '≤1.7', 3: '>2'}
        magnesium_group_label = st.selectbox("镁分组", options=[1, 2, 3], format_func=lambda x: mg_label_map.get(x, str(x)), index=0)
        aptt = st.number_input("APTT（秒）", value=30.0, step=0.1, min_value=10.0, max_value=100.0)
        bun = st.number_input("血尿素氮（mmol/L）", value=5.0, step=0.1)

    with col3:
        hb = st.number_input("血红蛋白（g/L）", value=120.0, step=1.0)
        alp = st.number_input("碱性磷酸酶（U/L）", value=80.0, step=1.0)
        mg_cv = st.number_input("镁变异系数", value=5.0, step=0.1)

    # 预测按钮
    predict_button = st.button("开始预测", type="primary")

    if predict_button:
        # 根据模型的特征顺序构建输入DataFrame
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

        if model_feature_names:
            # 使用模型自带顺序，构建严格对齐的输入
            # 将“模型中的名字”映射到“页面输入键名”
            alias_to_user_key = {
                'APTT_.s.': 'APTT_.s',
                'APTT_.s': 'APTT_.s',
                'aptt_.s': 'APTT_.s',
                'APTT_s': 'APTT_.s',
                'APTT.s': 'APTT_.s',
                'magnesium_group': 'Magnesium_group',
                'ICU_admission': 'icu_admission',
                'icu_Admission': 'icu_admission',
            }

            resolved_values = []
            missing_features = []
            for c in model_feature_names:
                ui_key = alias_to_user_key.get(c, c)
                val = user_inputs.get(ui_key, None)
                if val is None:
                    missing_features.append(c)
                resolved_values.append(val)

            if missing_features:
                st.error(f"以下模型特征未在页面录入或名称不匹配：{missing_features}。\n请核对特征名（尤其是 APTT_.s），必要时告知我pkl中的精确特征名。")
                # 调试信息
                with st.expander("调试信息：模型与输入特征名对比"):
                    st.write("模型特征名：", model_feature_names)
                    st.write("页面输入键：", list(user_inputs.keys()))
                return

            input_df = pd.DataFrame([resolved_values], columns=model_feature_names)
        else:
            # 回退为我们定义的9个特征顺序
            ordered_cols = feature_names_display
            input_df = pd.DataFrame([[user_inputs[c] for c in ordered_cols]], columns=ordered_cols)

        # 简单检查缺失
        if input_df.isnull().any().any():
            st.error("存在缺失的输入值，请完善后重试。")
            return

        # 进行预测（概率）
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)[0]
                # 假设第1列为阴性（未AKI），第2列为阳性（AKI）
                if len(proba) == 2:
                    no_aki_prob = float(proba[0])
                    aki_prob = float(proba[1])
                else:
                    # 二分类但返回异常长度，回退为decision_function或predict
                    raise ValueError("predict_proba返回的维度异常")
            else:
                # 没有predict_proba，尝试decision_function或predict
                if hasattr(model, 'decision_function'):
                    score = float(model.decision_function(input_df))
                    # 将score映射到(0,1)（logit近似），仅作退路
                    aki_prob = 1 / (1 + np.exp(-score))
                    no_aki_prob = 1 - aki_prob
                else:
                    pred = int(model.predict(input_df)[0])
                    aki_prob = float(pred)
                    no_aki_prob = 1 - aki_prob

            # 显示预测结果
            st.header("AKI风险预测结果")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("未发生AKI概率")
                st.progress(no_aki_prob)
                st.write(f"{no_aki_prob:.2%}")
            with col2:
                st.subheader("AKI发生概率")
                st.progress(aki_prob)
                st.write(f"{aki_prob:.2%}")

            # 风险分层
            risk_level = "低风险" if aki_prob < 0.3 else ("中等风险" if aki_prob < 0.7 else "高风险")
            risk_color = "green" if aki_prob < 0.3 else ("orange" if aki_prob < 0.7 else "red")
            st.markdown(f"### AKI风险评估: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)

            # 模型解释
            st.write("---")
            st.subheader("模型解释（SHAP）")
            # ===== SHAP 解释开始 =====
            try:
                # 创建SHAP解释器
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)

                # 处理SHAP值格式（兼容二分类的不同返回形式）
                if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                    shap_value = shap_values[0, :, 1]
                    expected_value = explainer.expected_value[1]
                elif isinstance(shap_values, list):
                    shap_value = np.array(shap_values[1][0])
                    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                else:
                    shap_value = np.array(shap_values[0])
                    expected_value = explainer.expected_value

                # 仅保留瀑布图和力图
                current_features = list(input_df.columns)

                # SHAP瀑布图
                st.subheader("SHAP瀑布图")
                try:
                    # 创建SHAP瀑布图
                    import matplotlib.font_manager as fm

                    # 尝试设置中文字体
                    try:
                        # 尝试使用系统中文字体（包含Linux云端服务器常用字体）
                        chinese_fonts = [
                            'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
                            'Source Han Sans SC', 'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB'
                        ]
                        available_fonts = [f.name for f in fm.fontManager.ttflist]

                        local_chinese_font = None
                        for font in chinese_fonts:
                            if font in available_fonts:
                                local_chinese_font = font
                                break

                        if local_chinese_font:
                            plt.rcParams['font.sans-serif'] = [local_chinese_font, 'DejaVu Sans']
                            plt.rcParams['font.family'] = 'sans-serif'
                        else:
                            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
                            plt.rcParams['font.family'] = 'sans-serif'

                    except Exception:
                        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
                        plt.rcParams['font.family'] = 'sans-serif'

                    plt.rcParams['axes.unicode_minus'] = False

                    fig_waterfall = plt.figure(figsize=(12, 8))

                    # 创建英文特征名（兜底）
                    english_names = [f for f in current_features]

                    # 将类别/二值特征的取值映射为中文显示（用于左侧标签的“值 = 特征名”）
                    display_data = input_df.iloc[0].copy()
                    try:
                        if 'icu_admission' in display_data.index:
                            display_data['icu_admission'] = {0: '否', 1: '是'}.get(int(display_data['icu_admission']), display_data['icu_admission'])
                        if 'Hypertension' in display_data.index:
                            display_data['Hypertension'] = {0: '无', 1: '有'}.get(int(display_data['Hypertension']), display_data['Hypertension'])
                        if 'Magnesium_group' in display_data.index:
                            display_data['Magnesium_group'] = {1: '1.7-2', 2: '≤1.7', 3: '>2'}.get(int(display_data['Magnesium_group']), display_data['Magnesium_group'])
                    except Exception:
                        pass

                    # 尝试使用中文特征名，如果失败则使用英文
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
                        st.warning("中文特征名显示失败，使用英文特征名")
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=shap_value,
                                base_values=expected_value,
                                data=display_data.values,
                                feature_names=english_names
                            ),
                            max_display=len(current_features),
                            show=False
                        )

                    # 修复瀑布图中的 Unicode 负号（\u2212）为 ASCII 负号（-），并强制使用中文字体
                    for ax in fig_waterfall.get_axes():
                        # 文本对象（条形标签、注释等）
                        for text in ax.texts:
                            s = text.get_text()
                            if '−' in s:
                                text.set_text(s.replace('−', '-'))
                            if chinese_font:
                                text.set_fontfamily(chinese_font)
                        # y 轴刻度（特征名称处）
                        for label in ax.get_yticklabels():
                            t = label.get_text()
                            if '−' in t:
                                label.set_text(t.replace('−', '-'))
                            if chinese_font:
                                label.set_fontfamily(chinese_font)
                        # x 轴刻度（数值刻度）
                        for label in ax.get_xticklabels():
                            t = label.get_text()
                            if '−' in t:
                                label.set_text(t.replace('−', '-'))
                            if chinese_font:
                                label.set_fontfamily(chinese_font)
                        # 轴标题/图标题
                        if chinese_font:
                            ax.set_xlabel(ax.get_xlabel(), fontfamily=chinese_font)
                            ax.set_ylabel(ax.get_ylabel(), fontfamily=chinese_font)
                            ax.set_title(ax.get_title(), fontfamily=chinese_font)

                    plt.tight_layout()
                    st.pyplot(fig_waterfall)
                    plt.close(fig_waterfall)

                except Exception as e:
                    st.error(f"无法生成瀑布图: {str(e)}")

                # SHAP力图
                st.subheader("SHAP力图")
                try:
                    # 使用官方SHAP力图，HTML格式
                    import streamlit.components.v1 as components
                    import matplotlib

                    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
                    matplotlib.rcParams['axes.unicode_minus'] = False

                    # 力图也使用映射后的显示数据，避免出现 0/1 等数值
                    display_data_fp = input_df.iloc[0].copy()
                    try:
                        if 'icu_admission' in display_data_fp.index:
                            display_data_fp['icu_admission'] = {0: '否', 1: '是'}.get(int(display_data_fp['icu_admission']), display_data_fp['icu_admission'])
                        if 'Hypertension' in display_data_fp.index:
                            display_data_fp['Hypertension'] = {0: '无', 1: '有'}.get(int(display_data_fp['Hypertension']), display_data_fp['Hypertension'])
                        if 'Magnesium_group' in display_data_fp.index:
                            display_data_fp['Magnesium_group'] = {1: '1.7-2', 2: '≤1.7', 3: '>2'}.get(int(display_data_fp['Magnesium_group']), display_data_fp['Magnesium_group'])
                    except Exception:
                        pass

                    force_plot = shap.force_plot(
                        expected_value,
                        shap_value,
                        display_data_fp,
                        feature_names=[feature_dict.get(f, f) for f in current_features]
                    )

                    shap_html = f"""
                    <head>
                        {shap.getjs()}
                        <style>
                            body {{ margin: 0; padding: 20px 10px 40px 10px; overflow: visible; }}
                            .force-plot {{ margin: 20px 0 40px 0 !important; padding: 20px 0 40px 0 !important; }}
                            svg {{ margin: 20px 0 40px 0 !important; }}
                            .tick text {{ margin-bottom: 20px !important; }}
                            .force-plot-container {{ min-height: 200px !important; padding-bottom: 50px !important; }}
                        </style>
                    </head>
                    <body>
                        <div class="force-plot-container">{force_plot.html()}</div>
                    </body>
                    """
                    components.html(shap_html, height=400, scrolling=False)
                except Exception as e:
                    st.error(f"无法生成HTML力图: {str(e)}")
                    st.info("请检查SHAP版本是否兼容")

            except Exception as e:
                st.error(f"无法生成SHAP解释: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

        except Exception as e:
            st.error(f"预测或结果展示失败: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

    # 版权或说明
    st.write("---")
    st.caption("")

if __name__ == "__main__":
    main()
