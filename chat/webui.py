import streamlit as st
import streamlit_antd_components as sac

import __init__
from server.utils import api_address
from webui_pages.kb_chat import kb_chat
from webui_pages.knowledge_base.knowledge_base import knowledge_base_page
from webui_pages.utils import *

api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    st.set_page_config(
        "Biology Chat WebUI",
        get_img_base64("chatchat_icon_blue_square_v2.png"),
        initial_sidebar_state="expanded",
        layout="centered",
    )

    st.markdown(
        """
        <style>
        [data-testid="stSidebarUserContent"] {
            padding-top: 20px;
        }
        .block-container {
            padding-top: 25px;
        }
        [data-testid="stBottomBlockContainer"] {
            padding-bottom: 20px;
        }
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.image(
            get_img_base64("logo-long-chatchat-trans-v2.png"), use_column_width=True
        )
        st.caption(
            f"""<p align="right">当前版本：{__init__.__version__}</p>""",
            unsafe_allow_html=True,
        )

        selected_page = sac.menu(
            [
                sac.MenuItem("RAG 对话", icon="database"),
                sac.MenuItem("知识库管理", icon="hdd-stack"),
            ],
            key="selected_page",
            open_index=0,
        )

        sac.divider()

    if selected_page == "知识库管理":
        knowledge_base_page(api=api)
    else:
        kb_chat(api=api)
