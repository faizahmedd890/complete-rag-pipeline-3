import streamlit as st

from state import initialize_state
from supabase_client import (
    sign_in,
    sign_up,
    sign_out
)


st.set_page_config(
    page_title="NovaAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


initialize_state()


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .nova-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .nova-subtitle {
        color: #777;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .user-box {
        padding: 12px;
        border-radius: 10px;
        background-color: rgba(128, 128, 128, 0.08);
        margin-bottom: 15px;
    }

    .stChatMessage {
        border-radius: 12px;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def authentication_page():

    st.markdown(
        '<div class="nova-title">NovaAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="nova-subtitle">'
        'AI-powered enterprise knowledge assistant'
        '</div>',
        unsafe_allow_html=True
    )

    login_tab, signup_tab = st.tabs(
        ["Login", "Create account"]
    )

    with login_tab:

        email = st.text_input(
            "Email",
            key="login_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button(
            "Login",
            type="primary",
            use_container_width=True
        ):

            if not email or not password:
                st.error(
                    "Please enter email and password."
                )
                return

            try:
                response = sign_in(
                    email,
                    password
                )

                if response.user:
                    st.session_state.authenticated = True
                    st.session_state.user = response.user

                    st.success("Login successful.")

                    st.rerun()

            except Exception as exc:
                st.error(
                    f"Login failed: {exc}"
                )

    with signup_tab:

        email = st.text_input(
            "Email",
            key="signup_email"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="signup_password"
        )

        confirm_password = st.text_input(
            "Confirm password",
            type="password",
            key="signup_confirm_password"
        )

        if st.button(
            "Create account",
            type="primary",
            use_container_width=True
        ):

            if not email or not password:
                st.error(
                    "Please fill all fields."
                )
                return

            if password != confirm_password:
                st.error(
                    "Passwords do not match."
                )
                return

            if len(password) < 6:
                st.error(
                    "Password must contain at least 6 characters."
                )
                return

            try:
                response = sign_up(
                    email,
                    password
                )

                if response.user:
                    st.success(
                        "Account created. "
                        "Check your email if confirmation is required."
                    )

            except Exception as exc:
                st.error(
                    f"Signup failed: {exc}"
                )


def sidebar():

    with st.sidebar:

        st.markdown(
            "## 🤖 NovaAI"
        )

        st.divider()

        user = st.session_state.user

        if user:

            email = getattr(
                user,
                "email",
                "User"
            )

            st.markdown(
                f"""
                <div class="user-box">
                <b>Signed in as</b><br>
                {email}
                </div>
                """,
                unsafe_allow_html=True
            )

        if st.button(
            "➕ New Chat",
            use_container_width=True
        ):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.rerun()

        st.page_link(
            "pages/1_Chat.py",
            label="💬 Chat"
        )

        st.page_link(
            "pages/2_History.py",
            label="🕘 History"
        )

        st.divider()

        if st.button(
            "Logout",
            use_container_width=True
        ):

            try:
                sign_out()
            except Exception:
                pass

            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.session_id = None

            st.rerun()


if not st.session_state.authenticated:
    authentication_page()
else:
    sidebar()

    st.markdown(
        "# Welcome to NovaAI"
    )

    st.markdown(
        "Open **Chat** from the sidebar to start."
    )