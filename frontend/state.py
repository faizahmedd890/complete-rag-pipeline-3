
import streamlit as st


def initialize_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "user" not in st.session_state:
        st.session_state.user = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_id" not in st.session_state:
        st.session_state.chat_id = None

    if "page" not in st.session_state:
        st.session_state.page = "Chat"


def login_user(user):
    st.session_state.authenticated = True
    st.session_state.user = user


def logout_user():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.chat_id = None
    st.session_state.messages = []


def clear_chat():
    st.session_state.messages = []
    st.session_state.chat_id = None
