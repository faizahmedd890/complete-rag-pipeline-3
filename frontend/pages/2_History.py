
import streamlit as st

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from api import (
    get_sessions,
    delete_session
)

from state import initialize_state


st.set_page_config(
    page_title="NovaAI History",
    page_icon="🕘",
    layout="wide"
)


initialize_state()


if not st.session_state.authenticated:
    st.warning("Please login first.")
    st.stop()


user = st.session_state.user

user_id = str(user.id)


st.title("🕘 Chat History")


try:

    chats = get_sessions(user_id)

except Exception as exc:

    st.error(
        f"Failed to load chat history: {exc}"
    )

    st.stop()


if not chats:

    st.info(
        "No chat history found."
    )

else:

    for chat in chats:

        title = (
            chat.get("title")
            or "New Chat"
        )

        updated_at = (
            chat.get("updated_at")
            or ""
        )

        col1, col2 = st.columns(
            [8, 1]
        )

        with col1:

            st.markdown(
                f"### {title}"
            )

            if updated_at:

                st.caption(
                    f"Last activity: {updated_at}"
                )

        with col2:

            if st.button(
                "🗑️",
                key=f"delete_{chat['id']}"
            ):

                try:

                    delete_session(
                        chat["id"],
                        user_id
                    )

                    if (
                        st.session_state.chat_id
                        == chat["id"]
                    ):
                        st.session_state.chat_id = None
                        st.session_state.messages = []

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Failed to delete chat: {exc}"
                    )

        st.divider()

