
import sys
from pathlib import Path

import streamlit as st


sys.path.append(
    str(Path(__file__).resolve().parents[1])
)


from api import (
    create_session,
    get_sessions,
    delete_session,
    send_message
)

from state import initialize_state


st.set_page_config(
    page_title="NovaAI Chat",
    page_icon="💬",
    layout="wide"
)


initialize_state()


if not st.session_state.authenticated:
    st.warning("Please login first.")
    st.stop()


def main():

    user = st.session_state.user

    user_id = str(user.id)

    chats = get_sessions(user_id)

    chat_ids = {
        chat["id"]
        for chat in chats
    }

    if (
        st.session_state.chat_id
        and st.session_state.chat_id not in chat_ids
    ):
        st.session_state.chat_id = None

    # ==========================================
    # SIDEBAR
    # ==========================================

    with st.sidebar:

        st.title("NovaAI")

        if st.button(
            "＋ New Chat",
            use_container_width=True
        ):
            st.session_state.chat_id = None
            st.rerun()

        st.divider()

        for chat in chats:

            title = (
                chat.get("title")
                or "New Chat"
            )

            if len(title) > 50:
                title = title[:47] + "..."

            col1, col2 = st.columns([5, 1])

            with col1:

                if st.button(
                    title,
                    key=f"open_{chat['id']}",
                    use_container_width=True
                ):
                    st.session_state.chat_id = (
                        chat["id"]
                    )

                    st.rerun()

            with col2:

                if st.button(
                    "🗑️",
                    key=f"delete_{chat['id']}"
                ):

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

    # ==========================================
    # NO SESSION
    # ==========================================

    if st.session_state.chat_id is None:

        st.markdown(
            "### Start a new conversation"
        )

        st.caption(
            "Ask NovaAI a question."
        )

    # ==========================================
    # CURRENT SESSION
    # ==========================================

    else:

        current_chat = next(
            (
                chat
                for chat in chats
                if chat["id"]
                == st.session_state.chat_id
            ),
            None
        )

        if current_chat:

            st.markdown(
                f"### {current_chat.get('title', 'Chat')}"
            )

    # ==========================================
    # DISPLAY CURRENT MESSAGES
    # ==========================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

    # ==========================================
    # CHAT INPUT
    # ==========================================

    question = st.chat_input(
        "Ask NovaAI..."
    )

    if question:

        question = question.strip()

        if not question:

            st.warning(
                "Question cannot be empty."
            )

            st.stop()

        # ======================================
        # CREATE SESSION
        # ======================================

        if st.session_state.chat_id is None:

            session = create_session(
                user_id
            )

            st.session_state.chat_id = (
                session["id"]
            )

        # ======================================
        # SAVE USER MESSAGE LOCALLY
        # ======================================

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # ======================================
        # DISPLAY USER
        # ======================================

        with st.chat_message("user"):

            st.markdown(question)

        # ======================================
        # SEND TO FASTAPI
        # ======================================

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    result = send_message(
                        question=question,
                        user_id=user_id,
                        chat_id=(
                            st.session_state.chat_id
                        )
                    )

                    if result.get("error"):

                        st.error(
                            result["error"]
                        )

                    else:

                        answer = result.get(
                            "answer",
                            "No answer returned."
                        )

                        st.markdown(answer)

                        if result.get(
                            "cache_hit",
                            False
                        ):
                            st.caption(
                                "⚡ Answer from cache"
                            )

                        latency = result.get(
                            "latency_ms"
                        )

                        if latency is not None:

                            st.caption(
                                f"Response time: "
                                f"{latency:.0f} ms"
                            )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })

                except Exception as exc:

                    st.error(
                        f"Error: {exc}"
                    )

        st.rerun()


if __name__ == "__main__":
    main()

