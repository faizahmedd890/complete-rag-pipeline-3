

import os
import requests

BASE_URL = os.getenv("BACKEND_URL")


def create_session(user_id):
    response = requests.post(
        f"{BASE_URL}/api/sessions",
        json={
            "user_id": user_id
        },
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def get_sessions(user_id):
    response = requests.get(
        f"{BASE_URL}/api/sessions/{user_id}",
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def delete_session(
    chat_id,
    user_id
):
    response = requests.delete(
        f"{BASE_URL}/api/sessions/"
        f"{chat_id}/{user_id}",
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def send_message(
    question,
    user_id,
    chat_id
):
    response = requests.post(
        f"{BASE_URL}/api/chat",
        json={
            "question": question,
            "user_id": user_id,
            "chat_id": chat_id
        },
        timeout=120
    )

    if response.status_code == 400:
        detail = response.json().get(
            "detail",
            "Invalid request."
        )

        return {
            "error": detail
        }

    if response.status_code == 404:
        detail = response.json().get(
            "detail",
            "Session not found."
        )

        return {
            "error": detail
        }

    response.raise_for_status()

    return response.json()

