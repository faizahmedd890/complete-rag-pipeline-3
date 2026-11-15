import os

from dotenv import load_dotenv
from supabase import create_client


load_dotenv()


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is missing")

if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY is missing")


supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


def sign_up(
    email: str,
    password: str
):
    response = supabase.auth.sign_up({
        "email": email,
        "password": password
    })

    return response


def sign_in(
    email: str,
    password: str
):
    response = supabase.auth.sign_in_with_password({
        "email": email,
        "password": password
    })

    return response


def sign_out():
    supabase.auth.sign_out()


def get_user():
    try:
        return supabase.auth.get_user()
    except Exception:
        return None