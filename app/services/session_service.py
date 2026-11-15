from datetime import datetime, timezone


class SessionService:
    def __init__(self, supabase):
        self.supabase = supabase

    def create(self, user_id, title="New Chat"):
        now = datetime.now(timezone.utc).isoformat()

        response = (
            self.supabase
            .table("chats")
            .insert(
                {
                    "user_id": user_id,
                    "title": title,
                    "created_at": now,
                    "updated_at": now
                }
            )
            .execute()
        )

        if not response.data:
            raise RuntimeError(
                "Failed to create chat session."
            )

        return response.data[0]

    def list_sessions(self, user_id):
        response = (
            self.supabase
            .table("chats")
            .select("id, title, created_at, updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )

        return response.data or []

    def get_session(self, chat_id, user_id):
        response = (
            self.supabase
            .table("chats")
            .select("id, title, created_at, updated_at")
            .eq("id", chat_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )

        if not response.data:
            return None

        return response.data[0]

    def touch(self, chat_id, user_id):
        self.supabase.table("chats").update(
            {
                "updated_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }
        ).eq(
            "id", chat_id
        ).eq(
            "user_id", user_id
        ).execute()

    def update_title(
        self,
        chat_id,
        user_id,
        title
    ):
        title = str(title).strip()

        if not title:
            title = "New Chat"

        if len(title) > 100:
            title = title[:97] + "..."

        self.supabase.table("chats").update(
            {
                "title": title,
                "updated_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }
        ).eq(
            "id", chat_id
        ).eq(
            "user_id", user_id
        ).execute()

    def delete(self, chat_id, user_id):
        session = self.get_session(
            chat_id,
            user_id
        )

        if not session:
            return False

        # Delete memory first
        self.supabase.table(
            "session_memory"
        ).delete().eq(
            "chat_id",
            chat_id
        ).execute()

        # Delete messages
        self.supabase.table(
            "messages"
        ).delete().eq(
            "chat_id",
            chat_id
        ).execute()

        # Delete session
        response = (
            self.supabase
            .table("chats")
            .delete()
            .eq("id", chat_id)
            .eq("user_id", user_id)
            .execute()
        )

        return bool(response.data)

    def cleanup_inactive_sessions(
        self,
        days=7
    ):
        response = (
            self.supabase
            .rpc(
                "delete_inactive_chat_sessions",
                {
                    "inactive_days": days
                }
            )
            .execute()
        )

        return response.data