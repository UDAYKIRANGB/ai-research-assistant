"""
Conversation memory.

Implements a SQL-backed ConversationBufferMemory equivalent: every user/
assistant turn is persisted to `chat_messages`, and the session's
`last_active_doc_id` is updated whenever a question clearly references a
single document. This lets follow-up questions like "What are its
limitations?" resolve "its" to the document discussed in the previous turn
without the user repeating the document name.
"""
from typing import List, Optional

from sqlalchemy.orm import Session

from src.database.models import ChatMessage, ChatSession

# Simple pronoun list used to detect follow-up references to "the last document".
FOLLOWUP_PRONOUNS = {"it", "its", "it's", "that", "this", "the document", "the paper"}


class ConversationMemory:
    def __init__(self, db: Session, session_id: str):
        self.db = db
        self.session_id = session_id
        self.session = self._get_or_create_session()

    def _get_or_create_session(self) -> ChatSession:
        session = self.db.query(ChatSession).filter_by(session_id=self.session_id).first()
        if not session:
            session = ChatSession(session_id=self.session_id)
            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)
        return session

    def get_history_text(self, max_turns: int = 6) -> str:
        messages: List[ChatMessage] = (
            self.db.query(ChatMessage)
            .filter_by(session_id=self.session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(max_turns)
            .all()
        )
        messages.reverse()
        lines = [f"{m.role.capitalize()}: {m.content}" for m in messages]
        return "\n".join(lines)

    def add_turn(self, role: str, content: str) -> None:
        self.db.add(ChatMessage(session_id=self.session_id, role=role, content=content))
        self.db.commit()

    def resolve_doc_ids(self, question: str, explicit_doc_ids: Optional[List[str]]) -> Optional[List[str]]:
        """If the caller passed explicit doc_ids, use them. Otherwise, if the
        question looks like a pronoun follow-up ("its limitations?") and we
        have a last-active document for this session, scope retrieval to it."""
        if explicit_doc_ids:
            self.set_active_doc(explicit_doc_ids[0])
            return explicit_doc_ids

        lowered = question.lower()
        looks_like_followup = any(p in lowered.split() or p in lowered for p in FOLLOWUP_PRONOUNS)
        if looks_like_followup and self.session.last_active_doc_id:
            return [self.session.last_active_doc_id]
        return None

    def set_active_doc(self, doc_id: str) -> None:
        self.session.last_active_doc_id = doc_id
        self.db.add(self.session)
        self.db.commit()
