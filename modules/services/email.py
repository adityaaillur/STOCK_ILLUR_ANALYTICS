class EmailService:
    async def send_password_reset_email(self, email: str, reset_link: str) -> bool:
        """Send password reset email to user"""
        try:
            # Implement email sending logic here
            return True
        except Exception:
            return False 