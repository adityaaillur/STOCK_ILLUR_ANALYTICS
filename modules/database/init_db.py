import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from modules.database.session import AsyncSessionLocal
from modules.models.user import User
from modules.middleware.auth import get_password_hash
from modules.config import settings

async def init_db():
    try:
        async with AsyncSessionLocal() as session:
            # Check if admin exists
            result = await session.execute(
                "SELECT * FROM users WHERE username = 'admin'"
            )
            admin = result.first()
            
            if not admin:
                # Create admin user
                admin = User(
                    username="admin",
                    email="admin@yourdomain.com",
                    hashed_password=get_password_hash("your-secure-admin-password"),
                    is_active=True,
                    is_superuser=True,
                    roles=["admin"]
                )
                session.add(admin)
                await session.commit()
                print("Admin user created successfully!")
            else:
                print("Admin user already exists")
                
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(init_db()) 