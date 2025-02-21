import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from modules.database.session import AsyncSessionLocal
from modules.models.user import User
from modules.middleware.auth import get_password_hash
from typing import List

async def create_user(
    username: str,
    email: str,
    password: str,
    roles: List[str] = ["user"],
    is_active: bool = True,
    is_superuser: bool = False
) -> User:
    """Create a new user"""
    try:
        async with AsyncSessionLocal() as session:
            # Check if user exists
            result = await session.execute(
                f"SELECT * FROM users WHERE username = '{username}' OR email = '{email}'"
            )
            existing_user = result.first()
            
            if existing_user:
                raise ValueError(f"User with username {username} or email {email} already exists")
            
            # Create new user
            user = User(
                username=username,
                email=email,
                hashed_password=get_password_hash(password),
                roles=roles,
                is_active=is_active,
                is_superuser=is_superuser
            )
            session.add(user)
            await session.commit()
            print(f"User {username} created successfully!")
            return user
                
    except Exception as e:
        print(f"Error creating user: {e}")
        raise

# Example usage
async def create_example_users():
    # Create a trader
    await create_user(
        username="trader1",
        email="trader1@company.com",
        password="secure_trader_password",
        roles=["trader"]
    )
    
    # Create an analyst
    await create_user(
        username="analyst1",
        email="analyst1@company.com",
        password="secure_analyst_password",
        roles=["analyst"]
    )
    
    # Create a manager with multiple roles
    await create_user(
        username="manager1",
        email="manager1@company.com",
        password="secure_manager_password",
        roles=["trader", "analyst", "manager"]
    )

async def update_user(
    username: str,
    new_email: str = None,
    new_roles: List[str] = None,
    is_active: bool = None,
    is_superuser: bool = None
) -> User:
    """Update existing user"""
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                f"SELECT * FROM users WHERE username = '{username}'"
            )
            user = result.scalar_one()
            
            if new_email:
                user.email = new_email
            if new_roles:
                user.roles = new_roles
            if is_active is not None:
                user.is_active = is_active
            if is_superuser is not None:
                user.is_superuser = is_superuser
                
            await session.commit()
            print(f"User {username} updated successfully!")
            return user
    except Exception as e:
        print(f"Error updating user: {e}")
        raise

async def delete_user(username: str) -> bool:
    """Delete a user"""
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                f"SELECT * FROM users WHERE username = '{username}'"
            )
            user = result.scalar_one()
            
            await session.delete(user)
            await session.commit()
            print(f"User {username} deleted successfully!")
            return True
    except Exception as e:
        print(f"Error deleting user: {e}")
        return False

async def reset_password(username: str, new_password: str) -> bool:
    """Reset user password"""
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                f"SELECT * FROM users WHERE username = '{username}'"
            )
            user = result.scalar_one()
            
            user.hashed_password = get_password_hash(new_password)
            await session.commit()
            print(f"Password reset for {username} successful!")
            return True
    except Exception as e:
        print(f"Error resetting password: {e}")
        return False

if __name__ == "__main__":
    # Create individual user
    asyncio.run(create_user(
        username="new_user",
        email="new_user@company.com",
        password="secure_password",
        roles=["user"]
    ))
    
    # Or create example users
    # asyncio.run(create_example_users()) 