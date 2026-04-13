import hashlib
import logging
from typing import Optional, Tuple
from database import get_db_connection, ensure_migrated

logger = logging.getLogger(__name__)

# Ensure migrations on module load just in case (safe due to IF NOT EXISTS and .bak)
ensure_migrated()

def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def _ensure_admin():
    """Ensure the default administrator exists in the SQLite database."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT username FROM users WHERE username = ?', ('admin',))
    if not c.fetchone():
        c.execute('''
            INSERT INTO users (username, password, role, name, email) 
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', hash_password("administrator"), 'admin', 'Administrator', 'admin@edubridge.local'))
        conn.commit()
    conn.close()

# Keep admin check upon startup
_ensure_admin()

def create_user(username: str, password: str, role: str, name: str, email: str) -> Tuple[bool, str]:
    """
    Create a new user account.
    """
    if role not in ['teacher', 'student', 'admin']:
        return False, "Invalid role. Must be 'teacher', 'student', or 'admin'."
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters long."
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT username FROM users WHERE username = ?', (username,))
    if c.fetchone():
        conn.close()
        return False, "Username already exists."
    
    try:
        c.execute('''
            INSERT INTO users (username, password, role, name, email)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, hash_password(password), role, name, email))
        conn.commit()
        success = True
        msg = "Account created successfully!"
    except Exception as e:
        logger.error(f"Error saving user data: {str(e)}")
        success = False
        msg = "Error saving user data."
    finally:
        conn.close()
    
    return success, msg

def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[dict]]:
    """
    Authenticate a user via SQLite.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    
    if not user:
        logger.warning(f"Login attempt with non-existent username: {username}")
        return False, None
    
    hashed_input = hash_password(password)
    
    if user['password'] == hashed_input:
        logger.info(f"Successful login: {username} ({user['role']})")
        return True, {
            'username': user['username'],
            'role': user['role'],
            'name': user['name'],
            'email': user['email']
        }
    
    logger.warning(f"Failed login attempt for user: {username}")
    return False, None

def get_user_role(username: str) -> Optional[str]:
    """Get the role of a user."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    
    if row:
        return row['role']
    return None

def change_password(username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
    """Change user password."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    
    if not user:
        conn.close()
        return False, "User not found."
    
    if user['password'] != hash_password(old_password):
        conn.close()
        return False, "Incorrect old password."
    
    if len(new_password) < 6:
        conn.close()
        return False, "New password must be at least 6 characters long."
    
    try:
        c.execute('UPDATE users SET password = ? WHERE username = ?', (hash_password(new_password), username))
        conn.commit()
        success = True
        msg = "Password changed successfully!"
    except Exception as e:
        logger.error(f"Error saving changes: {str(e)}")
        success = False
        msg = "Error saving changes."
    finally:
        conn.close()
        
    return success, msg

def get_all_users() -> list:
    """Get list of all users (admin only)."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT username, role, name, email FROM users')
    rows = c.fetchall()
    conn.close()
    
    return [
        {
            'username': r['username'],
            'role': r['role'],
            'name': r['name'],
            'email': r['email']
        } for r in rows
    ]

def delete_user(username: str) -> Tuple[bool, str]:
    """Delete a user account (admin only)."""
    if username == "admin" or username == "administrator":
        return False, "Cannot delete primary admin account."
        
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT username FROM users WHERE username = ?', (username,))
    if not c.fetchone():
        conn.close()
        return False, "User not found."
    
    try:
        c.execute('DELETE FROM users WHERE username = ?', (username,))
        conn.commit()
        logger.info(f"User deleted: {username}")
        success = True
        msg = f"User '{username}' deleted successfully."
    except Exception as e:
        logger.error(f"Error saving changes: {str(e)}")
        success = False
        msg = "Error saving changes."
    finally:
        conn.close()
        
    return success, msg