import sqlite3
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = 'ruralminds.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Users Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            name TEXT,
            email TEXT
        )
    ''')
    
    # 2. Forum Posts Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS forum_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            user_role TEXT,
            title TEXT,
            content TEXT,
            category TEXT,
            related_document TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            status TEXT,
            upvotes INTEGER DEFAULT 0
        )
    ''')
    
    # 3. Forum Replies Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS forum_replies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER,
            username TEXT,
            user_role TEXT,
            content TEXT,
            created_at TIMESTAMP,
            is_answer BOOLEAN,
            FOREIGN KEY(post_id) REFERENCES forum_posts(id) ON DELETE CASCADE
        )
    ''')
    
    # 4. Documents Table for PDF tracking
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            upload_path TEXT,
            uploaded_by TEXT,
            uploaded_at TIMESTAMP
        )
    ''')
    
    # 5. Videos Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            name TEXT,
            video_path TEXT,
            caption_path TEXT,
            has_captions BOOLEAN DEFAULT 0,
            uploaded_by TEXT,
            uploaded_at TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def _migrate_users():
    if os.path.exists("users_db.json"):
        logger.info("Migrating users_db.json to SQLite database...")
        conn = get_db_connection()
        c = conn.cursor()
        
        try:
            with open("users_db.json", "r") as f:
                users_data = json.load(f)
            
            for username, data in users_data.items():
                c.execute('''
                    INSERT OR IGNORE INTO users (username, password, role, name, email)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, data.get('password'), data.get('role'), data.get('name'), data.get('email')))
            
            conn.commit()
            
            # Rename the file so it doesn't migrate again
            os.rename("users_db.json", "users_db.json.bak")
            logger.info("Migration successful: users_db.json.bak created.")
        except Exception as e:
            logger.error(f"Failed to migrate users: {e}")
        finally:
            conn.close()

def _migrate_forum():
    if os.path.exists("forum_db.json"):
        logger.info("Migrating forum_db.json to SQLite database...")
        conn = get_db_connection()
        c = conn.cursor()
        
        try:
            with open("forum_db.json", "r", encoding='utf-8') as f:
                forum_data = json.load(f)
            
            posts = forum_data.get('posts', [])
            for post in posts:
                # Insert post 
                # (keep original ID if possible, but SQLite autoincrement might ignore it if we just provide it. 
                # Doing it safely by assigning it)
                c.execute('''
                    INSERT OR IGNORE INTO forum_posts 
                    (id, username, user_role, title, content, category, related_document, created_at, updated_at, status, upvotes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    post['id'], post['username'], post.get('user_role', 'student'), 
                    post['title'], post['content'], post.get('category', 'General'), 
                    post.get('related_document'), post.get('created_at', datetime.now().isoformat()), 
                    post.get('updated_at', datetime.now().isoformat()), post.get('status', 'open'), 
                    post.get('upvotes', 0)
                ))
                
                # Insert replies
                for reply in post.get('replies', []):
                    c.execute('''
                        INSERT INTO forum_replies
                        (post_id, username, user_role, content, created_at, is_answer)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        post['id'], reply['username'], reply.get('user_role', 'student'), 
                        reply['content'], reply.get('created_at', datetime.now().isoformat()), 
                        reply.get('is_answer', False)
                    ))
            
            conn.commit()
            
            # Rename the file
            os.rename("forum_db.json", "forum_db.json.bak")
            logger.info("Migration successful: forum_db.json.bak created.")
        except Exception as e:
            logger.error(f"Failed to migrate forum: {e}")
        finally:
            conn.close()

def ensure_migrated():
    """Run all schema setups and migrations securely."""
    init_db()
    _migrate_users()
    _migrate_forum()
