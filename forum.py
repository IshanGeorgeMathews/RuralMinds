import os
from datetime import datetime
from typing import List, Optional, Tuple, Dict
import logging
from database import get_db_connection

logger = logging.getLogger(__name__)

def create_post(
    username: str, 
    user_role: str, 
    title: str, 
    content: str, 
    category: str = "General",
    related_document: Optional[str] = None
) -> Tuple[bool, str, Optional[int]]:
    """Create a new forum post."""
    if not title or not content:
        return False, "Title and content are required.", None
    
    if len(title) < 5:
        return False, "Title must be at least 5 characters long.", None
    
    if len(content) < 10:
        return False, "Content must be at least 10 characters long.", None
    
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO forum_posts (
                username, user_role, title, content, category, 
                related_document, created_at, updated_at, status, upvotes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            username, user_role, title, content, category, 
            related_document, datetime.now().isoformat(), datetime.now().isoformat(), 'open', 0
        ))
        conn.commit()
        post_id = c.lastrowid
        logger.info(f"Post created: ID={post_id}, Title='{title}', User={username}")
        success = True
        msg = "Post created successfully!"
    except Exception as e:
        logger.error(f"Error saving post: {str(e)}")
        success = False
        msg = "Error saving post."
        post_id = None
    finally:
        conn.close()
        
    return success, msg, post_id

def add_reply(
    post_id: int, 
    username: str, 
    user_role: str, 
    content: str,
    is_answer: bool = False
) -> Tuple[bool, str]:
    """Add a reply to a forum post."""
    if not content:
        return False, "Reply content is required."
    
    if len(content) < 5:
        return False, "Reply must be at least 5 characters long."
    
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT id FROM forum_posts WHERE id = ?', (post_id,))
    if not c.fetchone():
        conn.close()
        return False, "Post not found."
    
    try:
        is_ans = is_answer and user_role == "teacher"
        c.execute('''
            INSERT INTO forum_replies (post_id, username, user_role, content, created_at, is_answer)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (post_id, username, user_role, content, datetime.now().isoformat(), is_ans))
        
        # Update post timestamp and status
        status_update = "status = 'answered'," if is_ans else ""
        c.execute(f'''
            UPDATE forum_posts 
            SET {status_update} updated_at = ?
            WHERE id = ?
        ''', (datetime.now().isoformat(), post_id))
        
        conn.commit()
        logger.info(f"Reply added to post {post_id} by {username}")
        success = True
        msg = "Reply added successfully!"
    except Exception as e:
        logger.error(f"Error saving reply: {str(e)}")
        success = False
        msg = "Error saving reply."
    finally:
        conn.close()
        
    return success, msg

def _build_post_dict(c, post_row) -> dict:
    """Helper to shape post and fetch its replies"""
    post = dict(post_row)
    c.execute('SELECT * FROM forum_replies WHERE post_id = ? ORDER BY created_at ASC', (post['id'],))
    post['replies'] = [dict(r) for r in c.fetchall()]
    # SQLite boolean mapping
    for r in post['replies']:
        r['is_answer'] = bool(r['is_answer'])
    return post

def get_all_posts(
    filter_status: Optional[str] = None,
    filter_category: Optional[str] = None,
    sort_by: str = "recent"
) -> List[dict]:
    """Get all forum posts with filtering and sorting."""
    conn = get_db_connection()
    c = conn.cursor()
    
    query = "SELECT * FROM forum_posts WHERE 1=1"
    params = []
    
    if filter_status:
        query += " AND status = ?"
        params.append(filter_status)
        
    if filter_category and filter_category != "All":
        query += " AND category = ?"
        params.append(filter_category)
        
    if sort_by == "recent":
        query += " ORDER BY updated_at DESC"
    elif sort_by == "popular":
        query += " ORDER BY upvotes DESC"
    elif sort_by == "unanswered":
        query += " AND status = 'open' ORDER BY created_at DESC"
        
    c.execute(query, params)
    posts_rows = c.fetchall()
    
    posts = [_build_post_dict(c, row) for row in posts_rows]
    conn.close()
    return posts

def get_post_by_id(post_id: int) -> Optional[dict]:
    """Get a specific post by ID."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM forum_posts WHERE id = ?', (post_id,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        return None
        
    post = _build_post_dict(c, row)
    conn.close()
    return post

def upvote_post(post_id: int) -> Tuple[bool, str]:
    """Upvote a post."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT id FROM forum_posts WHERE id = ?', (post_id,))
    if not c.fetchone():
        conn.close()
        return False, "Post not found."
        
    try:
        c.execute('UPDATE forum_posts SET upvotes = upvotes + 1 WHERE id = ?', (post_id,))
        conn.commit()
        success = True
        msg = "Post upvoted!"
    except Exception as e:
        success = False
        msg = "Error saving upvote."
    finally:
        conn.close()
    return success, msg

def update_post_status(post_id: int, status: str) -> Tuple[bool, str]:
    """Update post status (teacher only)."""
    if status not in ["open", "answered", "closed"]:
        return False, "Invalid status."
    
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT id FROM forum_posts WHERE id = ?', (post_id,))
    if not c.fetchone():
        conn.close()
        return False, "Post not found."
        
    try:
        c.execute('UPDATE forum_posts SET status = ?, updated_at = ? WHERE id = ?', 
                  (status, datetime.now().isoformat(), post_id))
        conn.commit()
        success = True
        msg = f"Post status updated to {status}."
    except:
        success = False
        msg = "Error updating status."
    finally:
        conn.close()
    return success, msg

def delete_post(post_id: int, username: str, user_role: str) -> Tuple[bool, str]:
    """Delete a post (only by post owner or teacher)."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT username FROM forum_posts WHERE id = ?', (post_id,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        return False, "Post not found."
        
    if row['username'] == username or user_role == "teacher":
        try:
            c.execute('DELETE FROM forum_posts WHERE id = ?', (post_id,))
            conn.commit()
            logger.info(f"Post {post_id} deleted by {username}")
            success = True
            msg = "Post deleted successfully."
        except:
            success = False
            msg = "Error deleting post."
    else:
        success = False
        msg = "You don't have permission to delete this post."
        
    conn.close()
    return success, msg

def get_pending_posts_count() -> int:
    """Get count of posts without teacher replies (for notifications)."""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Needs a post that is open, and has no replies where user_role == 'teacher'
    c.execute('''
        SELECT COUNT(*) as count 
        FROM forum_posts p
        WHERE p.status = 'open' 
        AND NOT EXISTS (
            SELECT 1 FROM forum_replies r 
            WHERE r.post_id = p.id AND r.user_role = 'teacher'
        )
    ''')
    row = c.fetchone()
    conn.close()
    return row['count'] if row else 0

def get_forum_stats() -> Dict:
    """Get forum statistics."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT 
            COUNT(*) as total_posts,
            SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_posts,
            SUM(CASE WHEN status = 'answered' THEN 1 ELSE 0 END) as answered_posts
        FROM forum_posts
    ''')
    p_stats = c.fetchone()
    
    c.execute('SELECT COUNT(*) as total_replies FROM forum_replies')
    r_stats = c.fetchone()
    
    conn.close()
    
    total_posts = p_stats['total_posts'] or 0
    open_posts = p_stats['open_posts'] or 0
    answered_posts = p_stats['answered_posts'] or 0
    total_replies = r_stats['total_replies'] or 0
    
    return {
        "total_posts": total_posts,
        "open_posts": open_posts,
        "answered_posts": answered_posts,
        "closed_posts": total_posts - open_posts - answered_posts,
        "total_replies": total_replies,
        "pending_posts": get_pending_posts_count()
    }

def get_categories() -> List[str]:
    """Get list of all categories."""
    default_categories = [
        "All", 
        "General", 
        "PDF Questions", 
        "Video Questions", 
        "Technical Help", 
        "Study Tips", 
        "Other"
    ]
    return default_categories

def search_posts(query: str) -> List[dict]:
    """Search posts by title or content."""
    conn = get_db_connection()
    c = conn.cursor()
    
    q = f"%{query.lower()}%"
    c.execute('''
        SELECT * FROM forum_posts 
        WHERE LOWER(title) LIKE ? OR LOWER(content) LIKE ?
        ORDER BY updated_at DESC
    ''', (q, q))
    posts_rows = c.fetchall()
    
    posts = [_build_post_dict(c, row) for row in posts_rows]
    conn.close()
    return posts

def get_user_posts(username: str) -> List[dict]:
    """Get all posts by a specific user."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM forum_posts WHERE username = ? ORDER BY updated_at DESC', (username,))
    posts_rows = c.fetchall()
    
    posts = [_build_post_dict(c, row) for row in posts_rows]
    conn.close()
    return posts