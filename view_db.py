import sqlite3
import os

def view_database(db_path='ruralminds.db'):
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        return

    conn = sqlite3.connect(db_path)
    # This allows us to access columns by name
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row['name'] for row in cursor.fetchall() if row['name'] != 'sqlite_sequence']

    print("=" * 60)
    print(f" DATABASE SUMMARY: {db_path}")
    print("=" * 60)

    if not tables:
        print("No tables found in the database.")
    else:
        for table in tables:
            # Get count
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            count = cursor.fetchone()['count']
            
            print(f"\n[Table: {table}] - {count} records")
            print("-" * 40)
            
            # Get last 5 records
            cursor.execute(f"SELECT * FROM {table} LIMIT 5")
            rows = cursor.fetchall()
            
            if not rows:
                print("  (Table is empty)")
            else:
                # Get column names
                col_names = [description[0] for description in cursor.description]
                header = " | ".join(col_names)
                print(f"  {header}")
                print(f"  {'-' * len(header)}")
                
                for row in rows:
                    values = [str(row[col])[:30] + '...' if len(str(row[col])) > 30 else str(row[col]) for col in col_names]
                    print(f"  {' | '.join(values)}")
    
    print("\n" + "=" * 60)
    conn.close()

if __name__ == "__main__":
    view_database()
