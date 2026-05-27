import os
import json
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AuraVectorStore")

class AuraVectorStore:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.enabled = False
        self.conn = None
        
        if not self.db_url:
            logger.warning("DATABASE_URL not set. Running AuraVectorStore in Offline Sandbox Fallback Mode.")
            return

        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
            
            # Connect to PostgreSQL pgvector instance
            self.conn = psycopg2.connect(self.db_url)
            self.conn.autocommit = True
            
            # Register pgvector type handlers
            cur = self.conn.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(self.conn)
            cur.close()
            
            self.enabled = True
            logger.info("Successfully connected to live PostgreSQL database with pgvector support.")
            self.create_tables()
            self.create_briefings_table()
            
        except Exception as e:
            logger.error(f"Failed to initialize live pgvector database: {str(e)}. Falling back to Local Development Sandbox Mode.")
            self.enabled = False
        
        # In-memory fallback stores for offline sandbox mode
        self._mock_briefings = []
        self._mock_briefing_counter = 0

    def create_tables(self):
        """Creates the multimodal assets table with pgvector column."""
        if not self.enabled:
            return
        
        try:
            cur = self.conn.cursor()
            # 768 dimensions matches Google's text-embedding-004 model
            cur.execute("""
                CREATE TABLE IF NOT EXISTS aura_assets (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(50) NOT NULL,
                    content_chunk TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector(768) NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS aura_assets_embedding_idx 
                ON aura_assets USING hnsw (embedding vector_cosine_ops);
            """)
            cur.close()
            logger.info("Aura Asset vector tables verified/created successfully.")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")

    def create_briefings_table(self):
        """Creates the briefings table for client request submissions."""
        if not self.enabled:
            return
        
        try:
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS aura_briefings (
                    id SERIAL PRIMARY KEY,
                    briefing_id VARCHAR(20) UNIQUE NOT NULL,
                    campaign_name VARCHAR(255) NOT NULL,
                    target_market VARCHAR(255),
                    channels JSONB DEFAULT '[]',
                    deadline DATE,
                    budget_tier VARCHAR(50) DEFAULT 'standard',
                    detailed_brief TEXT NOT NULL,
                    reference_files JSONB DEFAULT '[]',
                    status VARCHAR(30) DEFAULT 'submitted',
                    priority VARCHAR(20) DEFAULT 'normal',
                    hub_notes TEXT,
                    submitted_by VARCHAR(100) DEFAULT 'client',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS aura_briefings_status_idx ON aura_briefings (status);
                CREATE INDEX IF NOT EXISTS aura_briefings_created_idx ON aura_briefings (created_at DESC);
            """)
            cur.close()
            logger.info("Aura Briefings table verified/created successfully.")
        except Exception as e:
            logger.error(f"Error creating briefings table: {str(e)}")

    def add_asset(self, filename: str, file_type: str, content_chunk: str, metadata: dict, embedding: list):
        """Inserts a new vectorized document chunk or asset into the database."""
        if not self.enabled:
            logger.info(f"[Offline Sandbox] Cached asset '{filename}' ({file_type}) locally.")
            return True
        
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO aura_assets (filename, file_type, content_chunk, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s);
            """, (filename, file_type, content_chunk, json.dumps(metadata), embedding))
            cur.close()
            logger.info(f"Successfully vectorized and stored '{filename}' chunk.")
            return True
        except Exception as e:
            logger.error(f"Failed to store asset vector: {str(e)}")
            return False

    def semantic_search(self, query_embedding: list, limit: int = 3):
        """Executes a Cosine Similarity search in the vector DB using pgvector."""
        if not self.enabled:
            logger.info("[Offline Sandbox] Executing local semantic search fallback.")
            return self._get_mock_search_results(limit)
        
        try:
            import numpy as np
            cur = self.conn.cursor()
            # <=> is the pgvector Cosine Distance operator
            cur.execute("""
                SELECT filename, file_type, content_chunk, metadata, (embedding <=> %s) as distance
                FROM aura_assets
                ORDER BY embedding <=> %s
                LIMIT %s;
            """, (np.array(query_embedding), np.array(query_embedding), limit))
            
            rows = cur.fetchall()
            cur.close()
            
            results = []
            for row in rows:
                results.append({
                    "filename": row[0],
                    "file_type": row[1],
                    "content_chunk": row[2],
                    "metadata": row[3] if isinstance(row[3], dict) else json.loads(row[3] or "{}"),
                    "similarity": round(1 - float(row[4]), 4) # Convert Cosine Distance to Similarity
                })
            return results
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return self._get_mock_search_results(limit)

    # --- BRIEFING CRUD OPERATIONS ---

    def _generate_briefing_id(self):
        """Generates a sequential briefing ID in format BRF-YYYYMMDD-NNN."""
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y%m%d")
        
        if self.enabled:
            try:
                cur = self.conn.cursor()
                cur.execute("""
                    SELECT COUNT(*) FROM aura_briefings 
                    WHERE briefing_id LIKE %s;
                """, (f"BRF-{today}-%",))
                count = cur.fetchone()[0]
                cur.close()
                return f"BRF-{today}-{str(count + 1).zfill(3)}"
            except Exception:
                pass
        
        self._mock_briefing_counter += 1
        return f"BRF-{today}-{str(self._mock_briefing_counter).zfill(3)}"

    def insert_briefing(self, data: dict):
        """Inserts a new briefing request into the database."""
        briefing_id = self._generate_briefing_id()
        
        if not self.enabled:
            from datetime import datetime
            record = {
                "briefing_id": briefing_id,
                "campaign_name": data.get("campaign_name"),
                "target_market": data.get("target_market"),
                "channels": data.get("channels", []),
                "deadline": data.get("deadline"),
                "budget_tier": data.get("budget_tier", "standard"),
                "detailed_brief": data.get("detailed_brief"),
                "reference_files": data.get("reference_files", []),
                "status": "submitted",
                "priority": data.get("priority", "normal"),
                "hub_notes": None,
                "submitted_by": data.get("submitted_by", "client"),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            self._mock_briefings.append(record)
            logger.info(f"[Offline Sandbox] Stored briefing '{briefing_id}' in memory.")
            return record
        
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO aura_briefings 
                    (briefing_id, campaign_name, target_market, channels, deadline, 
                     budget_tier, detailed_brief, reference_files, status, priority, submitted_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'submitted', %s, %s)
                RETURNING briefing_id, campaign_name, target_market, channels, deadline,
                          budget_tier, detailed_brief, reference_files, status, priority,
                          hub_notes, submitted_by, created_at, updated_at;
            """, (
                briefing_id,
                data.get("campaign_name"),
                data.get("target_market"),
                json.dumps(data.get("channels", [])),
                data.get("deadline"),
                data.get("budget_tier", "standard"),
                data.get("detailed_brief"),
                json.dumps(data.get("reference_files", [])),
                data.get("priority", "normal"),
                data.get("submitted_by", "client")
            ))
            row = cur.fetchone()
            cur.close()
            logger.info(f"Successfully stored briefing '{briefing_id}'.")
            return {
                "briefing_id": row[0],
                "campaign_name": row[1],
                "target_market": row[2],
                "channels": row[3] if isinstance(row[3], list) else json.loads(row[3] or "[]"),
                "deadline": str(row[4]) if row[4] else None,
                "budget_tier": row[5],
                "detailed_brief": row[6],
                "reference_files": row[7] if isinstance(row[7], list) else json.loads(row[7] or "[]"),
                "status": row[8],
                "priority": row[9],
                "hub_notes": row[10],
                "submitted_by": row[11],
                "created_at": row[12].isoformat() if row[12] else None,
                "updated_at": row[13].isoformat() if row[13] else None
            }
        except Exception as e:
            logger.error(f"Failed to insert briefing: {str(e)}")
            return None

    def list_briefings(self, status_filter: str = None):
        """Lists all briefings, optionally filtered by status."""
        if not self.enabled:
            logger.info("[Offline Sandbox] Returning in-memory briefings.")
            if status_filter:
                return [b for b in self._mock_briefings if b["status"] == status_filter]
            return list(reversed(self._mock_briefings))
        
        try:
            cur = self.conn.cursor()
            if status_filter:
                cur.execute("""
                    SELECT briefing_id, campaign_name, target_market, channels, deadline,
                           budget_tier, detailed_brief, reference_files, status, priority,
                           hub_notes, submitted_by, created_at, updated_at
                    FROM aura_briefings
                    WHERE status = %s
                    ORDER BY created_at DESC;
                """, (status_filter,))
            else:
                cur.execute("""
                    SELECT briefing_id, campaign_name, target_market, channels, deadline,
                           budget_tier, detailed_brief, reference_files, status, priority,
                           hub_notes, submitted_by, created_at, updated_at
                    FROM aura_briefings
                    ORDER BY created_at DESC;
                """)
            
            rows = cur.fetchall()
            cur.close()
            
            results = []
            for row in rows:
                results.append({
                    "briefing_id": row[0],
                    "campaign_name": row[1],
                    "target_market": row[2],
                    "channels": row[3] if isinstance(row[3], list) else json.loads(row[3] or "[]"),
                    "deadline": str(row[4]) if row[4] else None,
                    "budget_tier": row[5],
                    "detailed_brief": row[6],
                    "reference_files": row[7] if isinstance(row[7], list) else json.loads(row[7] or "[]"),
                    "status": row[8],
                    "priority": row[9],
                    "hub_notes": row[10],
                    "submitted_by": row[11],
                    "created_at": row[12].isoformat() if row[12] else None,
                    "updated_at": row[13].isoformat() if row[13] else None
                })
            return results
        except Exception as e:
            logger.error(f"Failed to list briefings: {str(e)}")
            return []

    def update_briefing_status(self, briefing_id: str, new_status: str, hub_notes: str = None):
        """Updates the status of a briefing (designed for future hub team admin panel)."""
        valid_statuses = ["submitted", "in_review", "in_progress", "delivered"]
        if new_status not in valid_statuses:
            logger.warning(f"Invalid status '{new_status}'. Must be one of {valid_statuses}.")
            return None
        
        if not self.enabled:
            for b in self._mock_briefings:
                if b["briefing_id"] == briefing_id:
                    from datetime import datetime
                    b["status"] = new_status
                    if hub_notes:
                        b["hub_notes"] = hub_notes
                    b["updated_at"] = datetime.utcnow().isoformat()
                    logger.info(f"[Offline Sandbox] Updated briefing '{briefing_id}' to '{new_status}'.")
                    return b
            return None
        
        try:
            cur = self.conn.cursor()
            if hub_notes:
                cur.execute("""
                    UPDATE aura_briefings 
                    SET status = %s, hub_notes = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE briefing_id = %s
                    RETURNING briefing_id, campaign_name, target_market, channels, deadline,
                              budget_tier, detailed_brief, reference_files, status, priority,
                              hub_notes, submitted_by, created_at, updated_at;
                """, (new_status, hub_notes, briefing_id))
            else:
                cur.execute("""
                    UPDATE aura_briefings 
                    SET status = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE briefing_id = %s
                    RETURNING briefing_id, campaign_name, target_market, channels, deadline,
                              budget_tier, detailed_brief, reference_files, status, priority,
                              hub_notes, submitted_by, created_at, updated_at;
                """, (new_status, briefing_id))
            
            row = cur.fetchone()
            cur.close()
            
            if not row:
                return None
            
            logger.info(f"Successfully updated briefing '{briefing_id}' to '{new_status}'.")
            return {
                "briefing_id": row[0],
                "campaign_name": row[1],
                "target_market": row[2],
                "channels": row[3] if isinstance(row[3], list) else json.loads(row[3] or "[]"),
                "deadline": str(row[4]) if row[4] else None,
                "budget_tier": row[5],
                "detailed_brief": row[6],
                "reference_files": row[7] if isinstance(row[7], list) else json.loads(row[7] or "[]"),
                "status": row[8],
                "priority": row[9],
                "hub_notes": row[10],
                "submitted_by": row[11],
                "created_at": row[12].isoformat() if row[12] else None,
                "updated_at": row[13].isoformat() if row[13] else None
            }
        except Exception as e:
            logger.error(f"Failed to update briefing status: {str(e)}")
            return None

    def _get_mock_search_results(self, limit: int):
        """Returns mock asset matching results to keep the demo fully operational without database active."""
        mock_library = [
            {
                "filename": "Aura_Brand_Guidelines_v4.pdf",
                "file_type": "pdf",
                "content_chunk": "Aura brand colors are defined by deep Teal (#0D9488) representing structure, and Royal Indigo (#6366F1) representing creative freedom.",
                "metadata": {"category": "branding", "page": 12},
                "similarity": 0.9421
            },
            {
                "filename": "Q3_CPG_Cafes_Campaign_Presentation.pptx",
                "file_type": "ppt",
                "content_chunk": "Forecourt cafes showed a 14% increase in commuter retention when morning promotions paired hot wraps with organic artisan coffee.",
                "metadata": {"category": "marketing", "slide": 8},
                "similarity": 0.8912
            },
            {
                "filename": "Roadster_Product_Specs_Sheet.pdf",
                "file_type": "pdf",
                "content_chunk": "The Aura Roadster hybrid-electric drivetrain achieves zero urban emission cruise modes using smart battery load balancing.",
                "metadata": {"category": "engineering", "page": 4},
                "similarity": 0.8654
            }
        ]
        return mock_library[:limit]
