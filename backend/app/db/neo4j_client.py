"""
Neo4j Graph Database Client for GraphRAG.
Connects to Neo4j Aura cloud for knowledge graph queries.
"""

from neo4j import GraphDatabase
from typing import Optional, Any, List
import logging

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j database client for GraphRAG operations."""
    
    _instance: Optional["Neo4jClient"] = None
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j connection URI (e.g., neo4j+s://xxx.databases.neo4j.io)
            user: Username (default: neo4j)
            password: Password
            database: Database name (default: neo4j)
        """
        self.uri = uri
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Neo4j client initialized: {uri}")
    
    @classmethod
    def get_instance(cls, uri: str = None, user: str = None, 
                     password: str = None, database: str = None) -> "Neo4jClient":
        """Get singleton instance of Neo4j client."""
        if cls._instance is None:
            if not uri:
                from app.core.config import get_settings
                settings = get_settings()
                uri = settings.neo4j_uri
                user = settings.neo4j_user
                password = settings.neo4j_password
                database = settings.neo4j_database
            
            cls._instance = cls(uri, user, password, database)
        
        return cls._instance
    
    def close(self):
        """Close the driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def verify_connection(self) -> bool:
        """
        Test connection to Neo4j.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                return record and record["test"] == 1
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False
    
    def run_query(self, query: str, parameters: dict = None) -> List[dict]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result dictionaries
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def run_write(self, query: str, parameters: dict = None) -> Any:
        """
        Execute a write query (CREATE, MERGE, SET, DELETE).
        
        Args:
            query: Cypher write query
            parameters: Query parameters
            
        Returns:
            Query result summary
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return result.consume()
    
    def get_node_counts(self) -> dict:
        """Get count of all node types in the graph."""
        query = """
        CALL db.labels() YIELD label
        CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) as count', {})
        YIELD value
        RETURN label, value.count as count
        """
        try:
            # Simpler version without APOC
            query = "MATCH (n) RETURN labels(n)[0] as label, count(*) as count"
            results = self.run_query(query)
            return {r["label"]: r["count"] for r in results if r["label"]}
        except Exception as e:
            logger.warning(f"Could not get node counts: {e}")
            return {}
    
    def get_relationship_counts(self) -> dict:
        """Get count of all relationship types in the graph."""
        query = "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count"
        try:
            results = self.run_query(query)
            return {r["type"]: r["count"] for r in results if r["type"]}
        except Exception as e:
            logger.warning(f"Could not get relationship counts: {e}")
            return {}


def get_neo4j_client() -> Neo4jClient:
    """Get singleton Neo4j client instance."""
    return Neo4jClient.get_instance()
