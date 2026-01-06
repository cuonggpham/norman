# LLM Response Generation Service
# TODO: Implement for Phase 3
#
# Uses retrieved chunks as context to generate answers with citations


class LLMService:
    """Service for generating answers using LLM."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.model = model
        # TODO: Initialize OpenAI client
    
    def generate_response(
        self,
        query: str,
        context_chunks: list[dict],
    ) -> dict:
        """
        Generate answer with citations.
        
        Args:
            query: User's question
            context_chunks: Retrieved chunks with metadata
        
        Returns:
            {
                "answer": "Theo Điều 1...",
                "sources": [
                    {
                        "law_title": "労働基準法",
                        "article": "第一条",
                        "text": "...",
                        "highlight_path": ["労働基準法", "第一章", "第一条"]
                    }
                ]
            }
        """
        # TODO: Implement
        pass
