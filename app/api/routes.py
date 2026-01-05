# API Routes
# TODO: Implement search endpoint (Phase 3)
#
# POST /api/search
# {
#   "query": "労働時間の制限",
#   "top_k": 5,
#   "filters": { "category": "労働" }
# }

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["search"])


# @router.post("/search")
# async def search(query: SearchQuery):
#     """Vector search + reranking endpoint"""
#     pass
