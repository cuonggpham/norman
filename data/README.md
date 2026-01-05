# Data Directory

Thư mục chứa toàn bộ dữ liệu của hệ thống Japanese Legal RAG.

## Cấu trúc thư mục

```
data/
├── xml_raw/        # XML thô từ e-Gov API
├── processed/      # JSON đã xử lý và cấu trúc hóa
├── chunks/         # Chunks nhỏ để embedding
└── embeddings/     # Vector embeddings (cache)
```

## Pipeline xử lý dữ liệu

```mermaid
graph LR
    A[e-Gov API] -->|downloader.py| B[xml_raw/]
    B -->|xml_parser.py| C[processed/]
    C -->|chunker.py| D[chunks/]
    D -->|embedder.py| E[embeddings/]
    E -->|indexer.py| F[(Qdrant)]
```

## Kích thước dữ liệu hiện tại

| Folder | Files | Mô tả |
|--------|-------|-------|
| `xml_raw/` | 13 files | ~50MB XML thô |
| `processed/` | 15 files | ~35MB JSON cấu trúc |
| `chunks/` | 15 files | ~32MB, 15,629 chunks |
| `embeddings/` | - | Chưa có |

## Lưu ý

- Các folder này được git ignore (trừ `.gitkeep`)
- Để tái tạo dữ liệu, chạy các scripts theo thứ tự trong pipeline
