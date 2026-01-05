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

---

## 1. XML Raw (`xml_raw/`)

### Nguồn dữ liệu
- **API**: `https://laws.e-gov.go.jp/api/2/laws?law_id={law_id}`
- **Script**: `scripts/downloader.py`

### Cấu trúc mẫu

```xml
<law_data_response>
  <attached_files_info/>
  <law_info>
    <law_type>Act</law_type>
    <law_id>419AC0000000128</law_id>
    <law_num>平成十九年法律第百二十八号</law_num>
    <law_num_era>Heisei</law_num_era>
    <law_num_year>19</law_num_year>
    <law_num_type>Act</law_num_type>
    <law_num_num>128</law_num_num>
    <promulgation_date>2007-12-05</promulgation_date>
  </law_info>
  <revision_info>
    <law_revision_id>419AC0000000128_20200401_430AC0000000071</law_revision_id>
    <law_title>労働契約法</law_title>
    <law_title_kana>ろうどうけいやくほう</law_title_kana>
    <abbrev>労契法</abbrev>
    <category>労働</category>
    <amendment_enforcement_date>2020-04-01</amendment_enforcement_date>
    <current_revision_status>CurrentEnforced</current_revision_status>
    <!-- ... more fields ... -->
  </revision_info>
  <law_full_text>
    <Law Era="Heisei" Lang="ja" LawType="Act" Num="128">
      <LawNum>平成十九年法律第百二十八号</LawNum>
      <LawBody>
        <LawTitle Abbrev="労契法" Kana="ろうどうけいやくほう">労働契約法</LawTitle>
        <TOC>
          <TOCLabel>目次</TOCLabel>
          <TOCChapter Num="1">
            <ChapterTitle>第一章　総則</ChapterTitle>
            <ArticleRange>（第一条―第五条）</ArticleRange>
          </TOCChapter>
        </TOC>
        <MainProvision>
          <Chapter Num="1">
            <ChapterTitle>第一章　総則</ChapterTitle>
            <Article Num="1">
              <ArticleCaption>（目的）</ArticleCaption>
              <ArticleTitle>第一条</ArticleTitle>
              <Paragraph Num="1">
                <ParagraphNum/>
                <ParagraphSentence>
                  <Sentence Num="1" WritingMode="vertical">この法律は...</Sentence>
                </ParagraphSentence>
              </Paragraph>
            </Article>
          </Chapter>
        </MainProvision>
        <SupplProvision>...</SupplProvision>
      </LawBody>
    </Law>
  </law_full_text>
</law_data_response>
```

### Các elements chính

| Element | Mô tả |
|---------|-------|
| `law_data_response` | Root wrapper từ e-Gov API |
| `law_info` | Metadata cơ bản: law_id, law_type, promulgation_date |
| `revision_info` | Thông tin sửa đổi: title, abbrev, category, enforcement_date |
| `law_full_text` | Nội dung đầy đủ của luật |
| `LawBody` | Body chứa title, TOC, MainProvision, SupplProvision |
| `Chapter` | Chương (có Num attribute) |
| `Article` | Điều (có Num attribute) |
| `Paragraph` | Khoản |
| `Sentence` | Câu (có WritingMode attribute) |

---

## 2. Processed (`processed/`)

### Script xử lý
- **Script**: `scripts/xml_parser.py`
- **Input**: `data/xml_raw/*.xml`
- **Output**: `data/processed/*.json`

### Cấu trúc mẫu - File riêng lẻ (`{law_id}.json`)

```json
{
  "source_file": "419AC0000000128.xml",
  "parsed_at": "2026-01-04T02:01:32.308198",
  "law_info": {
    "law_type": "Act",
    "law_id": "419AC0000000128",
    "law_num": "平成十九年法律第百二十八号",
    "law_num_era": "Heisei",
    "law_num_year": "19",
    "law_num_type": "Act",
    "law_num_num": "128",
    "promulgation_date": "2007-12-05"
  },
  "revision_info": {
    "law_revision_id": "419AC0000000128_20200401_430AC0000000071",
    "law_type": "Act",
    "law_title": "労働契約法",
    "law_title_kana": "ろうどうけいやくほう",
    "abbrev": "労契法",
    "category": "労働",
    "updated": "2024-07-22T14:14:50+09:00",
    "amendment_promulgate_date": "2018-07-06",
    "amendment_enforcement_date": "2020-04-01",
    "amendment_law_id": "430AC0000000071",
    "amendment_law_title": "働き方改革を推進するための関係法律の整備に関する法律",
    "current_revision_status": "CurrentEnforced"
  },
  "law_full_text": {
    "attributes": {
      "era": "Heisei",
      "lang": "ja",
      "law_type": "Act",
      "num": "128"
    },
    "law_num": "平成十九年法律第百二十八号",
    "law_body": {
      "title": {
        "text": "労働契約法",
        "abbrev": "労契法",
        "kana": "ろうどうけいやくほう"
      },
      "toc": {
        "label": "目次",
        "chapters": [
          { "num": "1", "title": "第一章　総則", "article_range": "（第一条―第五条）" }
        ]
      },
      "main_provision": {
        "chapters": [
          {
            "num": "1",
            "title": "第一章　総則",
            "articles": [
              {
                "num": "1",
                "caption": "（目的）",
                "title": "第一条",
                "paragraphs": [
                  {
                    "num": "1",
                    "sentences": [
                      {
                        "num": "1",
                        "writing_mode": "vertical",
                        "text": "この法律は、労働者及び使用者の..."
                      }
                    ]
                  }
                ]
              }
            ]
          }
        ]
      },
      "supplementary_provisions": [
        {
          "label": "附　則",
          "extract": "true",
          "articles": [...]
        }
      ]
    }
  }
}
```

### Cấu trúc mẫu - Index (`_index.json`)

```json
[
  {
    "law_id": "419AC0000000128",
    "title": "労働契約法",
    "title_kana": "ろうどうけいやくほう",
    "abbrev": "労契法",
    "category": "労働",
    "law_type": "Act",
    "promulgation_date": "2007-12-05",
    "current_revision_status": "CurrentEnforced",
    "amendment_enforcement_date": "2020-04-01",
    "chapter_count": 5,
    "article_count": 21,
    "file": "419AC0000000128.json"
  }
]
```

### Hierarchy

```
Law (法)
├── Main Provision (本則)
│   └── Chapter (章)
│       └── Article (条)
│           └── Paragraph (項)
│               └── Sentence (文)
│                   └── Item (号) [optional]
└── Supplementary Provisions (附則)
    └── Article (条) → Paragraph → Sentence
```

---

## 3. Chunks (`chunks/`)

### Script xử lý
- **Script**: `scripts/chunker.py`
- **Input**: `data/processed/*.json`
- **Output**: `data/chunks/*.json`
- **Chiến lược**: Paragraph (項) - mỗi khoản là 1 chunk

### Cấu trúc mẫu - Chunk

```json
{
  "chunk_id": "419AC0000000128_1_1",
  "text": "この法律は、労働者及び使用者の自主的な交渉の下で...",
  "text_with_context": "労働契約法 （目的） 第一条 この法律は...",
  "metadata": {
    "law_id": "419AC0000000128",
    "law_title": "労働契約法",
    "law_abbrev": "労契法",
    "category": "労働",
    "chapter_num": "1",
    "chapter_title": "第一章　総則",
    "article_num": "1",
    "article_title": "第一条",
    "article_caption": "（目的）",
    "paragraph_num": "1",
    "sentence_nums": ["1"],
    "source_type": "main",
    "suppl_amend_law_num": null
  },
  "char_count": 150,
  "token_estimate": 75,
  "highlight_path": {
    "law": "労働契約法",
    "article": "第一条",
    "chapter": "第一章　総則",
    "paragraph": "1項"
  }
}
```

### Cấu trúc mẫu - Stats (`_stats.json`)

```json
{
  "total_laws": 13,
  "total_chunks": 15629,
  "total_chars": 2998704,
  "by_category": {
    "労働": 867,
    "地方財政": 6226,
    "社会保険": 3758,
    "国税": 3515
  },
  "by_law": {
    "322AC0000000049": {
      "title": "労働基準法",
      "chunk_count": 429,
      "char_count": 56324
    }
  }
}
```

### Các trường quan trọng

| Trường | Mô tả | Mục đích |
|--------|-------|----------|
| `text` | Nội dung gốc | Hiển thị cho user |
| `text_with_context` | Text + context | Dùng để embedding |
| `metadata` | Thông tin chi tiết | Filtering, faceted search |
| `highlight_path` | Đường dẫn cấu trúc | Highlight trong UI |
| `char_count` | Số ký tự | Monitoring |
| `token_estimate` | Ước lượng tokens | Theo dõi chi phí OpenAI |

---

## 4. Embeddings (`embeddings/`)

### Script xử lý
- **Script**: `scripts/embedder.py` (🚧 chưa implement)
- **Input**: `data/chunks/*.json`
- **Output**: `data/embeddings/*.npy`, `*.json`

### Cấu trúc dự kiến

```
embeddings/
├── 322AC0000000049_embeddings.npy    # Numpy array of embeddings
├── 322AC0000000049_metadata.json     # Mapping chunk_id → index
├── _embedding_config.json            # Model config used
└── ...
```

### Config mẫu (`_embedding_config.json`)

```json
{
  "model": "text-embedding-3-small",
  "dimensions": 1536,
  "created_at": "2026-01-05T...",
  "total_chunks": 15629,
  "total_tokens_used": 450000
}
```

---

## Kích thước dữ liệu hiện tại

| Folder | Files | Mô tả |
|--------|-------|-------|
| `xml_raw/` | 13 files | ~50MB XML thô |
| `processed/` | 15 files | ~35MB JSON cấu trúc |
| `chunks/` | 15 files | ~32MB, 15,629 chunks |
| `embeddings/` | - | Chưa có |

## Lưu ý

- Các folder này được git ignore (trừ `.gitkeep` và `README.md`)
- Để tái tạo dữ liệu, chạy các scripts theo thứ tự trong pipeline

## Cách chạy Pipeline

```bash
# 1. Download XML từ e-Gov API
python scripts/downloader.py

# 2. Parse XML → JSON cấu trúc
python scripts/xml_parser.py

# 3. Chunk JSON → nhỏ chunks
python scripts/chunker.py

# 4. Tạo embeddings (chưa implement)
python scripts/embedder.py

# 5. Index vào Qdrant (chưa implement)
python scripts/indexer.py
```
