# intelligent-video-content-search-system

A system for semantic search in videos, allowing users to locate specific objects or words and jump directly to relevant timestamps, improving video accessibility and navigation — all within 20 seconds.

## Features
- **Fast Retrieval**: Jump to specific moments in a video based on keywords.
- **Semantic Understanding**: Combines object detection and scene captioning to understand frame content.
- **Smart Search**: Uses natural language processing and vector search to identify relevant video segments.

## Technologies Used

| Module | Purpose |
|--------|---------|
| BLIP   | Image captioning (scene understanding) |
| YOLO   | Object detection in video frames |
| spaCy  | NLP preprocessing (lemmatization, parsing) |
| TF-IDF + LSI | Semantic search on generated frame text |

## How It Works

### Frame Extraction
Video is split into frames at regular intervals.

### Captioning & Detection
- BLIP generates a caption for each frame.
- YOLO detects key objects.

### Text Processing
- Captions and object labels are combined and processed using spaCy.

### Search Indexing
- TF-IDF is applied to frame descriptions.
- Latent Semantic Indexing (LSI) captures conceptual similarity.

### Search Interface
- User enters a query.
- System returns most relevant timestamps and jumps directly to those video points.
