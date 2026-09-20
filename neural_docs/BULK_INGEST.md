# Bulk Ingest

The Neural Memory tab now accepts multi-sentence text and text/Markdown files.

1. Load or paste text.
2. Click **Store Text in Brain**.
3. The text is tokenized into words and punctuation.
4. Tokens are processed in bounded continuous-neural chunks (12 tokens by default).
5. Each finished chunk is immediately compacted and spooled to disk.
6. Neural region digests are indexed for exact recall.
7. Previously unknown token labels are learned from those same neural regions.

No source body is copied into the episodic records. The optional filename is metadata only.

Search returns matching neural chunks. **Open Selected Chunk** decodes one chunk; **Open Full Document** follows the chunk order and reconstructs the complete decoded token stream.

The chunk size is intentionally conservative. Larger chunks can improve compression slightly, but they increase temporary simulation work and make progress less responsive. Until the recurrent simulator is faster, bounded chunks are the safer home-computer default.
