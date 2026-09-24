from __future__ import annotations

import io
import math
import os
import re
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import streamlit as st

try:
    import fitz
except ImportError:
    fitz = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from transformers import CLIPModel, CLIPProcessor, pipeline
except ImportError:
    CLIPModel = None
    CLIPProcessor = None
    pipeline = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


TEXT_MODEL_NAME = os.getenv("TEXT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_MODEL_NAME = os.getenv("IMAGE_EMBED_MODEL", "openai/clip-vit-base-patch32")
CAPTION_MODEL_NAME = os.getenv("IMAGE_CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
MAX_TEXT_CHARS = 1200
TEXT_OVERLAP = 200
TOP_K_TEXT = 6
TOP_K_IMAGES = 3
EMBED_BATCH_SIZE = 32
MIN_IMAGE_DIM = 120
MIN_IMAGE_AREA = 40000
MIN_SENTENCE_SUPPORT = 0.42
MAX_IMAGES_PER_PAGE = 3
MAX_TOTAL_IMAGE_CHUNKS = 60
ENABLE_IMAGE_CAPTIONING = os.getenv("ENABLE_IMAGE_CAPTIONING", "false").lower() == "true"
IMAGE_EMBED_MAX_SIDE = 768
SUMMARY_QUERY_HINTS = (
    "what is this document about",
    "what is this pdf about",
    "summarize this document",
    "summarise this document",
    "summary of this document",
    "briefly explain this document",
    "tell briefly",
    "overview of this document",
)
VISUAL_QUERY_HINTS = ("image", "figure", "chart", "diagram", "graph", "table", "photo", "visual", "illustration")
STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "among", "and", "any", "are", "because", "been",
    "before", "being", "below", "between", "brief", "briefly", "could", "does", "document", "during", "each",
    "from", "have", "into", "like", "main", "more", "most", "other", "over", "page", "pages", "pdf", "should",
    "some", "such", "than", "that", "their", "them", "then", "there", "these", "they", "this", "those", "under",
    "what", "when", "where", "which", "while", "with", "would", "your",
}


@dataclass
class TextChunk:
    content: str
    document_name: str
    page_number: int
    chunk_id: str
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


@dataclass
class ImageChunk:
    image: Any
    document_name: str
    page_number: int
    image_id: str
    caption: str
    context_text: str
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    kind: str
    score: float
    citation: str
    content: str
    page_number: int
    document_name: str
    image: Any = None


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def tokenize(value: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]{3,}", (value or "").lower())


def split_text(text: str, max_chars: int = MAX_TEXT_CHARS, overlap: int = TEXT_OVERLAP) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            last_break = text.rfind(" ", start, end)
            if last_break > start + max_chars // 2:
                end = last_break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def keyword_overlap_score(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(text_tokens))


def query_terms(query: str) -> List[str]:
    return [token for token in tokenize(query) if token not in STOPWORDS]


def best_fuzzy_token_match(token: str, candidates: Sequence[str]) -> float:
    if not token or not candidates:
        return 0.0
    return max(SequenceMatcher(None, token, candidate).ratio() for candidate in candidates)


def fuzzy_term_coverage(query: str, text: str) -> float:
    terms = query_terms(query)
    candidates = tokenize(text)
    if not terms or not candidates:
        return 0.0
    matched = 0.0
    for term in terms:
        score = best_fuzzy_token_match(term, candidates)
        if score >= 0.84:
            matched += score
    return matched / len(terms)


def is_summary_query(query: str) -> bool:
    normalized_query = normalize_text(query).lower()
    if any(hint in normalized_query for hint in SUMMARY_QUERY_HINTS):
        return True
    summary_tokens = {"about", "brief", "briefly", "document", "overview", "pdf", "summary", "summarize", "summarise"}
    return len(set(tokenize(normalized_query)) & summary_tokens) >= 2


def is_visual_query(query: str) -> bool:
    normalized_query = normalize_text(query).lower()
    return any(hint in normalized_query for hint in VISUAL_QUERY_HINTS)


def first_sentences(text: str, limit: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", normalize_text(text))
    selected = [part for part in parts if part][:limit]
    return " ".join(selected).strip()


def clean_summary_snippet(text: str, max_length: int = 220) -> str:
    snippet = first_sentences(text, limit=2)
    if len(snippet) <= max_length:
        return snippet
    trimmed = snippet[:max_length].rsplit(" ", 1)[0].strip()
    return f"{trimmed}..."


def build_document_overviews(text_chunks: Sequence[TextChunk]) -> List[RetrievalResult]:
    overviews: List[RetrievalResult] = []
    grouped_chunks: Dict[str, List[TextChunk]] = {}
    for chunk in text_chunks:
        grouped_chunks.setdefault(chunk.document_name, []).append(chunk)

    for document_name, chunks in grouped_chunks.items():
        ordered_chunks = sorted(chunks, key=lambda item: (item.page_number, item.chunk_id))
        lead_chunks = ordered_chunks[:3]
        if not lead_chunks:
            continue
        overview_text = " ".join(first_sentences(chunk.content, limit=1) for chunk in lead_chunks if chunk.content)
        overview_text = normalize_text(overview_text)[:700]
        if not overview_text:
            continue
        pages = sorted({chunk.page_number for chunk in lead_chunks})
        citations = ", ".join(f"page {page}" for page in pages)
        overviews.append(
            RetrievalResult(
                kind="text",
                score=1.0,
                citation=f"{document_name}, {citations}",
                content=overview_text,
                page_number=pages[0],
                document_name=document_name,
            )
        )
    return overviews


def format_citations(results: Sequence[RetrievalResult]) -> List[str]:
    seen: set[str] = set()
    citations: List[str] = []
    for result in results:
        if result.citation not in seen:
            citations.append(result.citation)
            seen.add(result.citation)
    return citations


def unique_citations(citations: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for citation in citations:
        if citation in seen:
            continue
        ordered.append(citation)
        seen.add(citation)
    return ordered


def chunk_sequence(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def exact_match_score(query: str, text: str) -> float:
    normalized_query = normalize_text(query).lower()
    normalized_text = normalize_text(text).lower()
    if not normalized_query or not normalized_text:
        return 0.0
    if normalized_query in normalized_text:
        return 1.0
    terms = query_terms(normalized_query)
    if not terms:
        return 0.0
    phrase_hits = sum(1 for token in terms if token in normalized_text)
    return phrase_hits / len(terms)


def extract_relevant_sentences(query: str, results: Sequence[RetrievalResult], limit: int = 4) -> List[Tuple[float, str, str]]:
    visual_mode = is_visual_query(query)
    candidate_results = [result for result in results if result.kind == "text" or (visual_mode and result.kind == "image")]
    ranked_sentences: List[Tuple[float, str, str]] = []
    for result in candidate_results:
        sentences = re.split(r"(?<=[.!?])\s+", normalize_text(result.content))
        for sentence in sentences:
            cleaned = sentence.strip()
            if len(cleaned) < 25:
                continue
            score = (
                0.35 * keyword_overlap_score(query, cleaned)
                + 0.35 * exact_match_score(query, cleaned)
                + 0.30 * fuzzy_term_coverage(query, cleaned)
            )
            score += max(result.score, 0.0) * 0.2
            ranked_sentences.append((score, cleaned, result.citation))
    ranked_sentences.sort(key=lambda item: item[0], reverse=True)

    if not ranked_sentences:
        return []

    dominant_citation = ranked_sentences[0][2]
    dominant_score = ranked_sentences[0][0]
    picked: List[Tuple[float, str, str]] = []
    seen_sentences: set[str] = set()
    for score, sentence, citation in ranked_sentences:
        sentence_key = sentence.lower()
        if sentence_key in seen_sentences:
            continue
        if score < MIN_SENTENCE_SUPPORT:
            continue
        if citation != dominant_citation and score < dominant_score * 0.94 and not is_summary_query(query):
            continue
        picked.append((score, sentence, citation))
        seen_sentences.add(sentence_key)
        if len(picked) >= limit:
            break
    return picked


def derive_answer_citations(query: str, results: Sequence[RetrievalResult]) -> List[str]:
    if is_summary_query(query):
        return format_citations(results[:3])
    support = extract_relevant_sentences(query, results[: max(TOP_K_TEXT, 6)], limit=3)
    if support:
        return unique_citations([citation for _, _, citation in support])
    return format_citations(results[:2])


@st.cache_resource(show_spinner=False)
def get_text_embedder() -> Optional[Any]:
    if SentenceTransformer is None:
        return None
    return SentenceTransformer(TEXT_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_clip_components() -> Tuple[Optional[Any], Optional[Any]]:
    if CLIPModel is None or CLIPProcessor is None:
        return None, None
    processor = CLIPProcessor.from_pretrained(IMAGE_MODEL_NAME)
    model = CLIPModel.from_pretrained(IMAGE_MODEL_NAME)
    return processor, model


@st.cache_resource(show_spinner=False)
def get_captioner() -> Optional[Any]:
    if pipeline is None:
        return None
    try:
        return pipeline("image-to-text", model=CAPTION_MODEL_NAME)
    except Exception:
        return None


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    model = get_text_embedder()
    if model is not None:
        batches: List[np.ndarray] = []
        for batch in chunk_sequence(list(texts), EMBED_BATCH_SIZE):
            batch_embeddings = model.encode(list(batch), normalize_embeddings=True, show_progress_bar=False)
            batches.append(np.asarray(batch_embeddings))
        return np.vstack(batches) if batches else np.empty((0, 384), dtype=np.float32)

    fallback_vectors = []
    for text in texts:
        tokens = tokenize(text)
        vector = np.zeros(384, dtype=np.float32)
        for token in tokens:
            vector[hash(token) % 384] += 1.0
        norm = np.linalg.norm(vector)
        fallback_vectors.append(vector if norm == 0 else vector / norm)
    return np.asarray(fallback_vectors)


def embed_query_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def tensor_to_numpy(output: Any) -> np.ndarray:
    if hasattr(output, "detach") and hasattr(output, "cpu"):
        return output.detach().cpu().numpy()
    if hasattr(output, "cpu"):
        return output.cpu().numpy()
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        tensor = output.pooler_output
        return tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor.cpu().numpy()
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        tensor = output.last_hidden_state[:, 0, :]
        return tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor.cpu().numpy()
    raise TypeError(f"Unsupported model output type: {type(output)}")


def prepare_image_for_embedding(image: Any) -> Any:
    if Image is None or image is None:
        return image
    width, height = image.size
    max_side = max(width, height)
    if max_side <= IMAGE_EMBED_MAX_SIDE:
        return image
    scale = IMAGE_EMBED_MAX_SIDE / max_side
    resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    return resized


def embed_images(images: Sequence[Any]) -> Optional[np.ndarray]:
    processor, model = get_clip_components()
    if processor is None or model is None or not images:
        return None

    batches: List[np.ndarray] = []
    for batch in chunk_sequence(list(images), EMBED_BATCH_SIZE):
        prepared_batch = [prepare_image_for_embedding(image) for image in batch]
        inputs = processor(images=prepared_batch, return_tensors="pt", padding=True)
        outputs = model.get_image_features(**inputs)
        embeddings = tensor_to_numpy(outputs)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        batches.append(embeddings / norms)
    return np.vstack(batches) if batches else None


def embed_image_query(text: str) -> Optional[np.ndarray]:
    processor, model = get_clip_components()
    if processor is None or model is None:
        return None
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    outputs = model.get_text_features(**inputs)
    embedding = tensor_to_numpy(outputs)[0]
    norm = np.linalg.norm(embedding)
    return embedding if norm == 0 else embedding / norm


def caption_image(image: Any, page_number: int) -> str:
    if not ENABLE_IMAGE_CAPTIONING:
        return f"Image extracted from page {page_number}"
    captioner = get_captioner()
    if captioner is None:
        return f"Image extracted from page {page_number}"
    try:
        result = captioner(image)
        if result and isinstance(result, list):
            text = result[0].get("generated_text", "")
            cleaned = normalize_text(text)
            return cleaned or f"Image extracted from page {page_number}"
    except Exception:
        pass
    return f"Image extracted from page {page_number}"


def extract_document(file_bytes: bytes, document_name: str) -> Tuple[List[TextChunk], List[ImageChunk]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required. Install it with `pip install pymupdf`.")
    if Image is None:
        raise RuntimeError("Pillow is required. Install it with `pip install pillow`.")

    text_chunks: List[TextChunk] = []
    image_chunks: List[ImageChunk] = []
    document = fitz.open(stream=file_bytes, filetype="pdf")
    total_image_chunks = 0

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        page_number = page_index + 1
        page_text = normalize_text(page.get_text("text"))

        for chunk_index, chunk in enumerate(split_text(page_text)):
            text_chunks.append(
                TextChunk(
                    content=chunk,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_id=f"{document_name}-p{page_number}-t{chunk_index}",
                    keywords=tokenize(chunk),
                )
            )

        images = page.get_images(full=True)
        page_context = page_text[:400]
        page_image_count = 0
        for image_index, image_info in enumerate(images):
            if page_image_count >= MAX_IMAGES_PER_PAGE or total_image_chunks >= MAX_TOTAL_IMAGE_CHUNKS:
                break
            image_width = int(image_info[2]) if len(image_info) > 2 else 0
            image_height = int(image_info[3]) if len(image_info) > 3 else 0
            if image_width < MIN_IMAGE_DIM or image_height < MIN_IMAGE_DIM:
                continue
            if image_width * image_height < MIN_IMAGE_AREA:
                continue
            xref = image_info[0]
            base_image = document.extract_image(xref)
            image_bytes = base_image.get("image")
            if not image_bytes:
                continue
            extracted = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            caption = caption_image(extracted, page_number)
            image_chunks.append(
                ImageChunk(
                    image=extracted,
                    document_name=document_name,
                    page_number=page_number,
                    image_id=f"{document_name}-p{page_number}-i{image_index}",
                    caption=caption,
                    context_text=page_context,
                )
            )
            page_image_count += 1
            total_image_chunks += 1

    return text_chunks, image_chunks


def build_index(uploaded_files: Sequence[Any]) -> Tuple[List[TextChunk], List[ImageChunk]]:
    all_text_chunks: List[TextChunk] = []
    all_image_chunks: List[ImageChunk] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        text_chunks, image_chunks = extract_document(file_bytes, uploaded_file.name)
        all_text_chunks.extend(text_chunks)
        all_image_chunks.extend(image_chunks)

    if all_text_chunks:
        text_embeddings = embed_texts([chunk.content for chunk in all_text_chunks])
        for chunk, embedding in zip(all_text_chunks, text_embeddings):
            chunk.embedding = embedding

    if all_image_chunks:
        image_embeddings = embed_images([chunk.image for chunk in all_image_chunks])
        if image_embeddings is not None:
            for chunk, embedding in zip(all_image_chunks, image_embeddings):
                chunk.embedding = embedding

    return all_text_chunks, all_image_chunks


def retrieve(query: str, text_chunks: Sequence[TextChunk], image_chunks: Sequence[ImageChunk]) -> List[RetrievalResult]:
    query_embedding = embed_query_text(query)
    summary_mode = is_summary_query(query)
    visual_mode = is_visual_query(query)
    text_results: List[RetrievalResult] = []
    for chunk in text_chunks:
        dense_score = cosine_similarity(query_embedding, chunk.embedding) if chunk.embedding is not None else 0.0
        lexical_score = keyword_overlap_score(query, chunk.content)
        exact_score = exact_match_score(query, chunk.content)
        fuzzy_score = fuzzy_term_coverage(query, chunk.content)
        score = 0.40 * dense_score + 0.22 * lexical_score + 0.23 * exact_score + 0.15 * fuzzy_score
        if summary_mode and chunk.page_number <= 2:
            score += 0.12
        if exact_score >= 0.85:
            score += 0.18
        text_results.append(
            RetrievalResult(
                kind="text",
                score=score,
                citation=f"{chunk.document_name}, page {chunk.page_number}",
                content=chunk.content,
                page_number=chunk.page_number,
                document_name=chunk.document_name,
            )
        )

    image_results: List[RetrievalResult] = []
    if visual_mode:
        image_query_embedding = embed_image_query(query)
        for chunk in image_chunks:
            dense_score = 0.0
            if image_query_embedding is not None and chunk.embedding is not None:
                dense_score = cosine_similarity(image_query_embedding, chunk.embedding)
            lexical_score = keyword_overlap_score(query, f"{chunk.caption} {chunk.context_text}")
            exact_score = exact_match_score(query, f"{chunk.caption} {chunk.context_text}")
            fuzzy_score = fuzzy_term_coverage(query, f"{chunk.caption} {chunk.context_text}")
            score = 0.40 * dense_score + 0.22 * lexical_score + 0.23 * exact_score + 0.15 * fuzzy_score
            image_results.append(
                RetrievalResult(
                    kind="image",
                    score=score,
                    citation=f"{chunk.document_name}, page {chunk.page_number}",
                    content=f"Caption: {chunk.caption}\nContext: {chunk.context_text}",
                    page_number=chunk.page_number,
                    document_name=chunk.document_name,
                    image=chunk.image,
                )
            )

    top_text = sorted(text_results, key=lambda item: item.score, reverse=True)[:TOP_K_TEXT]
    top_images = sorted(image_results, key=lambda item: item.score, reverse=True)[:TOP_K_IMAGES]
    if summary_mode:
        overview_results = build_document_overviews(text_chunks)
        for index, overview in enumerate(overview_results):
            overview.score = 1.2 - (index * 0.05)
        top_text = overview_results[: max(1, min(3, len(overview_results)))] + top_text

    deduped_results: List[RetrievalResult] = []
    seen_keys: set[tuple[str, str]] = set()
    for result in sorted(top_text + top_images, key=lambda item: item.score, reverse=True):
        key = (result.citation, result.content)
        if key in seen_keys:
            continue
        deduped_results.append(result)
        seen_keys.add(key)
    return deduped_results


def build_context(results: Sequence[RetrievalResult]) -> str:
    sections = []
    for index, result in enumerate(results, start=1):
        sections.append(
            f"[{index}] Type: {result.kind}\nCitation: {result.citation}\nEvidence: {result.content}"
        )
    return "\n\n".join(sections)


def fallback_answer(query: str, results: Sequence[RetrievalResult]) -> str:
    if not results:
        return "I could not find relevant evidence in the uploaded documents."
    citations = derive_answer_citations(query, results)
    summary_mode = is_summary_query(query)

    answer_lines: List[str] = []
    if summary_mode:
        summary_bits: List[str] = []
        for result in results[:3]:
            snippet = clean_summary_snippet(result.content)
            if snippet:
                summary_bits.append(f"{snippet} [{result.citation}]")
        if summary_bits:
            answer_lines.append(" ".join(summary_bits))
        else:
            answer_lines.append("I found relevant parts of the document, but could not build a brief overview from them.")
        answer_lines.append("")
        answer_lines.append("Citations: " + "; ".join(citations))
        return "\n".join(answer_lines)

    lead = results[0]
    relevant_sentences = extract_relevant_sentences(query, results[:6], limit=3)
    best_support = relevant_sentences[0][0] if relevant_sentences else 0.0
    answer_lines.extend(
        [
            f"Best grounded answer for: {query}",
            f"The strongest evidence comes from {lead.citation}.",
            "",
            "Answer:",
        ]
    )
    if relevant_sentences and best_support >= MIN_SENTENCE_SUPPORT:
        merged_answer = " ".join(sentence for _, sentence, _ in relevant_sentences[:2])
        answer_lines.append(merged_answer)
    else:
        answer_lines.append(
            "I could not find enough exact support in the document to answer this confidently. "
            "Please try a more specific question or wording that matches the document text."
        )
    answer_lines.append("")
    answer_lines.append("Key evidence:")
    for _, sentence, citation in relevant_sentences:
        answer_lines.append(f"- {sentence} [{citation}]")
    answer_lines.append("")
    answer_lines.append("Citations: " + "; ".join(citations))
    return "\n".join(answer_lines)


def generate_answer(query: str, results: Sequence[RetrievalResult]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return fallback_answer(query, results)

    client = OpenAI(api_key=api_key)
    context = build_context(results)
    prompt = (
        "You are a grounded multimodal RAG assistant. Use only the provided evidence. "
        "If the evidence is incomplete, say so explicitly. "
        "When the user asks what a document is about, answer with a short overview in 3-5 sentences. "
        "For factual questions, do not guess or generalize from related text. "
        "Answer only when the evidence directly supports the claim. "
        "Cite sources inline using the exact citation text, and end with a 'Citations:' line.\n\n"
        f"Question: {query}\n\nEvidence:\n{context}"
    )
    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=prompt,
        )
        return response.output_text
    except Exception as exc:
        return f"OpenAI generation failed: {exc}\n\n{fallback_answer(query, results)}"


def render_sidebar() -> None:
    st.sidebar.title("Multimodal RAG")
    st.sidebar.markdown(
        "Upload PDFs with text, charts, and figures. The app extracts page text, images, captions, and citations."
    )
    st.sidebar.markdown("**Pipeline**")
    st.sidebar.markdown(
        "1. Parse PDF pages\n"
        "2. Chunk text\n"
        "3. Caption + embed images\n"
        "4. Hybrid retrieve text and visuals\n"
        "5. Answer with citations"
    )
    st.sidebar.markdown("**Optional models**")
    st.sidebar.code(
        "pip install streamlit pymupdf pillow sentence-transformers transformers torch openai"
    )
    st.sidebar.caption(
        "Fast indexing mode is on by default: image captioning is off, large images are capped, and mobile layout is optimized."
    )


def render_app_chrome() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(99, 102, 241, 0.22), transparent 28%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.16), transparent 22%),
                linear-gradient(180deg, #081120 0%, #0d1528 58%, #111827 100%);
            color: #e5eefc;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(11, 18, 32, 0.98), rgba(17, 24, 39, 0.95));
            border-right: 1px solid rgba(148, 163, 184, 0.15);
        }
        .hero-card, .message-card {
            background: rgba(15, 23, 42, 0.74);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 22px;
            box-shadow: 0 20px 45px rgba(2, 8, 23, 0.28);
            backdrop-filter: blur(14px);
        }
        .hero-card {
            padding: 1.4rem 1.5rem;
            margin-bottom: 1rem;
        }
        .hero-badge {
            display: inline-block;
            padding: 0.35rem 0.72rem;
            border-radius: 999px;
            border: 1px solid rgba(56, 189, 248, 0.35);
            background: rgba(56, 189, 248, 0.14);
            color: #c8f1ff;
            font-size: 0.78rem;
        }
        .hero-title {
            margin: 0.8rem 0 0.4rem 0;
            font-size: 2.35rem;
            line-height: 1.1;
            font-weight: 700;
            color: #f8fbff;
        }
        .hero-subtitle {
            margin: 0;
            max-width: 760px;
            color: #b3c5e5;
            line-height: 1.6;
        }
        .metric-card {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            margin-bottom: 0.8rem;
        }
        .metric-label {
            color: #8fa5ca;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            color: #f8fbff;
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 0.15rem;
        }
        .section-title {
            color: #eff6ff;
            font-size: 1.05rem;
            font-weight: 650;
            margin-bottom: 0.65rem;
        }
        .message-card {
            padding: 1rem 1.05rem;
            margin-bottom: 1rem;
        }
        .question-chip {
            display: inline-block;
            background: rgba(34, 197, 94, 0.12);
            color: #c8ffd9;
            border: 1px solid rgba(34, 197, 94, 0.26);
            border-radius: 999px;
            padding: 0.24rem 0.68rem;
            font-size: 0.76rem;
            margin-bottom: 0.55rem;
        }
        .answer-body {
            color: #e8eefc;
            font-size: 0.98rem;
            line-height: 1.7;
            white-space: pre-wrap;
        }
        .citations-line {
            margin-top: 0.8rem;
            color: #9cb2d8;
            font-size: 0.84rem;
        }
        @media (max-width: 900px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }
            .hero-card {
                padding: 1rem 1rem;
                border-radius: 18px;
            }
            .hero-title {
                font-size: 1.65rem;
                line-height: 1.15;
            }
            .hero-subtitle {
                font-size: 0.92rem;
            }
            .metric-card {
                padding: 0.8rem 0.9rem;
                border-radius: 14px;
            }
            .metric-value {
                font-size: 1.25rem;
            }
            .message-card {
                padding: 0.85rem 0.9rem;
                border-radius: 16px;
            }
            .answer-body {
                font-size: 0.93rem;
                line-height: 1.6;
            }
            [data-testid="stExpander"] {
                font-size: 0.92rem;
            }
            button[kind="primary"], button[kind="secondary"] {
                min-height: 2.75rem;
            }
        }
        @media (max-width: 640px) {
            .hero-badge {
                font-size: 0.72rem;
                padding: 0.28rem 0.56rem;
            }
            .hero-title {
                font-size: 1.42rem;
            }
            .section-title {
                font-size: 0.98rem;
            }
            .citations-line {
                font-size: 0.78rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">Multimodal RAG Workspace</div>
            <div class="hero-title">Chat with documents like a real product</div>
            <p class="hero-subtitle">
                Upload PDFs once, build the index, then keep asking follow-up questions with grounded answers,
                citations, and visual evidence from images or charts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(results: Sequence[RetrievalResult], heading: str = "Retrieved evidence") -> None:
    st.markdown(f'<div class="section-title">{heading}</div>', unsafe_allow_html=True)
    for result in results:
        with st.expander(f"{result.kind.title()} • {result.citation} • score {result.score:.3f}"):
            st.write(result.content)
            if result.image is not None:
                st.image(result.image, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Multimodal RAG Demo", layout="wide")
    render_app_chrome()
    render_sidebar()
    render_header()

    if "index_built" not in st.session_state:
        st.session_state.index_built = False
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = []
    if "image_chunks" not in st.session_state:
        st.session_state.image_chunks = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    overview_col, workspace_col = st.columns([0.85, 1.6], gap="large")
    ask_clicked = False
    query = ""

    with overview_col:
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        build_clicked = st.button("Build index", type="primary", disabled=not uploaded_files, use_container_width=True)
        reset_clicked = st.button("Reset workspace", use_container_width=True)
        st.caption(
            "Fast mode: captions are skipped by default and only the most useful images are indexed for quicker builds."
        )

        sample_docs = sorted({chunk.document_name for chunk in st.session_state.text_chunks})
        render_metric_card("Documents", str(len(sample_docs)))
        render_metric_card("Text Chunks", str(len(st.session_state.text_chunks)))
        render_metric_card("Image Chunks", str(len(st.session_state.image_chunks)))
        render_metric_card("Questions Asked", str(len(st.session_state.chat_history)))

        with st.container(border=True):
            st.markdown("**What makes this different**")
            st.markdown("- Hybrid retrieval across text and image evidence")
            st.markdown("- Page-level citations for grounded answers")
            st.markdown("- Multi-turn chat without rebuilding the index")

        with st.container(border=True):
            st.markdown("**Prompt ideas**")
            st.markdown("- `What is this document about?`")
            st.markdown("- `Summarize the main finding briefly.`")
            st.markdown("- `Which page explains the key method?`")

    if reset_clicked:
        st.session_state.index_built = False
        st.session_state.text_chunks = []
        st.session_state.image_chunks = []
        st.session_state.chat_history = []
        st.success("Index reset.")

    if build_clicked and uploaded_files:
        with st.spinner("Parsing documents and building embeddings..."):
            text_chunks, image_chunks = build_index(uploaded_files)
            st.session_state.text_chunks = text_chunks
            st.session_state.image_chunks = image_chunks
            st.session_state.index_built = True
        st.success(
            f"Indexed {len(uploaded_files)} file(s), {len(text_chunks)} text chunks, and {len(image_chunks)} images."
        )

    if st.session_state.index_built:
        with workspace_col:
            with st.form("question_form", clear_on_submit=True):
                query = st.text_input(
                    "Ask a question about the uploaded documents",
                    placeholder="What does the revenue chart show in Q4, and which page explains the trend?",
                )
                ask_clicked = st.form_submit_button("Ask", use_container_width=True)

        if ask_clicked and query.strip():
            with st.spinner("Retrieving evidence and generating answer..."):
                results = retrieve(query, st.session_state.text_chunks, st.session_state.image_chunks)
                answer = generate_answer(query, results)
                answer_citations = derive_answer_citations(query, results)
            st.session_state.chat_history.append(
                {
                    "question": query.strip(),
                    "answer": answer,
                    "citations": answer_citations,
                    "results": results,
                }
            )

        with workspace_col:
            with st.container(border=True):
                st.markdown("**Indexed documents**")
                indexed_docs = sorted({chunk.document_name for chunk in st.session_state.text_chunks})
                for document_name in indexed_docs:
                    st.markdown(f"- `{document_name}`")

            if st.session_state.chat_history:
                st.markdown('<div class="section-title">Conversation</div>', unsafe_allow_html=True)
                for item in reversed(st.session_state.chat_history):
                    citations = "; ".join(item.get("citations", format_citations(item["results"][:4])))
                    st.markdown(
                        f"""
                        <div class="message-card">
                            <div class="question-chip">Question</div>
                            <div><strong>{item['question']}</strong></div>
                            <div class="answer-body">{item['answer']}</div>
                            <div class="citations-line">Citations: {citations}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    render_results(item["results"], heading="Evidence trail")
            else:
                st.info("Your index is ready. Ask a question to start the conversation.")
    else:
        with workspace_col:
            st.info("Upload PDFs and click `Build index` to start chatting with your documents.")


if __name__ == "__main__":
    main()
