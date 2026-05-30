import io
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MODEL = "mistral-small-latest"
MAX_VISUAL_NODES = 55
MAX_CONTEXT_CHUNKS = 35
MAX_CONTEXT_CHARS = 24000
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "uses",
    "use",
    "used",
    "using",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass
class Chunk:
    id: str
    text: str
    page: int


def get_api_key(typed_api_key: str = "") -> str:
    if typed_api_key.strip():
        return typed_api_key.strip()
    try:
        return st.secrets["MISTRAL_API_KEY"]
    except Exception:
        return os.getenv("MISTRAL_API_KEY", "")


def extract_pdf_text(uploaded_file) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((page_number, text))
    return pages


def sentence_split(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_pages(pages: List[Tuple[int, str]], max_words: int = 95) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page_number, page_text in pages:
        current: List[str] = []
        current_words = 0
        for sentence in sentence_split(page_text):
            word_count = len(sentence.split())
            if current and current_words + word_count > max_words:
                chunk_id = f"chunk-{len(chunks) + 1}"
                chunks.append(Chunk(chunk_id, " ".join(current), page_number))
                current = []
                current_words = 0
            current.append(sentence)
            current_words += word_count
        if current:
            chunk_id = f"chunk-{len(chunks) + 1}"
            chunks.append(Chunk(chunk_id, " ".join(current), page_number))
    return chunks


def normalize_entity(entity: str) -> str:
    entity = re.sub(r"\s+", " ", entity).strip(" .,:;!?()[]{}")
    return entity.title()


def extract_entities(text: str, max_entities: int = 10) -> List[str]:
    capitalized = re.findall(r"\b(?:[A-Z][A-Za-z0-9+\-]*)(?:\s+[A-Z][A-Za-z0-9+\-]*)*\b", text)
    technical_terms = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+\-]{3,}(?:\s+[a-zA-Z][a-zA-Z0-9+\-]{3,}){0,2}\b", text)
    candidates = capitalized + technical_terms
    counts: Counter[str] = Counter()

    for candidate in candidates:
        words = [word for word in re.findall(r"[A-Za-z0-9+\-]+", candidate) if word.lower() not in STOPWORDS]
        if not words:
            continue
        if len(words) == 1 and len(words[0]) < 4:
            continue
        entity = normalize_entity(" ".join(words[:3]))
        if entity:
            counts[entity] += 1

    return [entity for entity, _ in counts.most_common(max_entities)]


def infer_relation(sentence: str, left: str, right: str) -> str:
    lowered = sentence.lower()
    patterns = [
        (" is a ", "is_a"),
        (" is an ", "is_a"),
        (" are a ", "is_a"),
        (" uses ", "uses"),
        (" use ", "uses"),
        (" stores ", "stores"),
        (" store ", "stores"),
        (" retrieves ", "retrieves"),
        (" retrieve ", "retrieves"),
        (" converts ", "converts"),
        (" convert ", "converts"),
        (" contains ", "contains"),
        (" includes ", "includes"),
        (" requires ", "requires"),
        (" depends on ", "depends_on"),
    ]
    for phrase, relation in patterns:
        if phrase in lowered:
            return relation
    left_pos = lowered.find(left.lower())
    right_pos = lowered.find(right.lower())
    if left_pos != -1 and right_pos != -1 and left_pos > right_pos:
        return "related_to"
    return "related_to"


def build_graph(chunks: List[Chunk]) -> nx.Graph:
    graph = nx.Graph()
    entity_to_chunks: Dict[str, set] = defaultdict(set)

    for chunk in chunks:
        entities = extract_entities(chunk.text)
        for entity in entities:
            graph.add_node(entity, kind="entity")
            entity_to_chunks[entity].add(chunk.id)

        for sentence in sentence_split(chunk.text):
            sentence_entities = [entity for entity in entities if entity.lower() in sentence.lower()]
            for index, left in enumerate(sentence_entities):
                for right in sentence_entities[index + 1 :]:
                    relation = infer_relation(sentence, left, right)
                    if graph.has_edge(left, right):
                        graph[left][right]["weight"] += 1
                        graph[left][right]["relations"].add(relation)
                        graph[left][right]["chunks"].add(chunk.id)
                    else:
                        graph.add_edge(
                            left,
                            right,
                            weight=1,
                            relations={relation},
                            chunks={chunk.id},
                        )

    for entity, chunk_ids in entity_to_chunks.items():
        graph.nodes[entity]["chunks"] = sorted(chunk_ids)
    return graph


def graph_to_display_edges(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for left, right, data in graph.edges(data=True):
        rows.append(
            {
                "Source": left,
                "Relation": ", ".join(sorted(data.get("relations", {"related_to"}))),
                "Target": right,
                "Evidence chunks": ", ".join(sorted(data.get("chunks", []))),
            }
        )
    return pd.DataFrame(rows)


def community_lookup(communities: List[List[str]]) -> Dict[str, int]:
    lookup = {}
    for index, community in enumerate(communities, start=1):
        for node in community:
            lookup[node] = index
    return lookup


def graph_nodes_table(graph: nx.Graph, communities: List[List[str]]) -> pd.DataFrame:
    lookup = community_lookup(communities)
    rows = []
    for node, data in graph.nodes(data=True):
        rows.append(
            {
                "Concept": node,
                "Community": lookup.get(node, "-"),
                "Connections": graph.degree(node),
                "Evidence chunks": ", ".join(data.get("chunks", [])),
            }
        )
    return pd.DataFrame(rows).sort_values(["Connections", "Concept"], ascending=[False, True])


def dot_escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def short_label(value: str, limit: int = 34) -> str:
    value = str(value)
    return value if len(value) <= limit else value[: limit - 3] + "..."


def build_graphviz_dot(graph: nx.Graph, communities: List[List[str]], nodes_to_show: Iterable[str]) -> str:
    shown_nodes = set(nodes_to_show)
    lookup = community_lookup(communities)
    colors = [
        "#f97316",
        "#22c55e",
        "#38bdf8",
        "#a78bfa",
        "#f43f5e",
        "#eab308",
        "#14b8a6",
        "#fb7185",
    ]
    lines = [
        "graph KnowledgeGraph {",
        '  graph [layout=dot, rankdir=LR, overlap=false, splines=true, bgcolor="transparent", pad="0.35"];',
        '  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11, margin="0.08,0.05", color="#1f2937", fontcolor="#f9fafb"];',
        '  edge [fontname="Arial", fontsize=9, color="#9ca3af", fontcolor="#d1d5db"];',
    ]

    for node in sorted(shown_nodes):
        community_index = lookup.get(node, 1)
        fill = colors[(community_index - 1) % len(colors)]
        lines.append(
            f'  "{dot_escape(node)}" [label="{dot_escape(short_label(node))}", fillcolor="{fill}"];'
        )

    for left, right, data in graph.edges(data=True):
        if left not in shown_nodes or right not in shown_nodes:
            continue
        relation = ", ".join(sorted(data.get("relations", {"related_to"})))
        lines.append(
            f'  "{dot_escape(left)}" -- "{dot_escape(right)}" [label="{dot_escape(short_label(relation, 24))}", penwidth="{1 + min(data.get("weight", 1), 4) * 0.35:.2f}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def most_connected_nodes(graph: nx.Graph, limit: int = MAX_VISUAL_NODES) -> List[str]:
    ranked = sorted(graph.degree, key=lambda item: item[1], reverse=True)
    return [node for node, _ in ranked[:limit]]


def cap_visual_nodes(graph: nx.Graph, nodes_to_show: Iterable[str], limit: int = MAX_VISUAL_NODES) -> Tuple[set, bool]:
    nodes = set(nodes_to_show)
    if len(nodes) <= limit:
        return nodes, False
    ranked = sorted(nodes, key=lambda node: graph.degree(node), reverse=True)
    return set(ranked[:limit]), True


def graph_json_export(graph: nx.Graph, communities: List[List[str]]) -> str:
    lookup = community_lookup(communities)
    payload = {
        "nodes": [
            {
                "id": node,
                "community": lookup.get(node),
                "chunks": data.get("chunks", []),
            }
            for node, data in graph.nodes(data=True)
        ],
        "edges": [
            {
                "source": left,
                "target": right,
                "relations": sorted(data.get("relations", {"related_to"})),
                "chunks": sorted(data.get("chunks", [])),
                "weight": data.get("weight", 1),
            }
            for left, right, data in graph.edges(data=True)
        ],
    }
    return json.dumps(payload, indent=2)


def detect_communities(graph: nx.Graph) -> List[List[str]]:
    if graph.number_of_nodes() == 0:
        return []
    communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
    return [sorted(list(community)) for community in communities]


def fit_vectorizer(chunks: List[Chunk]):
    texts = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def top_vector_chunks(query: str, chunks: List[Chunk], vectorizer, matrix, limit: int = 5) -> List[Tuple[Chunk, float]]:
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()
    ranked_indexes = np.argsort(scores)[::-1][:limit]
    return [(chunks[index], float(scores[index])) for index in ranked_indexes if scores[index] > 0]


def tokenize_terms(text: str) -> set:
    return {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9+\-]+", text)
        if len(term) > 2 and term.lower() not in STOPWORDS
    }


def choose_community(query_entities: List[str], communities: List[List[str]], query: str = "") -> List[str]:
    if not communities:
        return []
    query_entity_terms = {entity.lower() for entity in query_entities}
    query_words = set(query_entity_terms).union(tokenize_terms(query))
    best_score = -1
    best_community: List[str] = communities[0]
    for community in communities:
        community_terms = {entity.lower() for entity in community}
        community_words = set()
        for entity in community:
            community_words.update(tokenize_terms(entity))
        score = (len(query_entity_terms.intersection(community_terms)) * 4) + len(
            query_words.intersection(community_words)
        )
        if score > best_score:
            best_score = score
            best_community = community
    return best_community


def graph_paths(graph: nx.Graph, query_entities: List[str], community: List[str], max_paths: int = 4) -> List[List[str]]:
    community_set = set(community)
    matched = [node for node in graph.nodes if node.lower() in {entity.lower() for entity in query_entities}]
    if len(matched) < 2:
        matched = [node for node in graph.nodes if any(term.lower() in node.lower() for term in query_entities)]
    matched = [node for node in matched if node in community_set] or matched

    paths: List[List[str]] = []
    for index, left in enumerate(matched):
        for right in matched[index + 1 :]:
            try:
                path = nx.shortest_path(graph, left, right, weight=None)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if len(path) > 1:
                paths.append(path)
            if len(paths) >= max_paths:
                return paths
    return paths


def chunk_ids_from_paths(graph: nx.Graph, paths: Iterable[List[str]]) -> set:
    chunk_ids = set()
    for path in paths:
        for left, right in zip(path, path[1:]):
            chunk_ids.update(graph[left][right].get("chunks", set()))
    return chunk_ids


def chunk_sort_key(chunk: Chunk) -> Tuple[int, int]:
    match = re.search(r"(\d+)$", chunk.id)
    number = int(match.group(1)) if match else 0
    return chunk.page, number


def chunk_ids_from_community(graph: nx.Graph, community: List[str]) -> set:
    chunk_ids = set()
    for node in community:
        if node in graph.nodes:
            chunk_ids.update(graph.nodes[node].get("chunks", []))
    return chunk_ids


def limit_context_chunks(chunks: List[Chunk]) -> Tuple[List[Chunk], bool]:
    limited = []
    total_chars = 0
    was_limited = False
    for chunk in sorted(chunks, key=chunk_sort_key):
        next_chars = len(chunk.text)
        if len(limited) >= MAX_CONTEXT_CHUNKS or total_chars + next_chars > MAX_CONTEXT_CHARS:
            was_limited = True
            break
        limited.append(chunk)
        total_chars += next_chars
    return limited, was_limited


def retrieve_context(query: str, chunks: List[Chunk], graph: nx.Graph, communities: List[List[str]], vectorizer, matrix):
    query_entities = extract_entities(query, max_entities=6)
    community = choose_community(query_entities, communities, query)
    paths = graph_paths(graph, query_entities, community)
    evidence_chunk_ids = chunk_ids_from_paths(graph, paths)
    community_chunk_ids = chunk_ids_from_community(graph, community)

    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    community_chunks = [
        chunk_by_id[chunk_id] for chunk_id in community_chunk_ids if chunk_id in chunk_by_id
    ]
    graph_chunks = [chunk_by_id[chunk_id] for chunk_id in evidence_chunk_ids if chunk_id in chunk_by_id]
    vector_chunks = [chunk for chunk, _ in top_vector_chunks(query, chunks, vectorizer, matrix, limit=5)]

    merged: Dict[str, Chunk] = {}
    for chunk in community_chunks + graph_chunks + vector_chunks:
        merged[chunk.id] = chunk
    context_chunks, was_limited = limit_context_chunks(list(merged.values()))

    return {
        "query_entities": query_entities,
        "community": community,
        "community_chunk_count": len(community_chunks),
        "used_chunk_count": len(context_chunks),
        "was_limited": was_limited,
        "paths": paths,
        "chunks": context_chunks,
    }


def mistral_chat(messages: List[Dict[str, str]], api_key: str, model: str, temperature: float = 0.2) -> str:
    if not api_key:
        return "Add your MISTRAL_API_KEY in Streamlit secrets or your local environment to use the LLM."

    response = requests.post(
        MISTRAL_CHAT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()


def extract_json_object(text: str):
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def format_graph_paths(paths: List[List[str]]) -> str:
    if not paths:
        return "No direct graph path found. Use the retrieved chunks as evidence."
    return "\n".join(" -> ".join(path) for path in paths)


def context_text(retrieval: dict) -> str:
    return "\n\n".join(
        f"{chunk.id} (page {chunk.page}): {chunk.text}" for chunk in retrieval["chunks"]
    )


def retrieval_context_note(retrieval: dict) -> str:
    note = (
        f"Using {retrieval.get('used_chunk_count', len(retrieval['chunks']))} chunk(s) from "
        f"{retrieval.get('community_chunk_count', len(retrieval['chunks']))} chunk(s) linked to the selected community."
    )
    if retrieval.get("was_limited"):
        note += " The community was large, so the context was capped to fit the model."
    return note


def answer_question(query: str, retrieval: dict, api_key: str, model: str) -> str:
    context = context_text(retrieval)
    community_topics = ", ".join(retrieval.get("community", [])) or "No community found"
    context_note = retrieval_context_note(retrieval)
    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions from uploaded study material. Use only the provided context. "
                "Explain the graph reasoning chain briefly, then answer clearly. If evidence is missing, say so."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Detected entities: {', '.join(retrieval['query_entities']) or 'None'}\n\n"
                f"Selected community topics:\n{community_topics}\n\n"
                f"Context selection note:\n{context_note}\n\n"
                f"Graph path:\n{format_graph_paths(retrieval['paths'])}\n\n"
                f"Retrieved chunks:\n{context}"
            ),
        },
    ]
    return mistral_chat(messages, api_key, model)


def generate_exam(
    retrieval: dict,
    api_key: str,
    model: str,
    question_count: int,
    difficulty: str,
    mcq_count: int,
    short_count: int,
) -> dict:
    context = context_text(retrieval)
    community_topics = ", ".join(retrieval.get("community", [])) or "No community found"
    context_note = retrieval_context_note(retrieval)
    messages = [
        {
            "role": "system",
            "content": (
                "You create exam tests from study material. Use only the provided context. "
                "Return only valid JSON. Do not include markdown, comments, or extra text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Create an interactive exam with {question_count} total {difficulty} questions. "
                f"Include exactly {mcq_count} MCQ questions and exactly {short_count} descriptive questions.\n\n"
                "JSON schema:\n"
                "{\n"
                '  "title": "string",\n'
                '  "instructions": "string",\n'
                '  "questions": [\n'
                "    {\n"
                '      "id": "Q1",\n'
                '      "type": "mcq",\n'
                '      "question": "string",\n'
                '      "options": ["A", "B", "C", "D"],\n'
                '      "answer": "exact option text",\n'
                '      "explanation": "string",\n'
                '      "points": 1\n'
                "    },\n"
                "    {\n"
                '      "id": "Q2",\n'
                '      "type": "descriptive",\n'
                '      "question": "string",\n'
                '      "ideal_answer": "string",\n'
                '      "rubric": "string",\n'
                '      "points": 5\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                f"Selected community topics:\n{community_topics}\n\n"
                f"Context selection note:\n{context_note}\n\n"
                f"Important graph path:\n{format_graph_paths(retrieval['paths'])}\n\n"
                f"Study chunks:\n{context}"
            ),
        },
    ]
    raw_exam = mistral_chat(messages, api_key, model, temperature=0.4)
    exam = extract_json_object(raw_exam)
    exam["questions"] = normalize_exam_questions(exam.get("questions", []))
    return exam


def normalize_exam_questions(questions: List[dict]) -> List[dict]:
    normalized = []
    for index, question in enumerate(questions, start=1):
        item = dict(question)
        item["id"] = item.get("id") or f"Q{index}"
        item["type"] = "descriptive" if item.get("type") == "descriptive" else "mcq"
        item["question"] = item.get("question", "")
        if item["type"] == "mcq":
            item["options"] = [str(option) for option in item.get("options", [])][:4]
            if not item["options"]:
                item["options"] = ["Option A", "Option B", "Option C", "Option D"]
            item["answer"] = str(item.get("answer", ""))
            if item["answer"] not in item["options"]:
                item["answer"] = item["options"][0]
            item["points"] = int(item.get("points", 1) or 1)
        else:
            item["ideal_answer"] = str(item.get("ideal_answer", ""))
            item["rubric"] = str(item.get("rubric", "Award credit for correctness, completeness, and clarity."))
            item["points"] = int(item.get("points", 5) or 5)
        normalized.append(item)
    return normalized


def evaluate_descriptive_answers(exam: dict, answers: Dict[str, str], api_key: str, model: str) -> dict:
    descriptive = [
        {
            "id": question["id"],
            "question": question["question"],
            "ideal_answer": question.get("ideal_answer", ""),
            "rubric": question.get("rubric", ""),
            "points": question.get("points", 5),
            "student_answer": answers.get(question["id"], ""),
        }
        for question in exam.get("questions", [])
        if question.get("type") == "descriptive"
    ]
    if not descriptive:
        return {"results": []}

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict but fair exam evaluator. Grade only from the ideal answer and rubric. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate these descriptive answers. Give partial credit when correct ideas are present. "
                "Do not give credit for unsupported or incorrect claims.\n\n"
                "Return JSON in this schema:\n"
                '{ "results": [ { "id": "Q1", "score": 0, "max_points": 5, "feedback": "string", "correct_answer": "string" } ] }\n\n'
                f"Answers to evaluate:\n{json.dumps(descriptive, indent=2)}"
            ),
        },
    ]
    raw_result = mistral_chat(messages, api_key, model, temperature=0.1)
    return extract_json_object(raw_result)


def score_exam(exam: dict, answers: Dict[str, str], api_key: str, model: str) -> dict:
    results = []
    total_score = 0.0
    total_points = 0.0

    for question in exam.get("questions", []):
        if question.get("type") != "mcq":
            continue
        max_points = float(question.get("points", 1))
        student_answer = answers.get(question["id"], "")
        correct = student_answer == question.get("answer")
        score = max_points if correct else 0.0
        total_score += score
        total_points += max_points
        results.append(
            {
                "id": question["id"],
                "type": "mcq",
                "score": score,
                "max_points": max_points,
                "student_answer": student_answer,
                "correct_answer": question.get("answer", ""),
                "feedback": question.get("explanation", ""),
            }
        )

    descriptive_result = evaluate_descriptive_answers(exam, answers, api_key, model)
    for item in descriptive_result.get("results", []):
        score = float(item.get("score", 0) or 0)
        max_points = float(item.get("max_points", 0) or 0)
        total_score += score
        total_points += max_points
        results.append(
            {
                "id": item.get("id", ""),
                "type": "descriptive",
                "score": score,
                "max_points": max_points,
                "student_answer": answers.get(item.get("id", ""), ""),
                "correct_answer": item.get("correct_answer", ""),
                "feedback": item.get("feedback", ""),
            }
        )

    percentage = round((total_score / total_points) * 100, 2) if total_points else 0
    return {
        "total_score": round(total_score, 2),
        "total_points": round(total_points, 2),
        "percentage": percentage,
        "results": sorted(results, key=lambda item: item["id"]),
    }


def render_graph_summary(graph: nx.Graph, communities: List[List[str]], chunks: List[Chunk]) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Concepts", graph.number_of_nodes())
    col2.metric("Relations", graph.number_of_edges())
    col3.metric("Communities", len(communities))

    if graph.number_of_nodes() == 0:
        st.info("No relationships were detected yet. Try uploading a PDF with more connected concepts.")
        return

    node_table = graph_nodes_table(graph, communities)
    edges = graph_to_display_edges(graph)
    chunk_by_id = {chunk.id: chunk for chunk in chunks}

    st.markdown("#### Visual Knowledge Graph")
    view_mode = st.radio(
        "Graph view",
        ["Concept neighborhood", "Community view", "Top concepts"],
        horizontal=True,
        index=0,
    )

    nodes_to_show = set(most_connected_nodes(graph))
    if view_mode == "Community view" and communities:
        labels = [
            f"Community {index + 1}: {', '.join(community[:5])}{'...' if len(community) > 5 else ''}"
            for index, community in enumerate(communities)
        ]
        selected_label = st.selectbox("Select community", labels)
        selected_index = labels.index(selected_label)
        nodes_to_show = set(communities[selected_index])
    elif view_mode == "Concept neighborhood":
        concepts = node_table["Concept"].tolist()
        selected_concept = st.selectbox("Select concept", concepts)
        radius = st.slider("Relationship distance", 1, 2, 1)
        nodes_to_show = set(nx.ego_graph(graph, selected_concept, radius=radius).nodes)

    original_count = len(nodes_to_show)
    nodes_to_show, was_capped = cap_visual_nodes(graph, nodes_to_show)
    if was_capped:
        st.warning(
            f"Showing the {len(nodes_to_show)} most connected concepts from {original_count}. Full data is still available in the tables and downloads below."
        )

    dot = build_graphviz_dot(graph, communities, nodes_to_show)
    st.graphviz_chart(dot, use_container_width=True)

    export_col1, export_col2 = st.columns(2)
    export_col1.download_button(
        "Download Graph DOT",
        data=dot,
        file_name="knowledge_graph.dot",
        mime="text/vnd.graphviz",
        use_container_width=True,
    )
    export_col2.download_button(
        "Download Graph JSON",
        data=graph_json_export(graph, communities),
        file_name="knowledge_graph.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("#### Concept Explorer")
    selected_node = st.selectbox("Inspect a concept", node_table["Concept"].tolist(), key="inspect_concept")
    neighbors = sorted(graph.neighbors(selected_node))
    st.write(f"**{selected_node}** connects to {len(neighbors)} concept(s).")
    if neighbors:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Connected concept": neighbor,
                        "Relation": ", ".join(sorted(graph[selected_node][neighbor].get("relations", []))),
                        "Evidence chunks": ", ".join(sorted(graph[selected_node][neighbor].get("chunks", []))),
                    }
                    for neighbor in neighbors
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    evidence_ids = graph.nodes[selected_node].get("chunks", [])
    if evidence_ids:
        with st.expander("Show source chunks for this concept", expanded=False):
            for chunk_id in evidence_ids:
                chunk = chunk_by_id.get(chunk_id)
                if chunk:
                    st.markdown(f"**{chunk.id} · page {chunk.page}**")
                    st.write(chunk.text)

    st.markdown("#### All Concepts")
    st.dataframe(node_table, use_container_width=True, hide_index=True)

    st.markdown("#### All Relations")
    if not edges.empty:
        st.dataframe(edges, use_container_width=True, hide_index=True)

    if communities:
        st.markdown("#### Communities")
        community_rows = [
            {
                "Community": index + 1,
                "Concept count": len(community),
                "Concepts": ", ".join(community),
            }
            for index, community in enumerate(communities)
        ]
        st.dataframe(pd.DataFrame(community_rows), use_container_width=True, hide_index=True)


def process_pdf(uploaded_file):
    pages = extract_pdf_text(uploaded_file)
    chunks = chunk_pages(pages)
    graph = build_graph(chunks)
    communities = detect_communities(graph)
    vectorizer, matrix = fit_vectorizer(chunks)
    return {
        "file_name": uploaded_file.name,
        "pages": pages,
        "chunks": chunks,
        "graph": graph,
        "communities": communities,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }


def main() -> None:
    st.set_page_config(page_title="GraphRAG Exam Test Builder", page_icon="G", layout="wide")
    st.title("GraphRAG Exam Test Builder")
    st.caption("Upload a PDF, build a concept graph, ask questions, and generate exam tests with Mistral AI.")

    with st.sidebar:
        st.header("Settings")
        model = st.text_input("Mistral model", value=DEFAULT_MODEL)
        typed_api_key = st.text_input(
            "Mistral API key",
            type="password",
            placeholder="Paste key for local testing",
        )
        api_key = get_api_key(typed_api_key)
        if api_key:
            st.success("Mistral API key loaded")
        else:
            st.warning("Set MISTRAL_API_KEY or paste a key above to enable answers and exam generation")
        st.divider()
        st.markdown("**Pipeline**")
        st.markdown(
            "PDF upload -> text extraction -> chunking -> entity/relation extraction -> graph communities -> graph + vector retrieval -> Mistral answer."
        )

    uploaded_file = st.file_uploader("Upload a study PDF", type=["pdf"])
    if not uploaded_file:
        st.info("Upload a PDF to build the knowledge graph and start generating exam tests.")
        return

    file_signature = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("file_signature") != file_signature:
        with st.spinner("Reading PDF and building the graph..."):
            st.session_state["rag_state"] = process_pdf(uploaded_file)
            st.session_state["file_signature"] = file_signature

    rag_state = st.session_state["rag_state"]
    chunks: List[Chunk] = rag_state["chunks"]
    graph: nx.Graph = rag_state["graph"]
    communities: List[List[str]] = rag_state["communities"]

    st.subheader(rag_state["file_name"])
    st.write(f"Extracted {len(chunks)} chunks from {len(rag_state['pages'])} page(s).")

    tab_ask, tab_exam, tab_graph, tab_chunks = st.tabs(["Ask", "Generate Exam", "Knowledge Graph", "Chunks"])

    with tab_ask:
        query = st.text_input("Question", placeholder="Example: How does RAG use Pinecone?")
        if st.button("Answer", type="primary", disabled=not query):
            retrieval = retrieve_context(
                query,
                chunks,
                graph,
                communities,
                rag_state["vectorizer"],
                rag_state["matrix"],
            )
            st.session_state["last_retrieval"] = retrieval
            with st.spinner("Mistral is reasoning over the graph and chunks..."):
                try:
                    st.markdown(answer_question(query, retrieval, api_key, model))
                except requests.HTTPError as error:
                    st.error(f"Mistral API error: {error.response.text}")
                except Exception as error:
                    st.error(f"Could not generate answer: {error}")

        retrieval = st.session_state.get("last_retrieval")
        if retrieval:
            with st.expander("Retrieved reasoning context", expanded=False):
                st.write("Detected entities:", retrieval["query_entities"])
                st.write("Selected community topics:", retrieval.get("community", []))
                st.write(
                    f"LLM context: {retrieval.get('used_chunk_count', len(retrieval['chunks']))} chunk(s) used from "
                    f"{retrieval.get('community_chunk_count', len(retrieval['chunks']))} community-linked chunk(s)."
                )
                if retrieval.get("was_limited"):
                    st.warning("The selected community was large, so context was capped to fit the model.")
                st.code(format_graph_paths(retrieval["paths"]))
                for chunk in retrieval["chunks"]:
                    st.markdown(f"**{chunk.id} · page {chunk.page}**")
                    st.write(chunk.text)

    with tab_exam:
        topic = st.text_input("Exam topic or instruction", placeholder="Example: Test me on RAG and vector databases")
        col1, col2, col3 = st.columns(3)
        question_count = col1.slider("Total questions", 3, 20, 8)
        mcq_count = col2.slider("MCQs", 1, question_count - 1, min(5, question_count - 1))
        short_count = question_count - mcq_count
        difficulty = col3.selectbox("Difficulty", ["mixed", "easy", "medium", "hard"])
        st.caption(f"This test will include {mcq_count} MCQ question(s) and {short_count} descriptive question(s).")

        generate_col, reset_col = st.columns([1, 1])
        if generate_col.button("Create Test", type="primary", disabled=not topic):
            retrieval = retrieve_context(
                topic,
                chunks,
                graph,
                communities,
                rag_state["vectorizer"],
                rag_state["matrix"],
            )
            st.session_state["last_exam_retrieval"] = retrieval
            with st.spinner("Building your test from the selected graph community..."):
                try:
                    exam = generate_exam(
                        retrieval,
                        api_key,
                        model,
                        question_count,
                        difficulty,
                        mcq_count,
                        short_count,
                    )
                    st.session_state["active_exam"] = exam
                    st.session_state["exam_score"] = None
                except requests.HTTPError as error:
                    st.error(f"Mistral API error: {error.response.text}")
                except json.JSONDecodeError:
                    st.error("Mistral returned an exam, but it was not valid JSON. Click Create Test again.")
                except Exception as error:
                    st.error(f"Could not generate exam: {error}")

        if reset_col.button("Reset Test"):
            st.session_state["active_exam"] = None
            st.session_state["exam_score"] = None

        exam_retrieval = st.session_state.get("last_exam_retrieval")
        if exam_retrieval:
            with st.expander("Exam context sent to LLM", expanded=False):
                st.write("Selected community topics:", exam_retrieval.get("community", []))
                st.write(
                    f"LLM context: {exam_retrieval.get('used_chunk_count', len(exam_retrieval['chunks']))} chunk(s) used from "
                    f"{exam_retrieval.get('community_chunk_count', len(exam_retrieval['chunks']))} community-linked chunk(s)."
                )
                if exam_retrieval.get("was_limited"):
                    st.warning("The selected community was large, so context was capped to fit the model.")
                st.code(format_graph_paths(exam_retrieval["paths"]))

        active_exam = st.session_state.get("active_exam")
        if active_exam:
            st.divider()
            st.subheader(active_exam.get("title", "Generated Test"))
            st.write(active_exam.get("instructions", "Answer all questions, then submit your test."))

            with st.form("exam_attempt_form"):
                user_answers = {}
                for index, question in enumerate(active_exam.get("questions", []), start=1):
                    question_id = question["id"]
                    points = question.get("points", 1)
                    st.markdown(f"**{index}. {question.get('question', '')}**")
                    st.caption(f"{question.get('type', '').title()} · {points} point(s)")

                    if question.get("type") == "mcq":
                        options = question.get("options", [])
                        user_answers[question_id] = st.radio(
                            "Choose one answer",
                            options,
                            key=f"answer_{question_id}",
                            index=None,
                        )
                    else:
                        user_answers[question_id] = st.text_area(
                            "Your answer",
                            key=f"answer_{question_id}",
                            height=120,
                            placeholder="Write your answer in your own words.",
                        )
                    st.write("")

                submitted = st.form_submit_button("Submit Test", type="primary")

            if submitted:
                missing = [
                    question["id"]
                    for question in active_exam.get("questions", [])
                    if not user_answers.get(question["id"])
                ]
                if missing:
                    st.warning(f"Please answer all questions before submitting: {', '.join(missing)}")
                else:
                    with st.spinner("Checking your answers and evaluating descriptive responses..."):
                        try:
                            st.session_state["exam_score"] = score_exam(active_exam, user_answers, api_key, model)
                        except requests.HTTPError as error:
                            st.error(f"Mistral API error while evaluating: {error.response.text}")
                        except json.JSONDecodeError:
                            st.error("Mistral returned evaluation text, but it was not valid JSON. Submit again.")
                        except Exception as error:
                            st.error(f"Could not evaluate exam: {error}")

            score = st.session_state.get("exam_score")
            if score:
                st.divider()
                st.subheader("Result")
                result_col1, result_col2, result_col3 = st.columns(3)
                result_col1.metric("Score", f"{score['total_score']} / {score['total_points']}")
                result_col2.metric("Percentage", f"{score['percentage']}%")
                result_col3.metric(
                    "Status",
                    "Passed" if score["percentage"] >= 50 else "Needs practice",
                )

                for result in score["results"]:
                    with st.expander(
                        f"{result['id']} · {result['score']} / {result['max_points']} point(s)",
                        expanded=True,
                    ):
                        st.write("**Your answer:**")
                        st.write(result.get("student_answer") or "No answer")
                        st.write("**Correct / ideal answer:**")
                        st.write(result.get("correct_answer") or "Not provided")
                        st.write("**Feedback:**")
                        st.write(result.get("feedback") or "No feedback")

    with tab_graph:
        render_graph_summary(graph, communities, chunks)

    with tab_chunks:
        chunk_rows = [
            {"Chunk": chunk.id, "Page": chunk.page, "Text": chunk.text}
            for chunk in chunks
        ]
        st.dataframe(pd.DataFrame(chunk_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
