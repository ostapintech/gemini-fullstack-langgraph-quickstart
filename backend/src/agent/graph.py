import os
from pathlib import Path
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agent.tools_and_schemas import SearchQueryList
from agent.state import OverallState, QueryGenerationState, WebSearchState
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    local_extractor_instructions,
    answer_instructions,
)
from agent.utils import get_research_topic

load_dotenv()


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """Генерує ключові слова для пошуку в локальних файлах."""
    configurable = Configuration.from_runnable_config(config)

    llm = init_chat_model(
        model=configurable.query_generator_model,
        temperature=0,  # Для стабільності пошуку краще 0
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    formatted_prompt = query_writer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        number_queries=configurable.number_of_initial_queries,
    )

    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def local_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    target_dir = Path(configurable.local_dir)

    # Динамічно отримуємо ключові слова із запиту
    # (прибираємо знаки пунктуації та короткі слова)
    raw_query = state["search_query"].lower()
    keywords = [word.strip('?,.!-') for word in raw_query.split() if len(word) > 2]

    MAX_CHARS = 12000
    scored_files = []

    print(f"--- DEBUG: Dynamic search for keywords: {keywords} ---")

    for file_path in target_dir.rglob("*.md"):
        try:
            score = 0

            relative_path = str(file_path.relative_to(target_dir)).lower()
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            content_lower = content.lower()

            for kw in keywords:
                # 1. Найвища вага: ключове слово в назві файлу або шлях до нього
                if kw in relative_path:
                    score += 25

                    # 2. Висока вага: ключове слово в заголовку Markdown (# або ##)
                lines = content_lower.split('\n')
                for line in lines:
                    if line.strip().startswith('#') and kw in line:
                        score += 15

                # 3. Середня вага: частота згадування в самому тексті
                count = content_lower.count(kw)
                if count > 0:
                    score += min(count * 2, 20)  # максимум 20 балів за текст

            if score > 5:
                scored_files.append({
                    "score": score,
                    "path": str(file_path),
                    "content": content,
                    "name": str(file_path.relative_to(target_dir))
                })
        except:
            continue

    scored_files.sort(key=lambda x: x["score"], reverse=True)

    context_parts = []
    sources_used = []
    current_length = 0

    if not scored_files:
        return {"sources_gathered": [], "web_research_result": ["NO RELEVANT LOCAL DATA FOUND"]}

    for item in scored_files[:4]:
        if current_length >= MAX_CHARS: break
        remaining = MAX_CHARS - current_length
        chunk = item["content"][:remaining]

        context_parts.append(f"--- SOURCE: {item['name']} ---\n{chunk}")
        sources_used.append({"short_url": item['name'], "value": item['path']})
        current_length += len(chunk)

    return {
        "sources_gathered": sources_used,
        "search_query": [state["search_query"]],
        "web_research_result": ["\n\n".join(context_parts)],
    }

def finalize_answer(state: OverallState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    formatted_prompt = answer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state.get("web_research_result", [])),
    )

    llm = init_chat_model(model=reasoning_model, temperature=0)
    result = llm.invoke(formatted_prompt)

    # Фільтрація джерел, які реально згадані в тексті [filename.md]
    unique_sources = []
    for source in state.get("sources_gathered", []):
        if source["short_url"] in result.content:
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


def continue_to_local_research(state: QueryGenerationState):
    """Відправляє кожен пошуковий запит паралельно у вузол пошуку."""
    return [
        Send("local_research", {"search_query": q, "id": i})
        for i, q in enumerate(state["search_query"])
    ]


builder = StateGraph(OverallState, config_schema=Configuration)

builder.add_node("generate_query", generate_query)
builder.add_node("local_research", local_research)
builder.add_node("finalize_answer", finalize_answer)

builder.add_edge(START, "generate_query")
builder.add_conditional_edges(
    "generate_query", continue_to_local_research, ["local_research"]
)
builder.add_edge("local_research", "finalize_answer")
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="local-research-agent")