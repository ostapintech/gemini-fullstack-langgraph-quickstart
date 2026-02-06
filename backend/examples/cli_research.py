import argparse
import os
from langchain_core.messages import HumanMessage
from agent.graph import graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph research agent")
    parser.add_argument("question", help="Research question")
    parser.add_argument(
        "--dir",
        default="/Users/ostapzherebtsov/Desktop/langgraph_old_docs",
        type=str,
        help="Шлях до папки з .md файлами"
    )
    parser.add_argument("--initial-queries", type=int, default=2)
    parser.add_argument("--reasoning-model", default="groq:llama-3.3-70b-versatile")

    args = parser.parse_args()

    config = {"configurable": {
        "local_dir": args.dir,
        "query_generator_model": args.reasoning_model,
        "answer_model": args.reasoning_model,
        "number_of_initial_queries": args.initial_queries
    }}

    state = {
        "messages": [HumanMessage(content=args.question)],
        "initial_search_query_count": args.initial_queries,
    }

    print(f"--- Starting research in: {args.dir} ---")

    result = graph.invoke(state, config=config)

    # Вивід основної відповіді
    messages = result.get("messages", [])
    if messages:
        print("\n" + "=" * 50)
        print("TECHNICAL REPORT")
        print("=" * 50)
        print(messages[-1].content)

    # Вивід використаних джерел
    sources = result.get("sources_gathered", [])
    if sources:
        print("\n" + "=" * 50)
        print("USED SOURCES")
        print("=" * 50)

        unique_source_names = sorted(list(set([s['short_url'] for s in sources])))
        for idx, name in enumerate(unique_source_names, 1):
            print(f"{idx}. {name}")
    else:
        print("\n--- No specific local sources were cited in the answer ---")


if __name__ == "__main__":
    main()