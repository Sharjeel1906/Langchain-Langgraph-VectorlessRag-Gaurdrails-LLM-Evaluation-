import os
import re
import asyncio
import ast
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

load_dotenv()


async def score_with_retry(coro_fn, *args, max_retries=6, **kwargs):
    """Call an ascore-style coroutine, retrying on 429 rate limits with backoff."""
    for attempt in range(max_retries):
        try:
            return await coro_fn(*args, **kwargs)
        except RateLimitError as e:
            wait = 3.0
            match = re.search(r"try again in ([\d.]+)s", str(e))
            if match:
                wait = float(match.group(1)) + 0.5
            print(f"Rate limited, waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(wait)
        except Exception as e:
            if "rate_limit_exceeded" in str(e) or "429" in str(e):
                wait = 3.0
                match = re.search(r"try again in ([\d.]+)s", str(e))
                if match:
                    wait = float(match.group(1)) + 0.5
                print(f"Rate limited, waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded due to rate limiting")


async def main():

    # ---------------------------------
    # 1. Load dataset
    # ---------------------------------
    csv_path = Path(__file__).resolve().parent / "test_dataset.csv"
    df = pd.read_csv(csv_path)
    df["retrieved_contexts"] = df["retrieved_contexts"].apply(ast.literal_eval)

    # ---------------------------------
    # 2. Create OpenAI-compatible client pointed at Groq (for LLM judging)
    # ---------------------------------
    client = AsyncOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    llm = llm_factory(
        "llama-3.3-70b-versatile",
        provider="openai",
        client=client
    )

    # ---------------------------------
    # 3. Local embeddings (Groq has no embeddings API)
    # ---------------------------------
    embeddings = embedding_factory(
        "huggingface",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------------------------
    # 4. Metrics
    # ---------------------------------
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    scores = []

    for _, row in df.iterrows():

        question = row["user_input"]
        context = row["retrieved_contexts"]
        answer = row["response"]
        reference = row["reference"]

        try:
            # Run sequentially (not concurrently) to respect Groq's TPM limit
            faith_result = await score_with_retry(
                faithfulness.ascore,
                user_input=question,
                response=answer,
                retrieved_contexts=context
            )
            await asyncio.sleep(1)

            relevancy_result = await score_with_retry(
                answer_relevancy.ascore,
                user_input=question,
                response=answer
            )
            await asyncio.sleep(1)

            precision_result = await score_with_retry(
                context_precision.ascore,
                user_input=question,
                retrieved_contexts=context,
                reference=reference
            )
            await asyncio.sleep(1)

            recall_result = await score_with_retry(
                context_recall.ascore,
                user_input=question,
                retrieved_contexts=context,
                reference=reference
            )
            await asyncio.sleep(1)

        except Exception as e:
            print(f"Error scoring row '{question}': {e}")
            continue

        scores.append({
            "question": question,
            "faithfulness": faith_result.value,
            "answer_relevancy": relevancy_result.value,
            "context_precision": precision_result.value,
            "context_recall": recall_result.value,
        })

    results_df = pd.DataFrame(scores)

    print("\nResults:")
    print(results_df.to_string(index=False))

    print("\nAverage Scores:")
    print(f"Faithfulness:       {results_df['faithfulness'].mean():.4f}")
    print(f"Answer Relevancy:   {results_df['answer_relevancy'].mean():.4f}")
    print(f"Context Precision:  {results_df['context_precision'].mean():.4f}")
    print(f"Context Recall:     {results_df['context_recall'].mean():.4f}")

    output_path = Path(__file__).resolve().parent / "ragas_results.csv"
    results_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    asyncio.run(main())