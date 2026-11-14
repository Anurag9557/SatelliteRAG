# Side by side: Ollama and vLLM

One page that sends the same question, with the same retrieved passages
and the same sampling settings, to an Ollama server and a vLLM server,
and shows the two answers and their timings next to each other.

    python demo/server.py aditya-l1 --ollama-model qwen2.5:7b-instruct-fp16

Then open http://127.0.0.1:8765. The page is served to this machine only.

## Where the two servers are

Retrieval runs here, on the machine that holds the vector store. The two
model servers can be anywhere this machine can reach:

    OLLAMA_HOST        default http://localhost:11434
    SATRAG_VLLM_URL    default http://localhost:8000/v1

PowerShell, with both servers on another machine:

    $env:OLLAMA_HOST     = "http://<server-ip>:<ollama-port>"
    $env:SATRAG_VLLM_URL = "http://<server-ip>:<vllm-port>/v1"
    $env:HF_HUB_OFFLINE  = "1"
    python demo\server.py aditya-l1 --ollama-model qwen2.5:7b-instruct-fp16

The retrieved passages travel to the model servers with each question,
so the servers have to be on a network the documents are allowed on.

Both servers should hold the same weights at the same precision. If they
do not, the page is comparing two models and not two runtimes.

## The three tabs

**One question.** Type a question or pick one. "Both at once" starts the
two servers together, so they share the card while they write. "One after
the other" gives each the card to itself, which is the fair reading of
speed per request.

**A batch.** Sends several questions to one server, then the same ones to
the other, with a set number in flight. This is where the runtimes
differ: Ollama answers as many at once as it has slots
(`OLLAMA_NUM_PARALLEL`), vLLM batches whatever arrives.

**Measured.** The figures from `bench/results-gpu-quiet/` and
`bench/results-gpu-paired/`. A live run on a shared card moves with
whatever else is running, so these are the numbers to quote.

## What is held equal

Greedy decoding, a fixed seed, no repetition penalty, the same token
limit and the same prompt. Each side gets one throwaway request when the
page starts, so neither pays for loading its model in front of people.
Identifiers that are not in the retrieved passages are removed from both
answers by the same guard the app uses, and the page shows how many.
