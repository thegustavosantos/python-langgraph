# python-langgraph

Install dependencies with `python -m pip install -r requirements.txt`.

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`, then run
`python main.py`.

The agent uses the OpenAI Responses API and defaults to `gpt-6-luna`.
Set `OPENAI_MODEL` in `.env` to select another model available to your account.

Keep the script named `main.py`; naming it `langgraph.py` shadows the
installed `langgraph` package.
